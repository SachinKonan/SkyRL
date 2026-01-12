# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SearchArxiv Environment for paper acceptance prediction with arXiv search.

Supports two search modes:
- Semantic search: <ssearch> query </ssearch>
- Author search: <asearch> author1,author2 </asearch>
"""

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType
from typing import Any, Dict, Optional, Tuple, List
from skyrl_gym.envs.search_arxiv.utils import compute_score, extract_boxed_answer
from skyrl_gym.tools import SearchArxivToolGroup
import re
from omegaconf import DictConfig


class SearchArxivEnv(BaseTextEnv):
    """
    Environment for arXiv paper search and acceptance prediction tasks.

    Supports semantic search (<ssearch>) and author search (<asearch>).
    """

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__()

        # Get ground truth from reward_model (not reward_spec)
        assert "reward_model" in extras, "reward_model field is required in extras"
        assert "ground_truth" in extras["reward_model"], "ground_truth is required in reward_model field"
        self.ground_truth = extras["reward_model"]["ground_truth"]

        # Max turns (3 by default to keep token count reasonable)
        self.max_turns = extras.get("max_turns", 3)

        # Get year from extra_info and calculate cutoff date
        # Dec 15 of year before conference ensures temporal correctness
        extra_info = extras.get("extra_info", {})
        year = extra_info.get("year", None)
        upper_bound_datetime = None
        if year:
            upper_bound_datetime = f"{year - 1}-12-15"

        # Initialize the search tool group
        self.tool_group = SearchArxivToolGroup(
            search_url=env_config.get("search_url", "http://127.0.0.1:8000/retrieve"),
            topk=env_config.get("topk", 5),
            timeout=env_config.get("timeout", 30),
            log_requests=env_config.get("log_requests", True),
            upper_bound_datetime=upper_bound_datetime,
        )
        self.init_tool_groups([self.tool_group])

        # Chat history
        # role (user, assistant), content (tool observation or LLM response)
        self.chat_history: ConversationType = []

        # Track search calls for metrics
        self.num_search_calls = 0
        # Track the model's prediction (Accept/Reject) for majority vote
        self.prediction: Optional[str] = None

    def _extract_title_from_prompt(self, chat_history: ConversationType) -> Optional[str]:
        """
        Extract paper title from the user message in chat history.

        The prompt format is: "# {Title}\n\n# Abstract\n..."
        The title is the first line starting with "# " that doesn't contain "Abstract".

        Args:
            chat_history: List of message dicts with 'role' and 'content'.

        Returns:
            The paper title if found, None otherwise.
        """
        for msg in chat_history:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                lines = content.split("\n")
                for line in lines:
                    if line.startswith("# ") and "Abstract" not in line:
                        return line[2:].strip()  # Remove "# " prefix
        return None

    def init(self, chat_history: ConversationType) -> Tuple[ConversationType, Dict[str, Any]]:
        """
        Initialize the environment with the chat history.

        Extracts the paper title from the prompt and sets it on the tool group
        to filter out the same paper from search results.

        Args:
            chat_history: Initial chat history with the paper content.

        Returns:
            Tuple of (updated_chat_history, env_info).
        """
        # Call parent init first
        result = super().init(chat_history)

        # Extract paper title and set it on the tool group for filtering
        paper_title = self._extract_title_from_prompt(chat_history)
        if paper_title:
            self.tool_group.set_paper_title(paper_title)

        return result

    def _parse_action(self, action: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse the action for semantic search or author search queries.

        Args:
            action: The action string from the LLM.

        Returns:
            Tuple of (ssearch_query, asearch_query), both None if no search found.
        """
        ssearch_match = None
        asearch_match = None

        if "<ssearch>" in action and "</ssearch>" in action:
            ssearch_match = re.search(r"<ssearch>(.*?)</ssearch>", action, re.DOTALL)

        if "<asearch>" in action and "</asearch>" in action:
            asearch_match = re.search(r"<asearch>(.*?)</asearch>", action, re.DOTALL)

        ssearch_query = ssearch_match.group(1).strip() if ssearch_match else None
        asearch_query = asearch_match.group(1).strip() if asearch_match else None

        return ssearch_query, asearch_query

    def _get_reward(self, action: str, done: bool) -> float:
        """
        Compute the reward for the current action.

        Args:
            action: The action string from the LLM.
            done: Whether the episode is done.

        Returns:
            The reward value.
        """
        if done:
            # Concat all chat history into a single string and compute reward
            chat_history_str = "".join([item["content"] for item in self.chat_history])
            return compute_score(chat_history_str, self.ground_truth, score=1.0, format_score=0.1)
        else:
            # No reward for intermediate steps
            return 0

    def _is_done(self, action: str) -> bool:
        """
        Check if the episode is done.

        Args:
            action: The action string from the LLM.

        Returns:
            True if the episode is done, False otherwise.
        """
        if self.turns >= self.max_turns:
            return True
        return "<answer>" in action and "</answer>" in action

    def _validate_action(self, action: str):
        """
        Validate that stop tags are at the end of the action.

        Args:
            action: The action string from the LLM.

        Raises:
            AssertionError if stop tags are not at the end.
        """
        stop_tags = ["</ssearch>", "</asearch>", "</answer>"]
        for tag in stop_tags:
            if tag in action:
                assert action.split(tag, 1)[1] == "", (
                    f"{tag} detected in the response but it is not the last string generated. "
                    f"Use {stop_tags} as stop strings in the configuration."
                )

    def _execute_tool(self, tool_group_name: str, tool_name: str, tool_input: Any) -> str:
        """
        Execute a tool and wrap the output with information tags.

        Args:
            tool_group_name: Name of the tool group.
            tool_name: Name of the tool to execute.
            tool_input: Input for the tool.

        Returns:
            The tool output wrapped with <information> tags.
        """
        tool_output = super()._execute_tool(tool_group_name, tool_name, tool_input)
        return "\n<information>" + tool_output + "</information>\n"

    def step(self, action: str) -> BaseTextEnvStepOutput:
        """
        Execute one step in the environment.

        Args:
            action: The action string from the LLM.

        Returns:
            BaseTextEnvStepOutput with observations, reward, done, and metadata.
        """
        self.turns += 1
        self._validate_action(action)
        self.chat_history.append({"role": "assistant", "content": action})

        error = None
        done = self._is_done(action)
        reward = self._get_reward(action, done)

        if done:
            # Extract prediction for majority vote metrics
            chat_history_str = "".join([item["content"] for item in self.chat_history])
            self.prediction = extract_boxed_answer(chat_history_str)
            return BaseTextEnvStepOutput(observations=[], reward=reward, done=done, metadata={})

        try:
            ssearch_query, asearch_query = self._parse_action(action)
            observation = None

            if ssearch_query:
                observation = self._execute_tool("SearchArxivToolGroup", "semantic_search", (ssearch_query,))
                self.num_search_calls += 1
            elif asearch_query:
                observation = self._execute_tool("SearchArxivToolGroup", "author_search", (asearch_query,))
                self.num_search_calls += 1

            remaining_turns = self.max_turns - self.turns
            if remaining_turns == 1:
                turn_info = "(Last Turn, no more searching allowed, think concisely, and give answer now.)"
            else:
                turn_info = f"(Turns remaining: {remaining_turns})"

            if not ssearch_query and not asearch_query:
                # No valid search tag found, check if <answer> was attempted but malformed
                has_answer = "<answer>" in action
                has_ssearch = "<ssearch>" in action
                has_asearch = "<asearch>" in action

                if not (has_answer or has_ssearch or has_asearch):
                    # No valid tags at all - guide the model
                    observation = "\n<information>Your answer should end with <answer>...</answer>, <ssearch>...</ssearch>.</information>\n"
                else:
                    observation = None
        except Exception as e:
            error = str(e)
            observation = None

        if observation:
            observation = observation.replace("</information>", f"\n{turn_info}</information>")

        # Wrap the observation properly as a message
        if observation:
            new_obs = {"role": "user", "content": observation}
        elif error:
            # Give error as observation if any
            new_obs = {"role": "user", "content": f"\n<information>Error: {error}</information>\n"}
        else:
            new_obs = None

        info = {
            "tool_group": "SearchArxivToolGroup",
            "tool_name": "semantic_search" if ssearch_query else ("author_search" if asearch_query else None),
            "tool_input": ssearch_query or asearch_query,
        }

        # Update chat history
        if new_obs:
            self.chat_history.append(new_obs)

        return BaseTextEnvStepOutput(
            observations=[new_obs] if new_obs else [],
            reward=reward,
            done=done,
            metadata=info,
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Return environment-specific metrics for the episode."""
        return {
            "num_search_calls": self.num_search_calls,
            "turns": self.turns,
            "prediction": self.prediction,  # Accept, Reject, or None
            "ground_truth": self.ground_truth.get("target", [None])[0] if self.ground_truth else None,
        }
