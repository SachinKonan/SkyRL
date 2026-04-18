"""
Environment for ICLR accept/reject prediction with arXiv retrieval tools.

The agent issues one of:
  - <ssearch>query</ssearch>         — semantic search over the arXiv corpus
  - <asearch>lastname1,lastname2</asearch>  — author search
  - <answer>Accept|Reject</answer>   — final prediction

Retrieval results come back wrapped in <information>...</information>. Reward is
exact match of the answer tag against extras["reward_spec"]["ground_truth"]["target"]
(a list of strings, typically ["Accept"] or ["Reject"]).

Per-episode filter context (exclude_title, upper_bound_datetime) is threaded into
the underlying arxiv_retriever tool group so the model cannot retrieve the paper
it is currently evaluating and cannot see papers from after the submission date.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from omegaconf import DictConfig

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType
from skyrl_gym.envs.search_arxiv.utils import compute_score
from skyrl_gym.tools import SearchArxivToolGroup


@dataclass
class SearchArxivEnvConfig:
    log_requests: bool = False
    search_url: str = "http://127.0.0.1:8000/retrieve"
    topk: int = 5
    timeout: int = 30
    max_turns: int = 3
    search_enabled: bool = True


class SearchArxivEnv(BaseTextEnv):
    def __init__(self, env_config: Union[SearchArxivEnvConfig, DictConfig], extras: Dict[str, Any] = {}):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec is required"
        assert "ground_truth" in extras["reward_spec"], "reward_spec.ground_truth is required"
        self.ground_truth = extras["reward_spec"]["ground_truth"]

        # max_turns: env_config default, overridable per-sample via extras.
        self.max_turns = int(extras.get("max_turns", env_config.max_turns))
        self.search_enabled = bool(extras.get("search_enabled", env_config.search_enabled))

        self.tool_group = SearchArxivToolGroup(
            search_url=env_config.search_url,
            topk=env_config.topk,
            timeout=env_config.timeout,
            log_requests=env_config.log_requests,
        )
        # Per-episode filter context: pass through from the dataset.
        self.tool_group.set_task_context(
            upper_bound_datetime=extras.get("upper_bound_datetime"),
            exclude_title=extras.get("exclude_title"),
        )
        self.init_tool_groups([self.tool_group])

        self.chat_history: ConversationType = []

    _SEM_RE = re.compile(r"<ssearch>(.*?)</ssearch>", re.DOTALL)
    _AUTH_RE = re.compile(r"<asearch>(.*?)</asearch>", re.DOTALL)

    def _parse_action(self, action: str):
        """Return (tool_name, arg) or (None, None)."""
        sem = self._SEM_RE.search(action)
        if sem:
            return "semantic_search", sem.group(1).strip()
        auth = self._AUTH_RE.search(action)
        if auth:
            return "author_search", auth.group(1).strip()
        return None, None

    def _get_reward(self, done: bool) -> float:
        if not done:
            return 0.0
        chat_str = "".join(item["content"] for item in self.chat_history)
        return compute_score(chat_str, self.ground_truth)

    def _is_done(self, action: str) -> bool:
        if self.turns >= self.max_turns:
            return True
        return "<answer>" in action and "</answer>" in action

    def _execute_tool(self, tool_group_name: str, tool_name: str, tool_input: Any) -> str:
        out = super()._execute_tool(tool_group_name, tool_name, tool_input)
        return "\n<information>" + out + "</information>\n"

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        self.chat_history.append({"role": "assistant", "content": action})

        done = self._is_done(action)
        reward = self._get_reward(done)
        if done:
            return BaseTextEnvStepOutput(observations=[], reward=reward, done=True, metadata={})

        tool_name, arg = self._parse_action(action)
        new_obs: Optional[Dict[str, str]] = None
        info: Dict[str, Any] = {"tool_group": "SearchArxivToolGroup", "tool_name": tool_name, "tool_input": arg}

        if tool_name is None:
            new_obs = {
                "role": "user",
                "content": (
                    "\n<information>No valid tag found. Use <ssearch>query</ssearch>, "
                    "<asearch>lastname1,lastname2</asearch>, or <answer>Accept|Reject</answer>.</information>\n"
                ),
            }
        elif not self.search_enabled:
            new_obs = {
                "role": "user",
                "content": (
                    "\n<information>Search is disabled for this episode. "
                    "Respond with <answer>Accept</answer> or <answer>Reject</answer>.</information>\n"
                ),
            }
        else:
            try:
                observation = self._execute_tool("SearchArxivToolGroup", tool_name, [arg])
                new_obs = {"role": "user", "content": observation}
            except Exception as e:
                new_obs = {"role": "user", "content": f"\n<information>Tool error: {e}</information>\n"}

        if new_obs is not None:
            self.chat_history.append(new_obs)

        return BaseTextEnvStepOutput(
            observations=[new_obs] if new_obs else [],
            reward=reward,
            done=done,
            metadata=info,
        )
