"""
Environment for ICLR accept/reject prediction with arXiv retrieval + python tools.

The agent issues one of:
  - <ssearch>q1\\nq2\\n...</ssearch>  — semantic search; body is newline-separated
                                        queries (a single-line body is one query)
  - <python>code</python>             — execute Python; stdout/stderr comes back
  - <answer>...\\boxed{Accept|Reject}</answer>   — final prediction

Tool results come back wrapped in <information>...</information>. Reward is exact
match of the boxed answer against extras["reward_spec"]["ground_truth"]["target"]
(a list of strings, typically ["Accept"] or ["Reject"]).

Per-episode filter context (exclude_title, upper_bound_datetime) is threaded into
the underlying arxiv_retriever tool group so the model cannot retrieve the paper
it is currently evaluating and cannot see papers from after the submission date.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from omegaconf import DictConfig

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType
from skyrl_gym.envs.search_arxiv.utils import compute_format_score, compute_score
from skyrl_gym.tools import SearchArxivToolGroup
from skyrl_gym.tools.python import PythonCodeExecutorToolGroup


@dataclass
class SearchArxivEnvConfig:
    log_requests: bool = False
    search_url: str = "http://127.0.0.1:8000/retrieve"
    topk: int = 5
    timeout: int = 30
    max_turns: int = 3
    search_enabled: bool = True
    format_weight: float = 0.2
    """Weight (alpha) on the per-turn format reward. Final reward is
    (1 - alpha) * R_acc + alpha * R_fmt."""
    python_enabled: bool = True
    python_timeout: float = 10.0


class SearchArxivEnv(BaseTextEnv):
    def __init__(self, env_config: Union[SearchArxivEnvConfig, DictConfig], extras: Dict[str, Any] = {}):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec is required"
        assert "ground_truth" in extras["reward_spec"], "reward_spec.ground_truth is required"
        self.ground_truth = extras["reward_spec"]["ground_truth"]

        # max_turns: env_config default, overridable per-sample via extras.
        self.max_turns = int(extras.get("max_turns", env_config.max_turns))
        self.search_enabled = bool(extras.get("search_enabled", env_config.search_enabled))
        self.python_enabled = bool(extras.get("python_enabled", getattr(env_config, "python_enabled", True)))
        self.format_weight = float(extras.get("format_weight", env_config.format_weight))

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
        self.python_group = PythonCodeExecutorToolGroup(
            timeout=float(getattr(env_config, "python_timeout", 10.0)),
        )
        self.init_tool_groups([self.tool_group, self.python_group])

        self.chat_history: ConversationType = []

    _SEM_RE = re.compile(r"<ssearch>(.*?)</ssearch>", re.DOTALL)
    _PY_RE = re.compile(r"<python>(.*?)</python>", re.DOTALL)

    def _parse_action(self, action: str) -> Tuple[Optional[str], Any]:
        """Return (tool_name, arg) or (None, None).

        - <ssearch>body</ssearch> → ("semantic_search", List[str]); body is split
          on newlines and non-empty lines become queries.
        - <python>code</python>   → ("python", code_str).
        Author-search (<asearch>) has been retired.
        """
        sem = self._SEM_RE.search(action)
        if sem:
            body = sem.group(1)
            queries = [q.strip() for q in body.split("\n") if q.strip()]
            return "semantic_search", queries
        py = self._PY_RE.search(action)
        if py:
            return "python", py.group(1)
        return None, None

    def _get_reward(self, done: bool) -> float:
        """R = (1 - alpha) * R_acc + alpha * R_fmt on done, else 0.

        R_acc: EM over \\boxed{Accept|Reject} inside the final <answer> block.
        R_fmt: fraction of assistant turns that pass the structure check
        (</think> + <answer>...\\boxed{X}...</answer> on the final turn,
        </think> + <ssearch>...</ssearch> on intermediate turns).
        """
        if not done:
            return 0.0
        chat_str = "".join(item["content"] for item in self.chat_history)
        acc = compute_score(chat_str, self.ground_truth)
        fmt = compute_format_score(self.chat_history)
        alpha = self.format_weight
        return (1.0 - alpha) * acc + alpha * fmt

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
        tool_group_name = (
            "PythonCodeExecutorToolGroup" if tool_name == "python" else "SearchArxivToolGroup"
        )
        new_obs: Optional[Dict[str, str]] = None
        info: Dict[str, Any] = {"tool_group": tool_group_name, "tool_name": tool_name, "tool_input": arg}

        if tool_name is None:
            new_obs = {
                "role": "user",
                "content": (
                    "\n<information>No valid tag found. Use <ssearch>query</ssearch>, "
                    "<python>code</python>, or <answer>\\boxed{Accept|Reject}</answer>.</information>\n"
                ),
            }
        elif tool_name == "semantic_search" and not self.search_enabled:
            new_obs = {
                "role": "user",
                "content": (
                    "\n<information>Search is disabled for this episode. "
                    "Respond with <answer>\\boxed{Accept}</answer> or <answer>\\boxed{Reject}</answer>.</information>\n"
                ),
            }
        elif tool_name == "python" and not self.python_enabled:
            new_obs = {
                "role": "user",
                "content": (
                    "\n<information>Python execution is disabled for this episode.</information>\n"
                ),
            }
        else:
            try:
                observation = self._execute_tool(tool_group_name, tool_name, [arg])
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
