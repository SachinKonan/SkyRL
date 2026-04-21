"""
Utility functions for the SearchArxiv environment.

Answer grammar: ``<answer> \\boxed{Accept} </answer>`` or ``<answer> \\boxed{Reject} </answer>``.
The \\boxed{...} wrapper is preserved from the sibling's review_roles prompt — the
three-reviewer simulation produces cleaner outputs when the final answer is wrapped
in a \\boxed{} marker. We also accept a bare ``\\boxed{...}`` anywhere as a fallback.
"""

import re
from typing import List, Optional


def extract_boxed_answer(solution_str: str) -> Optional[str]:
    """Extract answer from ``\\boxed{X}`` inside ``<answer>...</answer>`` tags.

    Falls back to a bare ``\\boxed{X}`` anywhere in the string if no
    ``<answer>`` wrapper is found. Returns None if neither is present.
    """
    answer_matches = list(re.finditer(r"<answer>(.*?)</answer>", solution_str, re.DOTALL))
    if not answer_matches:
        m = re.search(r"\\boxed\{(.*?)\}", solution_str)
        return m.group(1).strip() if m else None

    # Last <answer>...</answer> wins if the model emitted multiple.
    inner = answer_matches[-1].group(1)
    m = re.search(r"\\boxed\{(.*?)\}", inner)
    if m:
        return m.group(1).strip()
    # Backward-compat: accept `<answer>Accept</answer>` without the \boxed wrapper.
    inner = inner.strip()
    return inner if inner else None


def normalize_answer(answer: Optional[str]) -> str:
    if answer is None:
        return ""
    return answer.lower().strip()


def em_check(prediction: Optional[str], golden_answers) -> int:
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    norm_pred = normalize_answer(prediction)
    for g in golden_answers:
        if normalize_answer(g) == norm_pred:
            return 1
    return 0


# Kept for backward-compat with any external imports.
extract_answer = extract_boxed_answer


def compute_score(solution_str: str, ground_truth: dict, score: float = 1.0, format_score: float = 0.0) -> float:
    """Reward == `score` on exact-match of target, `format_score` if a boxed answer
    is extracted but doesn't match any target, 0.0 if no answer at all."""
    answer = extract_boxed_answer(solution_str)
    if answer is None:
        return 0.0
    target = ground_truth.get("target", [])
    if isinstance(target, str):
        target = [target]
    if em_check(answer, target):
        return score
    return format_score


# --- Per-turn format scoring ----------------------------------------------

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_SSEARCH_RE = re.compile(r"<ssearch>(.*?)</ssearch>", re.DOTALL)
_VALID_ANSWERS = frozenset({"accept", "reject"})


def _turn_is_well_formatted(content: str, is_final: bool) -> bool:
    """One assistant turn passes the structural check when ALL hold:

    Common:
      - Turn contains ``</think>``.
      - The prefix before ``</think>`` (minus any opening ``<think>``) is non-empty
        after stripping whitespace. Empty think blocks are disqualifying.

    Final turn:
      - The post-``</think>`` text contains ``<answer>...</answer>``.
      - Inside the answer, exactly one ``\\boxed{...}`` appears.
      - The contents of that ``\\boxed{...}`` normalize to ``accept`` or ``reject``.

    Intermediate turn (multi-turn search only):
      - The post-``</think>`` text contains ``<ssearch>...</ssearch>`` with a
        non-empty query string inside (semantic_search takes a single str).
    """
    if _THINK_CLOSE not in content:
        return False
    pre, post = content.split(_THINK_CLOSE, 1)
    think_body = pre.split(_THINK_OPEN, 1)[1] if _THINK_OPEN in pre else pre
    if not think_body.strip():
        return False
    if is_final:
        m = _ANSWER_RE.search(post)
        if not m:
            return False
        boxed = _BOXED_RE.findall(m.group(1))
        if len(boxed) != 1:
            return False
        return boxed[0].strip().lower() in _VALID_ANSWERS
    m = _SSEARCH_RE.search(post)
    return bool(m and m.group(1).strip())


def compute_format_score(chat_history: List[dict]) -> float:
    """Fraction of assistant turns that pass the per-turn format check.
    Single-turn runs collapse to {0.0, 1.0}; multi-turn yields k/N.
    """
    turns = [m["content"] for m in chat_history if m.get("role") == "assistant"]
    if not turns:
        return 0.0
    final_idx = len(turns) - 1
    ok = sum(
        1 for i, content in enumerate(turns) if _turn_is_well_formatted(content, is_final=(i == final_idx))
    )
    return ok / len(turns)
