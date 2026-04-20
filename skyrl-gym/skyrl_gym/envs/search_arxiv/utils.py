"""
Utility functions for the SearchArxiv environment.

Answer grammar: ``<answer> \\boxed{Accept} </answer>`` or ``<answer> \\boxed{Reject} </answer>``.
The \\boxed{...} wrapper is preserved from the sibling's review_roles prompt — the
three-reviewer simulation produces cleaner outputs when the final answer is wrapped
in a \\boxed{} marker. We also accept a bare ``\\boxed{...}`` anywhere as a fallback.
"""

import re
from typing import Optional


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
