import re
import string


def _normalize(s: str) -> str:
    s = re.sub(r"\b(a|an|the)\b", " ", s.lower())
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    return " ".join(s.split())


def em_check(prediction: str, golden_answers) -> int:
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    norm_pred = _normalize(prediction)
    for gold in golden_answers:
        if _normalize(gold) == norm_pred:
            return 1
    return 0


def extract_answer(solution_str: str):
    matches = list(re.finditer(r"<answer>(.*?)</answer>", solution_str, re.DOTALL))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def compute_score(solution_str: str, ground_truth, score: float = 1.0, format_score: float = 0.0) -> float:
    answer = extract_answer(solution_str)
    if answer is None:
        return 0.0
    return score if em_check(answer, ground_truth["target"]) else format_score
