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
Utility functions for the SearchArxiv environment.

Scoring is based on extracting the answer from \\boxed{Accept} or \\boxed{Reject}
format within <answer>...</answer> tags.
"""

import re
from typing import Optional, List, Union


def extract_boxed_answer(solution_str: str) -> Optional[str]:
    """
    Extract answer from \\boxed{Accept} or \\boxed{Reject} within <answer> tags.

    Args:
        solution_str: The full solution string containing the answer.

    Returns:
        The extracted answer (e.g., "Accept" or "Reject"), or None if not found.
    """
    # Try to find <answer>...</answer> tags first
    answer_pattern = r"<answer>(.*?)</answer>"
    answer_matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))

    if not answer_matches:
        # No answer tags found, try to find boxed directly
        boxed_match = re.search(r"\\boxed\{(.*?)\}", solution_str)
        if boxed_match:
            return boxed_match.group(1).strip()
        return None

    # Get the last answer match (in case there are multiple)
    answer_content = answer_matches[-1].group(1)

    # Extract from \boxed{} within the answer
    boxed_match = re.search(r"\\boxed\{(.*?)\}", answer_content)
    if boxed_match:
        return boxed_match.group(1).strip()

    return None


def normalize_answer(answer: str) -> str:
    """
    Normalize the answer for comparison.

    Args:
        answer: The answer string to normalize.

    Returns:
        Normalized answer string.
    """
    if answer is None:
        return ""
    return answer.lower().strip()


def compute_score(
    solution_str: str,
    ground_truth: dict,
    score: float = 1.0,
    format_score: float = 0.0
) -> float:
    """
    Compute the score based on exact match of boxed answer.

    Args:
        solution_str: The solution text containing the answer.
        ground_truth: Dictionary with "target" key containing valid answers.
        score: Score to return for correct answer (default 1.0).
        format_score: Score to return for wrong answer but correct format (default 0.0).

    Returns:
        The computed score (0.0, format_score, or score).
    """
    answer = extract_boxed_answer(solution_str)

    if answer is None:
        return 0.0

    target = ground_truth.get("target", [])
    if isinstance(target, str):
        target = [target]

    # Normalize and compare
    answer_normalized = normalize_answer(answer)
    for t in target:
        if normalize_answer(t) == answer_normalized:
            return score

    # Answer was extracted but didn't match - could give format_score
    return format_score
