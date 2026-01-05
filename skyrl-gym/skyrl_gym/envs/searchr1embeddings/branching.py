"""
Gemini-based branching point selection for progressive rollouts.

Uses Gemini 2.5 Flash to intelligently select branch points based on
trajectory analysis rather than random selection.
"""

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

# Configuration defaults
MAX_RETRIES = 5
RETRY_DELAY = 1.0
DEFAULT_TIMEOUT = 30.0
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_MODEL = "gemini-2.5-flash"


@dataclass
class GeminiBranchingResult:
    """Result from Gemini branching point selection."""

    suggested_substring: Optional[str]
    raw_response: str
    success: bool
    fallback_reason: Optional[str]  # None if success, else reason for fallback
    retry_count: int


BRANCHING_PROMPT_TEMPLATE = """You are analyzing a completed trajectory from a reasoning/search task. Your goal is to identify the best point to create an alternative branch.

## Trajectory Information
- Final Reward: {reward}
- Total Turns: {num_turns}

## Full Trajectory Text
{full_trajectory_text}

## Task
Identify a specific point in the ASSISTANT's responses where the reasoning could have gone differently. Return an exact substring (5-50 characters) that appears in the assistant's text, and the branch should occur immediately AFTER this substring.

Requirements:
1. The substring MUST appear exactly in the assistant messages (not user/observation text)
2. Choose a decision point, pivot, or where reasoning direction was set
3. For failed trajectories (reward=0), identify where the mistake likely started
4. For successful trajectories (reward>0), identify alternative exploration points
5. Return ONLY the substring, nothing else. No quotes, no explanation.

Substring to branch after:"""


class GeminiBranchingClient:
    """Client for Gemini-based branching point selection.

    Uses Gemini 2.5 Flash to analyze trajectories and suggest where to branch.
    Thread-safe with semaphore protection for concurrent requests.
    """

    _semaphore: threading.Semaphore = threading.Semaphore(128)
    _client: Optional[OpenAI] = None
    _init_lock: threading.Lock = threading.Lock()

    def __init__(
        self,
        api_key: str = 'AIzaSyCIjfLlDmrPpVUUCSz4BXC3tDfmllWYF9M',
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = MAX_RETRIES,
    ):
        """Initialize the Gemini branching client.

        Args:
            api_key: Gemini API key. If None, uses GEMINI_API_KEY env var.
            model: Gemini model to use.
            timeout: Request timeout in seconds.
            max_retries: Maximum retry attempts.
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable not set and no api_key provided"
            )

        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

        # Initialize client (thread-safe)
        with self._init_lock:
            if GeminiBranchingClient._client is None:
                GeminiBranchingClient._client = OpenAI(
                    api_key=self.api_key,
                    base_url=GEMINI_BASE_URL,
                    timeout=timeout,
                )

    def select_branch_point(
        self,
        full_trajectory_text: str,
        reward: float,
        num_turns: int,
    ) -> GeminiBranchingResult:
        """Ask Gemini to suggest a branching point substring.

        Args:
            full_trajectory_text: The decoded full trajectory text.
            reward: Final reward of the trajectory.
            num_turns: Number of turns in the trajectory.

        Returns:
            GeminiBranchingResult with suggested substring or fallback info.
        """
        # Build prompt (limit context to avoid token limits)
        prompt = BRANCHING_PROMPT_TEMPLATE.format(
            reward=reward,
            num_turns=num_turns,
            full_trajectory_text=full_trajectory_text[:12000],
        )

        with self._semaphore:
            for attempt in range(self.max_retries):
                try:
                    start_time = time.time()
                    response = self._client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=100,
                        temperature=0.3,
                    )
                    elapsed_ms = (time.time() - start_time) * 1000

                    raw_response = response.choices[0].message.content
                    if raw_response is None:
                        raw_response = ""
                    raw_response = raw_response.strip()

                    # Validate response
                    if not raw_response:
                        return GeminiBranchingResult(
                            suggested_substring=None,
                            raw_response="",
                            success=False,
                            fallback_reason="empty_response",
                            retry_count=attempt,
                        )

                    # Clean up the response (remove quotes if present)
                    substring = raw_response.strip("\"'`")

                    # Validate length
                    if len(substring) < 3:
                        return GeminiBranchingResult(
                            suggested_substring=None,
                            raw_response=raw_response,
                            success=False,
                            fallback_reason="invalid_length_too_short",
                            retry_count=attempt,
                        )

                    if len(substring) > 200:
                        return GeminiBranchingResult(
                            suggested_substring=None,
                            raw_response=raw_response,
                            success=False,
                            fallback_reason="invalid_length_too_long",
                            retry_count=attempt,
                        )

                    logger.info(
                        f"Gemini branching took {elapsed_ms:.1f}ms, "
                        f"suggested: '{substring[:50]}{'...' if len(substring) > 50 else ''}'"
                    )

                    return GeminiBranchingResult(
                        suggested_substring=substring,
                        raw_response=raw_response,
                        success=True,
                        fallback_reason=None,
                        retry_count=attempt,
                    )

                except Exception as e:
                    if attempt < self.max_retries - 1:
                        wait_time = RETRY_DELAY * (1.5**attempt)
                        logger.warning(
                            f"Gemini branching attempt {attempt + 1}/{self.max_retries} "
                            f"failed: {repr(e)}. Retrying in {wait_time:.1f}s..."
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(
                            f"All {self.max_retries} Gemini branching attempts failed: {repr(e)}"
                        )
                        return GeminiBranchingResult(
                            suggested_substring=None,
                            raw_response="",
                            success=False,
                            fallback_reason=f"api_error:{type(e).__name__}",
                            retry_count=attempt,
                        )

        # Should not reach here, but just in case
        return GeminiBranchingResult(
            suggested_substring=None,
            raw_response="",
            success=False,
            fallback_reason="unexpected_error",
            retry_count=self.max_retries,
        )


def get_gemini_branch_point_sync(
    full_trajectory_text: str,
    reward: float,
    num_turns: int,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    timeout: float = DEFAULT_TIMEOUT,
    max_retries: int = MAX_RETRIES,
) -> GeminiBranchingResult:
    """Convenience function for getting Gemini branch point suggestion.

    Creates a client per call. For better performance with multiple calls,
    create a GeminiBranchingClient instance and reuse it.

    Args:
        full_trajectory_text: The decoded full trajectory text.
        reward: Final reward of the trajectory.
        num_turns: Number of turns in the trajectory.
        api_key: Gemini API key. If None, uses GEMINI_API_KEY env var.
        model: Gemini model to use.
        timeout: Request timeout in seconds.
        max_retries: Maximum retry attempts.

    Returns:
        GeminiBranchingResult with suggested substring or fallback info.
    """
    client = GeminiBranchingClient(
        api_key=api_key,
        model=model,
        timeout=timeout,
        max_retries=max_retries,
    )
    return client.select_branch_point(full_trajectory_text, reward, num_turns)
