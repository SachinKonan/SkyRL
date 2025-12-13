"""
Gemini embedding utilities for SearchR1 environment.

Uses OpenAI-compatible API with Gemini backend.
"""

import logging
import os
import time
from typing import List, Tuple

from openai import OpenAI

logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 10
RETRY_DELAY = 1.0

# Gemini OpenAI-compatible endpoint
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def get_gemini_embedding_sync(
    texts: List[str],
    api_key: str | None = "AIzaSyCIjfLlDmrPpVUUCSz4BXC3tDfmllWYF9M",
    model: str = "gemini-embedding-001",
    embedding_dim: int = 768,
    max_retries: int = MAX_RETRIES,
    retry_delay: float = RETRY_DELAY,
) -> Tuple[List[List[float]], int]:
    """
    Get embeddings from Gemini API using OpenAI-compatible interface.

    Args:
        texts: List of texts to embed
        api_key: Gemini API key (defaults to GEMINI_API_KEY env var)
        model: Gemini embedding model name
        embedding_dim: Desired embedding dimension (768 for gemini-embedding-001)
        max_retries: Maximum number of retries
        retry_delay: Initial delay between retries

    Returns:
        Tuple of (embeddings, retry_count) where:
            - embeddings: List of embeddings (list of list of floats)
            - retry_count: Number of retries needed (0 = success on first try)

    Raises:
        ValueError: If input validation fails
        Exception: If embedding generation fails after all retries
    """
    # Get API key
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set and no api_key provided")

    # Validate input
    if not texts:
        raise ValueError("No texts provided for embedding generation")

    for i, text in enumerate(texts):
        if not isinstance(text, str):
            raise ValueError(f"Text at index {i} is not a string: {type(text)} = {text}")
        if not text.strip():
            raise ValueError(f"Text at index {i} is empty or whitespace only")

    # Create OpenAI client with Gemini endpoint
    client = OpenAI(
        api_key=api_key,
        base_url=GEMINI_BASE_URL,
    )

    # Retry logic with exponential backoff
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                input=texts,
                model=model,
                dimensions=embedding_dim,
            )

            if not response.data:
                raise ValueError("No embeddings returned from Gemini API")

            if len(response.data) != len(texts):
                raise ValueError(f"Mismatch: expected {len(texts)} embeddings, got {len(response.data)}")

            # Extract embeddings in order
            embeddings: List[List[float]] = [item.embedding for item in response.data]

            return embeddings, attempt  # attempt = retry_count (0 = first try succeeded)

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (1.5**attempt)
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed for embedding ({len(texts)} texts): {repr(e)}. "
                    f"Retrying in {wait_time:.1f}s..."
                )
                time.sleep(wait_time)
            else:
                logger.error(f"All {max_retries} attempts failed for embedding ({len(texts)} texts): {repr(e)}")
                raise

    raise RuntimeError("Unexpected error in retry logic")
