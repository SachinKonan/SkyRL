"""
SearchR1 Environment with embedded Gemini embeddings and ChromaDB.

This environment performs retrieval-augmented generation using:
- Gemini embeddings (768 dims) via REST API for query encoding
- ChromaDB PersistentClient for vector search over Wikipedia

No separate server needed - everything runs in the driver process.
"""

import logging
import re
import threading
import time
from typing import Any, Dict, List, Optional

import chromadb
from omegaconf import DictConfig

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType
from skyrl_gym.envs.searchr1embeddings.embedding import get_gemini_embedding_sync
from skyrl_gym.envs.searchr1embeddings.utils import compute_score

logger = logging.getLogger(__name__)

# Retry configuration for ChromaDB queries
MAX_RETRIES = 10
INITIAL_RETRY_DELAY = 1.0


class SearchR1EmbeddingsEnv(BaseTextEnv):
    """
    Environment for Search-R1 tasks with embedded Gemini embeddings and ChromaDB.

    Uses class-level shared clients for efficiency:
    - ChromaDB PersistentClient (embedded, no HTTP server)
    - Gemini REST API for embeddings (uses GEMINI_API_KEY env var)
    - threading.Semaphore(128) for concurrency control

    Expected extras format:
        {
            "reward_model": {
                "ground_truth": {"target": ["answer1", "answer2"]}
            },
            "max_turns": 3  # optional, default=2
        }

    Config parameters:
        chroma_path: Path to ChromaDB persistent storage
        collection_name: ChromaDB collection name (default: "wiki_embeddings")
        topk: Number of documents to retrieve (default: 3)
    """

    # Class-level shared clients (initialized once, shared across all instances)
    _chroma_client: Optional[chromadb.ClientAPI] = None
    _chroma_collection: Optional[Any] = None
    _semaphore: threading.Semaphore = threading.Semaphore(128)
    _init_lock: threading.Lock = threading.Lock()
    _initialized: bool = False

    # Store config for later use
    _topk: int = 3

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__()

        # Support both reward_model and reward_spec for backwards compatibility
        if "reward_model" in extras:
            assert "ground_truth" in extras["reward_model"], "ground_truth is required in reward_model field"
            self.ground_truth = extras["reward_model"]["ground_truth"]
        elif "reward_spec" in extras:
            assert "ground_truth" in extras["reward_spec"], "ground_truth is required in reward_spec field"
            self.ground_truth = extras["reward_spec"]["ground_truth"]
        else:
            raise AssertionError("reward_model or reward_spec field is required")

        self.max_turns = extras.get("max_turns", 2)

        # Store instance config
        self.topk = getattr(env_config, "topk", 3)
        self.log_requests = getattr(env_config, "log_requests", True)

        # Initialize shared clients (thread-safe, once)
        self._ensure_clients_initialized(env_config)

        # Chat history
        self.chat_history: ConversationType = []

    @classmethod
    def _ensure_clients_initialized(cls, env_config: DictConfig) -> None:
        """Initialize shared ChromaDB client (thread-safe, once)."""
        with cls._init_lock:
            if cls._initialized:
                return

            # Get config parameters
            chroma_path = getattr(env_config, "chroma_path", None)
            if chroma_path is None:
                raise ValueError("env_config.chroma_path is required")

            collection_name = getattr(env_config, "collection_name", "wiki_embeddings")
            cls._topk = getattr(env_config, "topk", 3)

            logger.info(f"Initializing ChromaDB from {chroma_path}")
            cls._chroma_client = chromadb.PersistentClient(path=chroma_path)

            # List available collections for debugging
            collections = cls._chroma_client.list_collections()
            logger.info(f"Available collections: {[c.name for c in collections]}")

            cls._chroma_collection = cls._chroma_client.get_collection(collection_name)
            logger.info(f"Loaded collection '{collection_name}'")

            cls._initialized = True

    def _search(self, query: str) -> tuple[str, int]:
        """
        Perform search using embedded ChromaDB + Gemini REST API.

        Args:
            query: Search query string

        Returns:
            Tuple of (formatted search results, gemini_retry_count)
        """
        if not query or not query.strip():
            return "Error: Empty query provided", 0

        with self._semaphore:
            try:
                if self.log_requests:
                    logger.info(f"Processing search query: {query[:100]}...")

                # Get embedding from Gemini REST API
                embeddings, gemini_retry_count = get_gemini_embedding_sync([query.strip()])

                # Query ChromaDB with retry logic
                results = self._query_chroma_with_retry(embeddings, n_results=self.topk)

                # Format results
                formatted = self._format_results(results, query)

                if self.log_requests:
                    logger.info(f"Returning {len(results.get('documents', [[]])[0])} results (retries={gemini_retry_count})")

                return formatted, gemini_retry_count

            except Exception as e:
                logger.error(f"Search error: {e}")
                return f"Error during search: {str(e)}", 0

    def _query_chroma_with_retry(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 3,
        max_retries: int = MAX_RETRIES,
        initial_retry_delay: float = INITIAL_RETRY_DELAY,
    ) -> Dict[str, Any]:
        """Query ChromaDB with retry logic."""
        for attempt in range(max_retries):
            try:
                results = self._chroma_collection.query(
                    query_embeddings=query_embeddings,
                    n_results=n_results,
                )
                return results
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = initial_retry_delay * (1.5**attempt)
                    logger.warning(
                        f"ChromaDB query attempt {attempt + 1}/{max_retries} failed: {e}. "
                        f"Retrying in {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)
                else:
                    raise

        raise RuntimeError("All ChromaDB query attempts failed")

    def _format_results(self, results: Dict[str, Any], query: str) -> str:
        """Format ChromaDB results as a readable string."""
        if results.get("documents") is None or len(results["documents"]) == 0:
            return "No results found."

        # results["documents"] is a list of lists (one per query)
        # We only have one query, so take the first list
        documents = results["documents"][0] if results["documents"] else []

        if not documents:
            return "No results found."

        # Format as numbered documents
        formatted_parts = []
        for i, doc in enumerate(documents, 1):
            formatted_parts.append(f"Document {i}:\n{doc}")

        return "\n\n".join(formatted_parts)

    def _parse_action(self, action: str) -> Optional[str]:
        """Parse search query from action string."""
        if "<search>" in action and "</search>" in action:
            match = re.search(r"<search>(.*?)</search>", action, re.DOTALL)
            if match:
                return match.group(1).strip()
        return None

    def _get_reward(self, action: str, done: bool) -> float:
        if done:
            # Concat all chat history into a single string and compute reward
            chat_history_str = "".join([item["content"] for item in self.chat_history])
            return compute_score(chat_history_str, self.ground_truth)
        else:
            # No reward for intermediate steps
            return 0

    def _is_done(self, action: str) -> bool:
        if self.turns >= self.max_turns:
            return True
        return "<answer>" in action and "</answer>" in action

    def _validate_action(self, action: str):
        stop_tags = ["</search>", "</answer>"]
        for tag in stop_tags:
            if tag in action:
                assert action.split(tag, 1)[1] == "", (
                    f"{tag} detected in the response but it is not the last string generated. "
                    f"Use {stop_tags} as stop strings in the configuration."
                )

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        self._validate_action(action)
        self.chat_history.append({"role": "assistant", "content": action})

        error = None
        done = self._is_done(action)
        reward = self._get_reward(action, done)
        gemini_retry_count = 0

        if done:
            return BaseTextEnvStepOutput(observations=[], reward=reward, done=done, metadata={})

        # Parse and execute search
        query = self._parse_action(action)
        observation = None

        if query:
            try:
                search_result, gemini_retry_count = self._search(query)
                observation = "\n<information>" + search_result + "</information>\n"
            except Exception as e:
                error = str(e)
                observation = None
        else:
            # No valid search tag found
            observation = "\n<information>No search query found. Use <search>query</search> format.</information>\n"

        # Wrap the observation properly as a message
        if observation:
            new_obs = {"role": "user", "content": observation}
        elif error:
            new_obs = {"role": "user", "content": f"\n<information>Error: {error}</information>\n"}
        else:
            new_obs = None

        info = {
            "tool_name": "search",
            "tool_input": query,
            "gemini_retry_count": gemini_retry_count,
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
