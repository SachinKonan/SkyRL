"""
Tool group wrapping the arxiv_retriever FastAPI server.

The arxiv_retriever `/retrieve` endpoint accepts a unified request body that covers
both semantic search (by `query`) and author search (by `last_name_author_search`),
plus per-request `upper_bound_datetime` / `exclude_title` filters. We expose two
separate tools (`semantic_search`, `author_search`) so the env can map
`<ssearch>` / `<asearch>` tags cleanly; the filter context comes from the env at
init time (one paper-under-review per episode).
"""

import json
import logging
import threading
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import requests

from skyrl_gym.tools.core import ToolGroup, tool

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 1


def _post_retrieve(
    url: str,
    payload: Dict[str, Any],
    timeout: int,
    session: requests.Session,
    log_requests: bool,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    request_id = str(uuid.uuid4())[:8]
    last_error: Optional[str] = None
    for attempt in range(MAX_RETRIES):
        try:
            if log_requests:
                logger.info(f"[arxiv-retrieve {request_id}] attempt {attempt + 1}: {url} payload={payload}")
            resp = session.post(url, json=payload, timeout=timeout, headers={"Content-Type": "application/json"})
            if resp.status_code in (500, 502, 503, 504):
                last_error = f"server {resp.status_code}"
                import time as _time

                _time.sleep(INITIAL_RETRY_DELAY * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json(), None
        except requests.exceptions.RequestException as e:
            last_error = f"request error: {e}"
            import time as _time

            _time.sleep(INITIAL_RETRY_DELAY * (attempt + 1))
    return None, last_error


def _format_results(raw: Any) -> str:
    """
    arxiv_retriever returns {"result": [[{"document": {...}, "score": ...}, ...]]}
    for semantic-with-scores, or {"result": [[{"contents": "...", "title": "..."}]]}
    for author search / no-scores semantic. Normalize to a plain text block.
    """
    if raw is None:
        return ""
    batches = raw.get("result") or []
    if not batches:
        return "No results."
    lines = []
    for docs in batches:
        for idx, item in enumerate(docs):
            if "document" in item:
                doc = item["document"]
            else:
                doc = item
            title = doc.get("title") or ""
            contents = doc.get("contents") or doc.get("content") or ""
            header = f"Doc {idx + 1}"
            if title:
                header += f" — {title}"
            lines.append(f"{header}\n{contents.strip()}")
    return "\n---\n".join(lines) if lines else "No results."


class SearchArxivToolGroup(ToolGroup):
    _session_pool: Dict[str, requests.Session] = {}
    _session_lock = threading.Lock()

    @classmethod
    def _get_session(cls, base_url: str) -> requests.Session:
        with cls._session_lock:
            if base_url not in cls._session_pool:
                s = requests.Session()
                adapter = requests.adapters.HTTPAdapter(pool_connections=64, pool_maxsize=64, max_retries=0)
                s.mount("http://", adapter)
                s.mount("https://", adapter)
                cls._session_pool[base_url] = s
            return cls._session_pool[base_url]

    def __init__(
        self,
        search_url: str = "http://127.0.0.1:8000/retrieve",
        topk: int = 5,
        timeout: int = DEFAULT_TIMEOUT,
        log_requests: bool = False,
    ):
        self.search_url = search_url
        self.topk = topk
        self.timeout = timeout
        self.log_requests = log_requests
        parsed = urlparse(self.search_url)
        self.session = self._get_session(f"{parsed.scheme}://{parsed.netloc}")

        # Per-episode filter context. The env sets these after reset() based on the
        # paper being evaluated (exclude_title) and its submission date cap.
        self.upper_bound_datetime: Optional[str] = None
        self.exclude_title: Optional[str] = None

        super().__init__(name="SearchArxivToolGroup")

    def set_task_context(self, upper_bound_datetime: Optional[str], exclude_title: Optional[str]):
        self.upper_bound_datetime = upper_bound_datetime
        self.exclude_title = exclude_title

    def _base_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"topk": self.topk, "return_scores": False}
        if self.upper_bound_datetime:
            payload["upper_bound_datetime"] = self.upper_bound_datetime
        if self.exclude_title:
            payload["exclude_title"] = self.exclude_title
        return payload

    @tool
    def semantic_search(self, query: Union[str, List[str]]) -> str:
        """Run one or more semantic-search queries.

        `query` may be a single string or a list of strings. When a list, each
        query is posted to the retrieval server separately and results are
        concatenated with `## Query k: "<q>"` headers so the caller can see
        which results came from which query.
        """
        if isinstance(query, str):
            queries = [query.strip()] if query and query.strip() else []
        else:
            queries = [q.strip() for q in (query or []) if isinstance(q, str) and q.strip()]
        if not queries:
            return json.dumps({"result": "Empty query."})

        sections: List[str] = []
        for idx, q in enumerate(queries, start=1):
            payload = self._base_payload()
            payload["query"] = q
            raw, err = _post_retrieve(self.search_url, payload, self.timeout, self.session, self.log_requests)
            header = f'## Query {idx}: "{q}"'
            if err:
                sections.append(f"{header}\nRetrieval error: {err}")
            else:
                sections.append(f"{header}\n{_format_results(raw)}")
        return json.dumps({"result": "\n\n".join(sections)})

    @tool
    def author_search(self, authors: str) -> str:
        if not authors:
            return json.dumps({"result": "Empty author list."})
        payload = self._base_payload()
        payload["last_name_author_search"] = authors.strip()
        raw, err = _post_retrieve(self.search_url, payload, self.timeout, self.session, self.log_requests)
        if err:
            return json.dumps({"result": f"Retrieval error: {err}"})
        return json.dumps({"result": _format_results(raw)})
