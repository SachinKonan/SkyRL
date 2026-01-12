"""
ArXiv Search Tools for paper acceptance prediction.

Provides two search modes:
1. Semantic search: <ssearch> query </ssearch>
2. Author search: <asearch> author1,author2 </asearch>
"""

import json
import logging
import requests
import uuid
import time
import threading
from typing import Tuple, Optional, Any, Dict
from urllib.parse import urlparse

from skyrl_gym.tools.core import tool, ToolGroup

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_TIMEOUT = 30
DEFAULT_TOPK = 5
MAX_RETRIES = 10
INITIAL_RETRY_DELAY = 1


def call_semantic_search_api(
    retrieval_service_url: str,
    query: str,
    topk: int = DEFAULT_TOPK,
    timeout: int = DEFAULT_TIMEOUT,
    log_requests: bool = True,
    session: Optional[requests.Session] = None,
    upper_bound_datetime: Optional[str] = None,
    exclude_title: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Calls the search API with a semantic query.

    Args:
        retrieval_service_url: The URL of the search API endpoint.
        query: The semantic search query.
        topk: The number of results to return.
        timeout: The timeout for the request.
        log_requests: Whether to log requests.
        session: Optional requests.Session for connection reuse.
        upper_bound_datetime: Optional date filter in ISO format (YYYY-MM-DD).
        exclude_title: Optional title to exclude from results (paper being evaluated).

    Returns:
        response: The response from the search API (json if successful, None otherwise)
        error_msg: The error message if the request failed.
    """
    request_id = str(uuid.uuid4())
    log_prefix = f"[Semantic Search ID: {request_id}] "

    payload = {"query": query, "topk": topk, "return_scores": False}
    if upper_bound_datetime:
        payload["upper_bound_datetime"] = upper_bound_datetime
    if exclude_title:
        payload["exclude_title"] = exclude_title
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    return _make_api_request(
        retrieval_service_url, payload, headers, timeout,
        log_requests, log_prefix, session
    )


def call_author_search_api(
    retrieval_service_url: str,
    authors: str,
    topk: int = DEFAULT_TOPK,
    timeout: int = DEFAULT_TIMEOUT,
    log_requests: bool = True,
    session: Optional[requests.Session] = None,
    upper_bound_datetime: Optional[str] = None,
    exclude_title: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Calls the search API with an author search query.

    Args:
        retrieval_service_url: The URL of the search API endpoint.
        authors: Comma-separated list of author last names.
        topk: The number of results to return.
        timeout: The timeout for the request.
        log_requests: Whether to log requests.
        session: Optional requests.Session for connection reuse.
        upper_bound_datetime: Optional date filter in ISO format (YYYY-MM-DD).
        exclude_title: Optional title to exclude from results (paper being evaluated).

    Returns:
        response: The response from the search API (json if successful, None otherwise)
        error_msg: The error message if the request failed.
    """
    request_id = str(uuid.uuid4())
    log_prefix = f"[Author Search ID: {request_id}] "

    payload = {"last_name_author_search": authors, "topk": topk, "return_scores": False}
    if upper_bound_datetime:
        payload["upper_bound_datetime"] = upper_bound_datetime
    if exclude_title:
        payload["exclude_title"] = exclude_title
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    return _make_api_request(
        retrieval_service_url, payload, headers, timeout,
        log_requests, log_prefix, session
    )


def _make_api_request(
    url: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    timeout: int,
    log_requests: bool,
    log_prefix: str,
    session: Optional[requests.Session] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Common API request logic with retry handling.
    """
    # Use provided session or create a new one for this request
    if session is None:
        session = requests.Session()
        should_close_session = True
    else:
        should_close_session = False

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            if log_requests:
                logger.info(
                    f"{log_prefix}Attempt {attempt + 1}/{MAX_RETRIES}: Calling API at {url}"
                )
            start_time = time.time()
            response = session.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            end_time = time.time()

            # Check for Gateway Timeout (504) and other server errors for retrying
            if response.status_code in [500, 502, 503, 504]:
                last_error = f"{log_prefix}API Request Error: Server Error ({response.status_code}) on attempt {attempt + 1}/{MAX_RETRIES}"
                logger.warning(last_error)
                if attempt < MAX_RETRIES - 1:
                    delay = INITIAL_RETRY_DELAY * (attempt + 1)
                    logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                    time.sleep(delay)
                continue

            # Check for other HTTP errors (e.g., 4xx)
            response.raise_for_status()

            # If successful (status code 2xx)
            if log_requests:
                logger.info(f"{log_prefix}API call successful on attempt {attempt + 1} in {(end_time - start_time):.1f}")

            # Close session if we created it
            if should_close_session:
                session.close()

            return response.json(), None

        except requests.exceptions.ConnectionError as e:
            last_error = f"{log_prefix}Connection Error: {e}"
            logger.warning(last_error)
            if attempt < MAX_RETRIES - 1:
                delay = INITIAL_RETRY_DELAY * (attempt + 1)
                logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                time.sleep(delay)
            continue
        except requests.exceptions.Timeout as e:
            last_error = f"{log_prefix}Timeout Error: {e}"
            logger.warning(last_error)
            if attempt < MAX_RETRIES - 1:
                delay = INITIAL_RETRY_DELAY * (attempt + 1)
                logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                time.sleep(delay)
            continue
        except requests.exceptions.RequestException as e:
            last_error = f"{log_prefix}API Request Error: {e}"
            break  # Exit retry loop on other request errors
        except json.JSONDecodeError as e:
            raw_response_text = response.text if "response" in locals() else "N/A"
            last_error = f"{log_prefix}API Response JSON Decode Error: {e}, Response: {raw_response_text[:200]}"
            break  # Exit retry loop on JSON decode errors
        except Exception as e:
            last_error = f"{log_prefix}Unexpected Error: {e}"
            break  # Exit retry loop on other unexpected errors

    # If we reach here, all attempts failed
    logger.error(f"{log_prefix}API Request Failed after {MAX_RETRIES} attempts: {last_error}")

    # Close session if we created it
    if should_close_session:
        session.close()

    return None, last_error


def _passages2string(retrieval_result):
    """Format retrieval results into a readable string."""
    format_reference = ""
    for idx, doc_item in enumerate(retrieval_result):
        # Handle both formats: {"document": {"contents": ...}} and {"contents": ...}
        if "document" in doc_item:
            content = doc_item["document"]["contents"].strip()
        else:
            content = doc_item.get("contents", "").strip()
        format_reference += f"Doc {idx+1}: {content}\n"
    return format_reference


class SearchArxivToolGroup(ToolGroup):
    """
    Tool group for arXiv paper search with semantic and author search capabilities.
    """
    # Class-level session pool shared across all instances
    _session_pool = {}
    _session_lock = threading.Lock()

    @classmethod
    def _get_shared_session(cls, base_url: str) -> requests.Session:
        """Get or create a shared session for the given base URL"""
        with cls._session_lock:
            if base_url not in cls._session_pool:
                session = requests.Session()
                # Configure connection pooling
                adapter = requests.adapters.HTTPAdapter(
                    pool_connections=20,  # Number of connection pools
                    pool_maxsize=20,  # Max connections per pool
                    max_retries=0,  # We handle retries ourselves
                    pool_block=False,  # Don't block if pool is full
                )
                session.mount("http://", adapter)
                session.mount("https://", adapter)
                cls._session_pool[base_url] = session
                logger.info(f"Created shared session pool for {base_url}")
            return cls._session_pool[base_url]

    def __init__(
        self,
        search_url: str = "http://127.0.0.1:8000/retrieve",
        topk: int = DEFAULT_TOPK,
        timeout: int = DEFAULT_TIMEOUT,
        log_requests: bool = True,
        upper_bound_datetime: Optional[str] = None,
        paper_title: Optional[str] = None,
    ):
        self.search_url = search_url
        self.topk = topk
        self.timeout = timeout
        self.log_requests = log_requests
        self.upper_bound_datetime = upper_bound_datetime
        self.paper_title = paper_title  # Title of the paper being evaluated (for server-side filtering)

        # Extract base URL for session sharing
        parsed_url = urlparse(self.search_url)
        self.base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"

        # Get shared session for this base URL
        self.session = self._get_shared_session(self.base_url)
        if self.log_requests:
            logger.info(f"SearchArxivToolGroup initialized using shared session pool for {self.base_url}")

        super().__init__(name="SearchArxivToolGroup")

    def set_paper_title(self, paper_title: str):
        """
        Set the paper title for filtering out the same paper from search results.

        The filtering is done server-side by passing exclude_title to the API.

        Args:
            paper_title: The title of the paper being evaluated.
        """
        self.paper_title = paper_title
        if self.log_requests:
            logger.info(f"SearchArxivToolGroup: Set paper title for server-side filtering: {paper_title[:50]}...")

    @tool
    def semantic_search(self, query: str) -> str:
        """
        Execute a semantic search on arXiv papers.

        Args:
            query: The search query (should be descriptive, expand abbreviations).

        Returns:
            JSON string with search results or error message.
        """
        if query is None:
            return json.dumps({"result": "No query provided."})

        query = query.strip()
        if not query:
            return json.dumps({"result": "Empty query provided."})

        try:
            api_response, error_msg = call_semantic_search_api(
                retrieval_service_url=self.search_url,
                query=query,
                topk=self.topk,
                timeout=self.timeout,
                log_requests=self.log_requests,
                session=self.session,
                upper_bound_datetime=self.upper_bound_datetime,
                exclude_title=self.paper_title,  # Server-side filtering
            )
        except Exception as e:
            error_msg = f"API Request Exception during semantic search: {e}"
            api_response = None
            logger.error(f"Semantic search: {error_msg}")

        return self._format_response(api_response, error_msg, "Semantic search", query)

    @tool
    def author_search(self, authors: str) -> str:
        """
        Execute an author search on arXiv papers.

        Args:
            authors: Comma-separated list of author last names.

        Returns:
            JSON string with search results or error message.
        """
        if authors is None:
            return json.dumps({"result": "No authors provided."})

        authors = authors.strip()
        if not authors:
            return json.dumps({"result": "Empty authors list provided."})

        try:
            api_response, error_msg = call_author_search_api(
                retrieval_service_url=self.search_url,
                authors=authors,
                topk=self.topk,
                timeout=self.timeout,
                log_requests=self.log_requests,
                session=self.session,
                upper_bound_datetime=self.upper_bound_datetime,
                exclude_title=self.paper_title,  # Server-side filtering
            )
        except Exception as e:
            error_msg = f"API Request Exception during author search: {e}"
            api_response = None
            logger.error(f"Author search: {error_msg}")

        return self._format_response(api_response, error_msg, "Author search", authors)

    def _format_response(
        self,
        api_response: Optional[Dict[str, Any]],
        error_msg: Optional[str],
        search_type: str,
        query: str
    ) -> str:
        """Format the API response into a clean, readable string."""
        if error_msg:
            logger.error(f"{search_type}: API error occurred: {error_msg}")
            return f"Search error: {error_msg}"

        if not api_response:
            logger.error(f"{search_type}: Unknown API state.")
            return "Search request failed or timed out after retries."

        # Check for error in response
        if "error" in api_response:
            logger.error(f"{search_type}: API returned error: {api_response['error']}")
            return f"Search error: {api_response['error']}"

        try:
            raw_results = api_response.get("result", [])
            if not raw_results:
                if self.log_requests:
                    logger.info(f"{search_type}: No results found")
                return "No search results found."

            # Format results cleanly (filtering is done server-side)
            formatted_docs = []
            for retrieval in raw_results:
                for idx, doc_item in enumerate(retrieval):
                    # Handle both formats: {"document": {"contents": ...}} and {"contents": ...}
                    if "document" in doc_item:
                        content = doc_item["document"]["contents"].strip()
                    else:
                        content = doc_item.get("contents", "").strip()

                    formatted_docs.append(f"[{len(formatted_docs) + 1}] {content}")

            if self.log_requests:
                logger.info(f"{search_type}: Successful, got {len(formatted_docs)} results")

            if not formatted_docs:
                return "No search results found."

            return "\n\n".join(formatted_docs)

        except Exception as e:
            error_msg = f"Error processing search results: {e}"
            logger.error(f"{search_type}: {error_msg}")
            return error_msg
