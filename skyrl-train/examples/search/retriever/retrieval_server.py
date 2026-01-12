import json
import warnings
import logging
import difflib
from typing import List, Optional
import argparse

import faiss
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import datasets

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


def should_exclude_by_title(result_title: str, exclude_title: str, threshold: float = 0.98) -> bool:
    """
    Check if a result should be excluded due to title similarity.

    Uses difflib.SequenceMatcher for fuzzy string matching.

    Args:
        result_title: Title from the search result.
        exclude_title: Title to exclude (the paper being evaluated).
        threshold: Similarity threshold (default 0.98 = 98% match).

    Returns:
        True if the titles are too similar and should be excluded.
    """
    if not exclude_title or not result_title:
        return False
    similarity = difflib.SequenceMatcher(
        None,
        result_title.lower().strip(),
        exclude_title.lower().strip()
    ).ratio()
    return similarity > threshold

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("retrieval_server")


# Model-to-pooling mapping for auto-detection
MODEL2POOLING = {
    "e5": "mean",
    "bge": "cls",
    "contriever": "mean",
    "jina": "mean",
    "qwen": "last_token",  # Qwen3 uses last token pooling
}

# Global DataFrame for author search (loaded if --arxiv_metadata_path provided)
arxiv_df = None


def load_corpus(corpus_path: str):
    corpus = datasets.load_dataset("json", data_files=corpus_path, split="train", num_proc=4)
    return corpus


def read_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_docs(corpus, doc_idxs):
    results = [corpus[int(idx)] for idx in doc_idxs]
    return results


def load_model(model_path: str, use_fp16: bool = False):
    # model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.cuda()
    if use_fp16:
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    return model, tokenizer


def pooling(pooler_output, last_hidden_state, attention_mask=None, pooling_method="mean"):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    elif pooling_method == "last_token":
        # Qwen3-style last token pooling
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_state[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_state.shape[0]
            return last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
    else:
        raise NotImplementedError("Pooling method not implemented!")


class Encoder:
    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16, task_description=None):
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16
        self.task_description = task_description or "Given a web search query, retrieve relevant passages that answer the query"

        self.model, self.tokenizer = load_model(model_path=model_path, use_fp16=use_fp16)
        self.model.eval()

    @torch.no_grad()
    def encode(self, query_list: List[str], is_query=True) -> np.ndarray:
        # processing query for different encoders
        if isinstance(query_list, str):
            query_list = [query_list]

        if "e5" in self.model_name.lower():
            if is_query:
                query_list = [f"query: {query}" for query in query_list]
            else:
                query_list = [f"passage: {query}" for query in query_list]

        if "bge" in self.model_name.lower():
            if is_query:
                query_list = [
                    f"Represent this sentence for searching relevant passages: {query}" for query in query_list
                ]

        # Qwen3 query instruction
        if "qwen" in self.model_name.lower():
            if is_query:
                query_list = [f"Instruct: {self.task_description}\nQuery:{query}" for query in query_list]
            # Note: Documents don't need instruction prefix for Qwen3

        inputs = self.tokenizer(
            query_list, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt"
        )
        inputs = {k: v.cuda() for k, v in inputs.items()}

        if "T5" in type(self.model).__name__:
            # T5-based retrieval model
            decoder_input_ids = torch.zeros((inputs["input_ids"].shape[0], 1), dtype=torch.long).to(
                inputs["input_ids"].device
            )
            output = self.model(**inputs, decoder_input_ids=decoder_input_ids, return_dict=True)
            query_emb = output.last_hidden_state[:, 0, :]
        else:
            output = self.model(**inputs, return_dict=True)
            # Some models (e.g., Qwen3) don't have pooler_output
            pooler_output = getattr(output, 'pooler_output', None)
            query_emb = pooling(
                pooler_output, output.last_hidden_state, inputs["attention_mask"], self.pooling_method
            )
            if "dpr" not in self.model_name.lower():
                query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        query_emb = query_emb.detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32, order="C")

        return query_emb


class BaseRetriever:
    def __init__(self, config):
        self.config = config
        self.retrieval_method = config.retrieval_method
        self.topk = config.retrieval_topk

        self.index_path = config.index_path
        self.corpus_path = config.corpus_path

    def _search(self, query: str, num: int, return_score: bool, return_indices: bool = False):
        raise NotImplementedError

    def _batch_search(self, query_list: List[str], num: int, return_score: bool):
        raise NotImplementedError

    def search(self, query: str, num: int = None, return_score: bool = False, return_indices: bool = False):
        return self._search(query, num, return_score, return_indices)

    def batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        return self._batch_search(query_list, num, return_score)


class BM25Retriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        from pyserini.search.lucene import LuceneSearcher

        self.searcher = LuceneSearcher(self.index_path)
        self.contain_doc = self._check_contain_doc()
        if not self.contain_doc:
            self.corpus = load_corpus(self.corpus_path)
        self.max_process_num = 8

    def _check_contain_doc(self):
        return self.searcher.doc(0).raw() is not None

    def _search(self, query: str, num: int = None, return_score: bool = False, return_indices: bool = False):
        if return_indices:
            raise NotImplementedError("BM25Retriever does not support return_indices. Use DenseRetriever for arxiv.")
        if num is None:
            num = self.topk
        hits = self.searcher.search(query, num)
        if len(hits) < 1:
            if return_score:
                return [], []
            else:
                return []
        scores = [hit.score for hit in hits]
        if len(hits) < num:
            warnings.warn("Not enough documents retrieved!")
        else:
            hits = hits[:num]

        if self.contain_doc:
            all_contents = [json.loads(self.searcher.doc(hit.docid).raw())["contents"] for hit in hits]
            results = [
                {
                    "title": content.split("\n")[0].strip('"'),
                    "text": "\n".join(content.split("\n")[1:]),
                    "contents": content,
                }
                for content in all_contents
            ]
        else:
            results = load_docs(self.corpus, [hit.docid for hit in hits])

        if return_score:
            return results, scores
        else:
            return results

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        results = []
        scores = []
        for query in query_list:
            item_result, item_score = self._search(query, num, True)
            results.append(item_result)
            scores.append(item_score)
        if return_score:
            return results, scores
        else:
            return results


class DenseRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        self.index = faiss.read_index(self.index_path)
        if config.faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)

        self.corpus = load_corpus(self.corpus_path)
        self.encoder = Encoder(
            model_name=self.retrieval_method,
            model_path=config.retrieval_model_path,
            pooling_method=config.retrieval_pooling_method,
            max_length=config.retrieval_query_max_length,
            use_fp16=config.retrieval_use_fp16,
            task_description=config.task_description,
        )
        self.topk = config.retrieval_topk
        self.batch_size = config.retrieval_batch_size

    def _search(self, query: str, num: int = None, return_score: bool = False, return_indices: bool = False):
        if num is None:
            num = self.topk
        query_emb = self.encoder.encode(query)
        scores, idxs = self.index.search(query_emb, k=num)
        idxs = idxs[0]
        scores = scores[0]
        results = load_docs(self.corpus, idxs)
        if return_indices:
            return results, scores, idxs
        elif return_score:
            return results, scores
        else:
            return results

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk

        results = []
        scores = []
        for start_idx in range(0, len(query_list), self.batch_size):
            query_batch = query_list[start_idx : start_idx + self.batch_size]
            batch_emb = self.encoder.encode(query_batch)
            batch_scores, batch_idxs = self.index.search(batch_emb, k=num)

            batch_scores = batch_scores.tolist()
            batch_idxs = batch_idxs.tolist()
            # load_docs is not vectorized, but is a python list approach
            flat_idxs = sum(batch_idxs, [])
            batch_results = load_docs(self.corpus, flat_idxs)
            # chunk them back
            batch_results = [batch_results[i * num : (i + 1) * num] for i in range(len(batch_idxs))]
            results.extend(batch_results)
            scores.extend(batch_scores)
        if return_score:
            return results, scores
        else:
            return results


def get_retriever(config):
    if config.retrieval_method == "bm25":
        return BM25Retriever(config)
    else:
        return DenseRetriever(config)


#####################################
# FastAPI server below
#####################################


class Config:
    """
    Minimal config class (simulating your argparse)
    Replace this with your real arguments or load them dynamically.
    """

    def __init__(
        self,
        retrieval_method: str = "bm25",
        retrieval_topk: int = 10,
        index_path: str = "./index/bm25",
        corpus_path: str = "./data/corpus.jsonl",
        dataset_path: str = "./data",
        data_split: str = "train",
        faiss_gpu: bool = True,
        retrieval_model_path: str = "./model",
        retrieval_pooling_method: str = "mean",
        retrieval_query_max_length: int = 256,
        retrieval_use_fp16: bool = False,
        retrieval_batch_size: int = 128,
        task_description: str = None,
    ):
        self.retrieval_method = retrieval_method
        self.retrieval_topk = retrieval_topk
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.dataset_path = dataset_path
        self.data_split = data_split
        self.faiss_gpu = faiss_gpu
        self.retrieval_model_path = retrieval_model_path
        self.retrieval_pooling_method = retrieval_pooling_method
        self.retrieval_query_max_length = retrieval_query_max_length
        self.retrieval_use_fp16 = retrieval_use_fp16
        self.retrieval_batch_size = retrieval_batch_size
        self.task_description = task_description


class QueryRequest(BaseModel):
    query: Optional[str] = None
    last_name_author_search: Optional[str] = None  # Comma-separated last names
    topk: Optional[int] = None
    return_scores: bool = False
    lower_bound_datetime: Optional[str] = None  # ISO format: "2020-01-01"
    upper_bound_datetime: Optional[str] = None  # ISO format: "2023-12-31"
    exclude_title: Optional[str] = None  # Title to exclude (paper being evaluated)


app = FastAPI()


def build_date_mask(df, lower_bound, upper_bound):
    """Build a boolean mask for date filtering."""
    mask = pd.Series([True] * len(df))
    if lower_bound:
        mask = mask & (df['upload_date'] >= pd.to_datetime(lower_bound))
    if upper_bound:
        mask = mask & (df['upload_date'] <= pd.to_datetime(upper_bound))
    return mask


@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    """
    Endpoint that accepts a single query and performs retrieval.
    Input format:
    {
      "query": "What is Python?",
      "topk": 3,
      "return_scores": true,
      "lower_bound_datetime": "2020-01-01",
      "upper_bound_datetime": "2023-12-31"
    }
    Or for author search:
    {
      "last_name_author_search": "smith, jones",
      "topk": 5
    }
    """
    # Log incoming request
    logger.info(f"REQUEST: query='{request.query}', author_search='{request.last_name_author_search}', topk={request.topk}, date_filter=[{request.lower_bound_datetime}, {request.upper_bound_datetime}]")

    if not request.topk:
        request.topk = config.retrieval_topk  # fallback to default

    has_date_filter = request.lower_bound_datetime or request.upper_bound_datetime

    # Author search mode
    if request.last_name_author_search:
        if arxiv_df is None:
            logger.warning("RESPONSE: Author search not available - no metadata loaded")
            return {"error": "Author search not available. Server not started with --arxiv_metadata_path"}

        # Parse comma-separated last names
        last_names = [name.strip().lower() for name in request.last_name_author_search.split(',')]

        # Build filter: all names must be present
        mask = pd.Series([True] * len(arxiv_df))
        for name in last_names:
            mask = mask & arxiv_df['authors_lower'].str.contains(name, na=False)

        # Add date filter if specified
        if has_date_filter:
            mask = mask & build_date_mask(arxiv_df, request.lower_bound_datetime, request.upper_bound_datetime)

        # Add title exclusion filter if specified
        if request.exclude_title:
            mask = mask & ~arxiv_df['title'].apply(
                lambda t: should_exclude_by_title(t, request.exclude_title)
            )

        # Get matching rows (include title for filtering)
        matching_rows = arxiv_df.loc[mask].head(request.topk)
        results = [{"contents": row['content'], "title": row['title']} for _, row in matching_rows.iterrows()]

        if request.exclude_title:
            logger.info(f"RESPONSE [author_search]: {len(results)} results for authors '{request.last_name_author_search}' (excluding title similar to '{request.exclude_title[:50]}...')")
        else:
            logger.info(f"RESPONSE [author_search]: {len(results)} results for authors '{request.last_name_author_search}'")
        if results:
            logger.debug(f"  First result preview: {results[0]['contents'][:200]}...")
        return {"result": [results]}

    # Regular embedding search
    if not request.query:
        logger.warning("RESPONSE: Error - no query or author_search provided")
        return {"error": "Either 'query' or 'last_name_author_search' must be provided"}

    # Determine how many results to fetch
    needs_post_filtering = has_date_filter or request.exclude_title
    if needs_post_filtering:
        if arxiv_df is None:
            logger.warning("RESPONSE: Error - date/title filtering requires metadata")
            return {"error": "Date/title filtering requires --arxiv_metadata_path"}
        fetch_num = request.topk * 20  # Fetch 20x for post-filtering
    else:
        fetch_num = request.topk

    # Get results with indices (always need indices for DataFrame lookup)
    results, scores, idxs = retriever.search(
        query=request.query, num=fetch_num, return_score=True, return_indices=True
    )

    # Filter by date and title if needed, and lookup content from DataFrame
    if arxiv_df is not None:
        # Use DataFrame content (includes publish date and title for filtering)
        filtered_results = []
        filtered_scores = []
        excluded_count = 0
        lower_dt = pd.to_datetime(request.lower_bound_datetime) if request.lower_bound_datetime else None
        upper_dt = pd.to_datetime(request.upper_bound_datetime) if request.upper_bound_datetime else None
        for score, idx in zip(scores, idxs):
            row = arxiv_df.iloc[idx]
            if has_date_filter:
                doc_date = row['upload_date']
                if lower_dt and doc_date < lower_dt:
                    continue
                if upper_dt and doc_date > upper_dt:
                    continue
            # Filter by title similarity if exclude_title is specified
            if request.exclude_title and should_exclude_by_title(row['title'], request.exclude_title):
                excluded_count += 1
                continue
            filtered_results.append({"contents": row['content'], "title": row['title']})
            filtered_scores.append(score)
            if len(filtered_results) >= request.topk:
                break
        if excluded_count > 0:
            logger.info(f"Excluded {excluded_count} results due to title similarity with '{request.exclude_title[:50]}...'")
        results = filtered_results
        scores = filtered_scores
    # else: use corpus results as-is (no DataFrame loaded)

    # Format response
    resp = []
    if request.return_scores:
        combined = []
        for doc, score in zip(results, scores):
            combined.append({"document": doc, "score": float(score)})
        resp.append(combined)
    else:
        resp.append(results[:request.topk])

    # Log response
    num_results = len(resp[0]) if resp else 0
    logger.info(f"RESPONSE [semantic_search]: {num_results} results for query '{request.query[:100]}{'...' if len(request.query) > 100 else ''}'")
    if resp and resp[0]:
        first_doc = resp[0][0]
        if isinstance(first_doc, dict):
            content = first_doc.get('contents', first_doc.get('document', {}).get('contents', ''))[:200]
            logger.debug(f"  First result preview: {content}...")

    return {"result": resp}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Launch the local faiss retriever.")
    parser.add_argument(
        "--index_path", type=str, default="/home/peterjin/mnt/index/wiki-18/e5_Flat.index", help="Corpus indexing file."
    )
    parser.add_argument(
        "--corpus_path",
        type=str,
        default="/home/peterjin/mnt/data/retrieval-corpus/wiki-18.jsonl",
        help="Local corpus file.",
    )
    parser.add_argument("--topk", type=int, default=3, help="Number of retrieved passages for one query.")
    parser.add_argument("--retriever_name", type=str, default="e5", help="Name of the retriever model.")
    parser.add_argument(
        "--retriever_model", type=str, default="intfloat/e5-base-v2", help="Path of the retriever model."
    )
    parser.add_argument("--faiss_gpu", action="store_true", help="Use GPU for computation")
    parser.add_argument("--pooling_method", type=str, default=None,
        help="Pooling method (mean, cls, last_token). Auto-detected if not specified.")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--task_description", type=str, default=None,
        help="Custom task description for Qwen3 query instruction prefix.")
    parser.add_argument("--arxiv_metadata_path", type=str, default=None,
        help="Path to arxiv metadata jsonl for author search.")

    args = parser.parse_args()

    # Auto-detect pooling method based on retriever name
    if args.pooling_method is None:
        pooling_method = "mean"  # default
        for k, v in MODEL2POOLING.items():
            if k in args.retriever_name.lower():
                pooling_method = v
                break
    else:
        pooling_method = args.pooling_method

    print(f"Using pooling method: {pooling_method}")

    # 1) Build a config (could also parse from arguments).
    #    In real usage, you'd parse your CLI arguments or environment variables.
    config = Config(
        retrieval_method=args.retriever_name,  # or "dense"
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        retrieval_topk=args.topk,
        faiss_gpu=args.faiss_gpu,
        retrieval_model_path=args.retriever_model,
        retrieval_pooling_method=pooling_method,
        retrieval_query_max_length=256,
        retrieval_use_fp16=True,
        retrieval_batch_size=512,  # this is unused in the current retrieval implementation, which only supports single query
        task_description=args.task_description,
    )

    # 2) Instantiate a global retriever so it is loaded once and reused.
    retriever = get_retriever(config)

    # 3) Load arxiv metadata for author search if provided
    if args.arxiv_metadata_path:
        print("Loading arxiv metadata for author search...")
        arxiv_df = pd.read_json(args.arxiv_metadata_path, lines=True)
        # Extract upload date from first version
        arxiv_df['upload_date'] = pd.to_datetime(arxiv_df['versions'].map(lambda x: x[0]['created']))

        # Helper to truncate abstract to max_chars
        def truncate_abstract(abstract, max_chars=150):
            if pd.isna(abstract):
                return ""
            abstract = str(abstract).strip()
            if len(abstract) > max_chars:
                return abstract[:max_chars] + "..."
            return abstract

        # Create content column (same format as index building + publish date)
        # Abstract truncated to 150 chars to keep token count reasonable
        arxiv_df['content'] = (
            '"' + arxiv_df['title'].str.replace('\n', ' ').str.strip() + '"\n' +
            'Authors:' + arxiv_df['authors'].str.replace('\n', ' ').str.strip() + '\n\n' +
            'Abstract:' + arxiv_df['abstract'] + '\n\n' +
            'DOI:' + arxiv_df['doi'].fillna('N/A') + '\n' +
            'Publish Date:' + arxiv_df['upload_date'].dt.strftime('%Y-%m-%d')
        )
        arxiv_df['authors_lower'] = arxiv_df['authors'].str.lower()
        print(f"Loaded {len(arxiv_df)} arxiv papers for author/date search")

    # 4) Launch the server.
    print(f"Starting server on port {args.port}...")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
