"""
Convert the ICLR accept/reject ShareGPT dataset into the SkyRL `search_arxiv`
parquet schema.

Input JSON (one file per split):
  [
    {
      "conversations": [
        {"from": "system", "value": "..."},
        {"from": "human",  "value": "I am giving you a paper. ... # TITLE ... ABSTRACT ..."},
        {"from": "gpt",    "value": "\\boxed{Accept}"}    # optional; target also in metadata
      ],
      "_metadata": {
        "answer": "Accept"|"Reject", "authors": "[\"First Last\", ...]",
        "year": 2020|2023|2025|2026, "submission_id": "...", ...
      }
    },
    ...
  ]

Output parquet columns (matches examples/train/search/searchr1_dataset.py):
  data_source, prompt, ability, env_class, reward_spec, extra_info, metadata
"""

import argparse
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_CONTENT = (
    "You are an expert academic reviewer. For each ICLR paper you are given, "
    "decide whether it was accepted or rejected. You may call arXiv retrieval tools "
    "to gather context on related work before making your decision.\n\n"
    "Tools:\n"
    "  <ssearch>your query</ssearch>  — semantic search over arXiv.\n"
    "  <asearch>lastname1,lastname2</asearch>  — author search (comma-separated last names).\n"
    "Retrieval results come back between <information> and </information>.\n\n"
    "When ready, output exactly one of:\n"
    "  <answer>Accept</answer>\n"
    "  <answer>Reject</answer>\n"
    "ICLR accepts roughly 30% of submissions."
)

# Per-conference submission cutoff (papers uploaded after this date should be
# hidden from retrieval). Approximate ICLR submission deadlines.
ICLR_UPPER_BOUND = {
    2020: "2019-09-25",
    2023: "2022-09-28",
    2025: "2024-10-01",
    2026: "2025-10-01",
}

TITLE_RE = re.compile(r"#\s*([^\n]+)", re.MULTILINE)
PREAMBLE_RE = re.compile(r"I am giving you a paper[\s\S]*?Question:\s*", re.MULTILINE)


def extract_title(paper_body: str) -> Optional[str]:
    """Best-effort: first markdown H1 in the user prompt."""
    m = TITLE_RE.search(paper_body)
    if not m:
        return None
    title = m.group(1).strip()
    # Sometimes the heading is "ABSTRACT" etc. — skip if it's an all-caps section header of 1-2 words.
    if len(title.split()) <= 2 and title.isupper():
        return None
    return title


def parse_authors(authors_raw: Any) -> List[str]:
    """`_metadata.authors` is a JSON-encoded string list. Return a list of last names."""
    if not authors_raw:
        return []
    try:
        if isinstance(authors_raw, str):
            names = json.loads(authors_raw)
        else:
            names = list(authors_raw)
    except json.JSONDecodeError:
        return []
    last_names = []
    for n in names:
        if not isinstance(n, str):
            continue
        tokens = n.strip().split()
        if tokens:
            last_names.append(tokens[-1])
    return last_names


def normalize_answer(ans: Any) -> Optional[str]:
    if not isinstance(ans, str):
        return None
    a = ans.strip().lower()
    if a in ("accept", "accepted"):
        return "Accept"
    if a in ("reject", "rejected"):
        return "Reject"
    return None


def process_row(row: Dict[str, Any], split: str, index: int) -> Optional[Dict[str, Any]]:
    conv = row.get("conversations") or []
    meta = row.get("_metadata") or {}

    target = normalize_answer(meta.get("answer"))
    if target is None:
        return None

    # Extract the human turn (paper body).
    human_turn = next((m["value"] for m in conv if m.get("from") == "human"), None)
    if not human_turn:
        return None

    title = extract_title(human_turn)
    year = meta.get("year")
    upper = ICLR_UPPER_BOUND.get(int(year)) if isinstance(year, (int, float, str)) and str(year).isdigit() else None

    # Replace the original "\boxed{Accept}" instruction wording with SkyRL's tag-based grammar.
    # We keep the paper body unchanged; the system message instructs the model on the grammar.
    user_content = human_turn

    prompt = [
        {"role": "system", "content": SYSTEM_CONTENT},
        {"role": "user", "content": user_content},
    ]

    reward_spec = {"ground_truth": {"target": [target]}}

    extra_info = {
        "index": index,
        "split": split,
        "need_tools_kwargs": False,
        "title": title,
        "exclude_title": title,
        "upper_bound_datetime": upper,
        "submission_id": meta.get("submission_id"),
        "year": year,
        "authors_last_names": parse_authors(meta.get("authors")),
    }

    return {
        "data_source": "iclr_arxiv",
        "prompt": prompt,
        "ability": "search_arxiv",
        "env_class": "search_arxiv",
        "reward_spec": reward_spec,
        "extra_info": extra_info,
        "metadata": meta,
    }


def convert_split(input_json: str, output_parquet: str, split_name: str, max_rows: Optional[int]) -> int:
    logger.info(f"Loading {input_json}")
    with open(input_json, "r") as f:
        data = json.load(f)
    if max_rows is not None:
        data = data[:max_rows]

    processed = []
    dropped = 0
    for i, row in enumerate(data):
        out = process_row(row, split_name, i)
        if out is None:
            dropped += 1
            continue
        processed.append(out)
    logger.info(f"  {len(processed)} rows kept, {dropped} dropped")

    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    df = pd.DataFrame(processed)
    df.to_parquet(output_parquet, index=False)
    logger.info(f"  wrote {output_parquet}")
    return len(processed)


DEFAULT_INPUT_BASE = (
    "/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer/data/"
    "iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered"
)
DEFAULT_OUTPUT_DIR = "/scratch/gpfs/ZHUANGL/sk7524/SkyRLMain/data/iclr_arxiv_text"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_base", default=DEFAULT_INPUT_BASE, help="Prefix; script appends _{split}/data.json")
    ap.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    ap.add_argument("--max_rows", type=int, default=None)
    args = ap.parse_args()

    for split in args.splits:
        in_path = f"{args.input_base}_{split}/data.json"
        out_path = os.path.join(args.output_dir, f"{split}.parquet")
        if not os.path.exists(in_path):
            logger.warning(f"skipping {split}: {in_path} not found")
            continue
        convert_split(in_path, out_path, split, args.max_rows)


if __name__ == "__main__":
    main()
