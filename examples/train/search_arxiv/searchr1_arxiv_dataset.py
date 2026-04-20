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

_REVIEW_ROLES_PREAMBLE = """You are an academic reviewer assistant for the ICLR conference, tasked with evaluating research papers. Your task is to predict whether a paper will be accepted or rejected.
 - Note: ICLR generally has a ~30% acceptance rate

Simulate an OpenReview-review amongst three expert reviewers with distinct roles:
- Reviewer 1 (Advocate): Find the strongest arguments FOR acceptance. What is novel, well-executed, or impactful about this work?
- Reviewer 2 (Critic): Find the strongest arguments AGAINST acceptance. What are the weaknesses, missing baselines, overclaims, or lack of novelty?
- Reviewer 3 (Calibrator): Compare this paper to typical ICLR papers at the accept/reject boundary. Is it above or below that bar?

Each reviewer responds in two sentences to each of:
1. The content and contribution of the paper
2. Whether the investigation is executed well and the paper is high quality
3. Whether the paper is novel w.r.t. existing work

IMPORTANT GUIDELINES:
- Reference specific elements of the paper: cite sections, figures, tables, equations, or quote key claims when making your arguments.
- When claiming something is novel, explain WHY relative to specific prior work (e.g., "Unlike [method X] which does Y, this paper does Z because...").
- When claiming a weakness, point to the specific missing experiment, flawed assumption, or unsupported claim.

For calibration, here is how a similar panel evaluated two papers:

ACCEPTED: "A Meta-Transfer Objective for Learning to Disentangle Causal Mechanisms"
- Advocate highlighted a genuinely novel connection between transfer learning and causal inference, with strong theoretical grounding in independent mechanisms.
- Critic noted limited scale of experiments but acknowledged the conceptual contribution was substantial.
- Calibrator: clearly above the accept bar — introduces a new paradigm rather than incremental improvement.

REJECTED: "Scalable Deep Neural Networks via Low-Rank Matrix Factorization"
- Advocate noted practical utility of flexible model sizing after training and reasonable empirical results on standard benchmarks.
- Critic argued SVD-based compression is well-studied, improvements over existing factorization methods are marginal, and no comparison to recent pruning approaches.
- Calibrator: below the bar — addresses a real problem but lacks the novelty and rigor expected at a top venue.

After the three reviews, you are the meta-reviewer. Weigh all perspectives carefully, considering both the Advocate's and Critic's arguments on their merits."""

_ANSWER_INSTRUCTION_NOSEARCH = (
    "\n\nOutput your final answer: <answer> \\boxed{Accept} </answer> or "
    "<answer> \\boxed{Reject} </answer>."
)

_ANSWER_INSTRUCTION_SEARCH = (
    "\n\nYou may call arXiv retrieval tools before answering:\n"
    "  <ssearch>your query</ssearch>  — semantic search over arXiv.\n"
    "  <asearch>lastname1,lastname2</asearch>  — author search (comma-separated last names).\n"
    "Retrieval results come back between <information> and </information>.\n\n"
    "Output your final answer: <answer> \\boxed{Accept} </answer> or "
    "<answer> \\boxed{Reject} </answer>."
)

# Preamble emitted by LLaMA-Factory's SFT dataset at the top of each user turn —
# conflicts with our answer grammar ("\boxed{Accept}" as bare text, no <answer> tag).
# Strip it verbatim so the user turn starts cleanly at the paper's "# TITLE".
_LLAMAFACTORY_PREAMBLE = (
    "I am giving you a paper. I want to predict its acceptance outcome at ICLR.\n"
    " - Your answer will either be: \\boxed{Accept} or \\boxed{Reject}\n"
    " - Note: ICLR generally has a ~30% acceptance rate\n\n"
)


def system_content_for_mode(prompt_mode: str) -> str:
    if prompt_mode == "search":
        return _REVIEW_ROLES_PREAMBLE + _ANSWER_INSTRUCTION_SEARCH
    if prompt_mode == "nosearch":
        return _REVIEW_ROLES_PREAMBLE + _ANSWER_INSTRUCTION_NOSEARCH
    raise ValueError(f"Unknown prompt_mode: {prompt_mode}")


def strip_user_preamble(user_turn: str) -> str:
    """Remove the LLaMA-Factory \\boxed{...} preamble so only paper content remains."""
    return user_turn.replace(_LLAMAFACTORY_PREAMBLE, "", 1).lstrip("\n")


# Backward-compat: default prompt (used if a caller imports SYSTEM_CONTENT).
SYSTEM_CONTENT = system_content_for_mode("nosearch")

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


def process_row(row: Dict[str, Any], split: str, index: int, prompt_mode: str = "nosearch") -> Optional[Dict[str, Any]]:
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

    # Strip the LLaMA-Factory SFT preamble (mentions \boxed grammar as bare text,
    # conflicts with our <answer>-wrapped form). System prompt carries the task framing.
    user_content = strip_user_preamble(human_turn)

    prompt = [
        {"role": "system", "content": system_content_for_mode(prompt_mode)},
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


def convert_split(
    input_json: str, output_parquet: str, split_name: str, max_rows: Optional[int], prompt_mode: str = "nosearch"
) -> int:
    logger.info(f"Loading {input_json}")
    with open(input_json, "r") as f:
        data = json.load(f)
    if max_rows is not None:
        data = data[:max_rows]

    processed = []
    dropped = 0
    for i, row in enumerate(data):
        out = process_row(row, split_name, i, prompt_mode=prompt_mode)
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
    ap.add_argument(
        "--prompt_mode",
        choices=["nosearch", "search"],
        default="nosearch",
        help="System prompt variant. 'nosearch' (default): review_roles only, no tool instructions. "
        "'search': review_roles + <ssearch>/<asearch> tool instructions.",
    )
    args = ap.parse_args()

    for split in args.splits:
        in_path = f"{args.input_base}_{split}/data.json"
        out_path = os.path.join(args.output_dir, f"{split}.parquet")
        if not os.path.exists(in_path):
            logger.warning(f"skipping {split}: {in_path} not found")
            continue
        convert_split(in_path, out_path, split, args.max_rows, prompt_mode=args.prompt_mode)


if __name__ == "__main__":
    main()
