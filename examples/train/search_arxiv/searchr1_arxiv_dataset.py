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

_GENERIC_CORE = """Your response must include two parts:

1. **Reasoning**: A detailed, free-flowing chain of thought enclosed in `<think>` and `</think>` tags.
2. **Final Answer**: A clear, conversational explanation enclosed in `<answer>` and `</answer>` tags, with a concise final result enclosed in \\boxed{} notation if the question has a definitive answer.

---

### Reasoning Instructions

* The reasoning section must be inside `<think>` … `</think>` tags.
* The reasoning should resemble a stream of consciousness: explore, test hypotheses, backtrack if necessary, reflect, and refine.
* Let the reasoning flow naturally while progressing toward a conclusion.
* Use reasoning strategies such as:
  * **Planning** – outline possible angles of critique before committing.
  * **Exploration** – consider multiple interpretations of the paper's contribution, rigor, and novelty, even unlikely ones.
  * **Evaluation** – compare alternatives and verify against specific sections, figures, tables, equations, or claims.
  * **Reflection** – revisit earlier judgments if new evidence arises.
* {EXAMINE_LINE}
* If the signal is ambiguous, make a reasonable inference based on venue norms (~30% acceptance rate).
* End the reasoning once you are confident in the conclusion.

---

### Final Answer Instructions

* The answer section must be enclosed in `<answer>` … `</answer>` tags.
* The `<answer>` section should stand on its own: it must provide necessary context, the paper's setup, and justification so that a reader can understand and verify the conclusion without reading `<think>`.
  - Do NOT refer to the `<think>` section (avoid phrases like "as explained above" or "from the reasoning").
  - Restate the paper's essential claim/method in words before delivering the verdict.
* Default detail requirement: the `<answer>` must be a complete, readable review rather than a short summary.
* Boxed result: include exactly one `\\boxed{Accept}` or `\\boxed{Reject}` at the end.
* Consistency rule: The boxed value must match the conclusion supported by your explanation.

---

### Format Example

<think>
Detailed reasoning goes here...
</think>
<answer>
Self-contained explanation with the paper's setup, strengths, weaknesses, and calibration against ICLR norms.
Final verdict: \\boxed{Accept} or \\boxed{Reject}.
</answer>"""

_GENERIC_OPENER_TEXT = (
    "You are an expert academic reviewer for the ICLR conference, predicting whether a paper "
    "will be accepted or rejected. ICLR generally has a ~30% acceptance rate.\n\n"
)
_GENERIC_OPENER_VL = (
    "You are an expert academic reviewer for the ICLR conference, predicting whether a paper "
    "will be accepted or rejected. The paper is provided as page images. "
    "ICLR generally has a ~30% acceptance rate.\n\n"
)
_EXAMINE_TEXT = "Thoroughly examine the paper's abstract, method, experiments, and related work before narrowing down."
_EXAMINE_VL = (
    "Thoroughly examine the abstract, figures, tables, equations, and prose across the page images before narrowing down."
)

_TOOL_USE_APPENDIX = (
    "\n\n---\n\n"
    "### Tool Use\n\n"
    "Inside your reasoning you may call a semantic search over arXiv:\n"
    "  <ssearch>your query</ssearch>\n"
    "Retrieval results come back between <information> and </information>. "
    "Use them to calibrate against prior work."
)

# Rating-prompt variant: model emits a continuous 2-decimal-place confidence in
# [0.00, 1.00] inside \boxed{}. Reward is distance-based (1 - |pred - target|),
# so borderline papers still produce a meaningful gradient.
_GENERIC_RATING_CORE = """Your response must include two parts:

1. **Reasoning**: A detailed, free-flowing chain of thought enclosed in `<think>` and `</think>` tags.
2. **Final Answer**: A clear, conversational explanation enclosed in `<answer>` and `</answer>` tags, ending with a 2-decimal-place rating in \\boxed{} notation.

---

### Reasoning Instructions

* The reasoning section must be inside `<think>` … `</think>` tags.
* The reasoning should resemble a stream of consciousness: explore, test hypotheses, backtrack if necessary, reflect, and refine.
* Let the reasoning flow naturally while progressing toward a calibrated rating.
* Use reasoning strategies such as:
  * **Planning** – outline possible angles of critique before committing.
  * **Exploration** – consider multiple interpretations of the paper's contribution, rigor, and novelty.
  * **Evaluation** – compare alternatives and verify against specific sections, figures, tables, equations, or claims.
  * **Reflection** – revisit earlier judgments if new evidence arises.
* {EXAMINE_LINE}
* End the reasoning once you are confident in your rating.

---

### Final Answer Instructions

* The answer section must be enclosed in `<answer>` … `</answer>` tags.
* The `<answer>` section should stand on its own: it must provide necessary context, the paper's setup, and justification so that a reader can understand and verify the conclusion without reading `<think>`.
  - Do NOT refer to the `<think>` section.
  - Restate the paper's essential claim/method in words before delivering the rating.
* End the `<answer>` with exactly one rating in `\\boxed{X.XX}` notation, where X.XX is a 2-decimal-place number in [0.00, 1.00] representing your confidence the paper would be accepted at ICLR:
  - **0.00** = certain reject
  - **0.50** = unable to predict / borderline
  - **1.00** = certain accept
  - Calibration anchor: ICLR has ~30% acceptance rate, so the average paper would rate around 0.30.
* Use exactly 2 decimal places (e.g. `\\boxed{0.72}`, not `\\boxed{0.7}` or `\\boxed{0.723}`).

---

### Format Example

<think>
Detailed reasoning goes here, weighing strengths and weaknesses against ICLR norms...
</think>
<answer>
Self-contained explanation with the paper's setup, strengths, weaknesses, and calibration.
Final rating: \\boxed{0.62}.
</answer>"""

# Preamble emitted by LLaMA-Factory's SFT dataset at the top of each user turn —
# conflicts with our answer grammar ("\boxed{Accept}" as bare text, no <answer> tag).
# Strip it verbatim so the user turn starts cleanly at the paper's "# TITLE".
_LLAMAFACTORY_PREAMBLE = (
    "I am giving you a paper. I want to predict its acceptance outcome at ICLR.\n"
    " - Your answer will either be: \\boxed{Accept} or \\boxed{Reject}\n"
    " - Note: ICLR generally has a ~30% acceptance rate\n\n"
)


def system_content_for_mode(prompt_mode: str, *, vl: bool = False) -> str:
    if prompt_mode not in ("nosearch", "search", "nosearch_rating", "search_rating"):
        raise ValueError(f"Unknown prompt_mode: {prompt_mode}")
    opener = _GENERIC_OPENER_VL if vl else _GENERIC_OPENER_TEXT
    examine = _EXAMINE_VL if vl else _EXAMINE_TEXT
    core_template = _GENERIC_RATING_CORE if prompt_mode.endswith("_rating") else _GENERIC_CORE
    core = core_template.replace("{EXAMINE_LINE}", examine)
    tail = _TOOL_USE_APPENDIX if prompt_mode in ("search", "search_rating") else ""
    return opener + core + tail


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

    # Continuous rating in [0,1]; defaults to 0.5 (borderline) if missing so the
    # rating-reward path produces a near-zero gradient on unlabeled rows.
    pct_rating = float(meta.get("pct_rating", 0.5))
    # "Easy" curriculum tag: clear-Accept (pct_rating > 0.6) or clear-Reject
    # (pct_rating < 0.4). The CurriculumSampler reads this from extra_info.
    raw_decision = (meta.get("decision") or "").strip().lower()
    accept_decisions = {"accept", "poster", "spotlight", "oral"}
    is_easy = (raw_decision == "reject" and pct_rating < 0.4) or (
        raw_decision in accept_decisions and pct_rating > 0.6
    )

    # Strip the LLaMA-Factory SFT preamble (mentions \boxed grammar as bare text,
    # conflicts with our <answer>-wrapped form). System prompt carries the task framing.
    user_content = strip_user_preamble(human_turn)

    prompt = [
        {"role": "system", "content": system_content_for_mode(prompt_mode)},
        {"role": "user", "content": user_content},
    ]

    reward_spec = {
        "ground_truth": {
            "target": [target],
            "pct_rating": pct_rating,
        }
    }

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
        "pct_rating": pct_rating,
        "is_easy": is_easy,
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
        choices=["nosearch", "search", "nosearch_rating", "search_rating"],
        default="nosearch",
        help="System prompt variant. 'nosearch' (default): generic <think>/<answer>/\\boxed{Accept|Reject} prompt. "
        "'search': adds <ssearch> tool instructions. "
        "'nosearch_rating' / 'search_rating': model emits a 2-decimal-place rating \\boxed{X.XX} in [0,1] instead.",
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
