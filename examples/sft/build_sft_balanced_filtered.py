"""Filter + year-balance the existing iclr_arxiv_sft_yearbalanced dataset.

Inputs:
- ``${SCRATCH_ROOT}/SkyRLMain/data/iclr_arxiv_sft_yearbalanced/{train,val}/``
  (output of ``build_sft_from_expert_traces.py``, with columns
  ``messages, submission_id, year, turn_idx, total_turns, target``)

Filters (all must pass):
1. ``total_turns ≤ 7``  (cut the long-tool-loop tail)
2. cumulative tokens ≤ 32,000 under Qwen2.5-7B-Instruct chat template
3. last assistant message contains ``\\boxed{Accept|Reject}`` matching
   ``target`` (defensive — should be 100% on the source dataset)

Year-balance:
- For each year, take ``min(Accept_count, Reject_count)`` from each side
- Stratified shuffle (seed=42) within year, then concatenate

Output:
- ``${SCRATCH_ROOT}/SkyRLMain/data/iclr_arxiv_sft_yearbalanced_short/{train,val}/``
  (sharded parquets, drop-in for HF ``load_dataset(<dir>)``)
"""

import argparse
import logging
import os
import random
import re
from collections import defaultdict
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_SRC = "/scratch/gpfs/ZHUANGL/sk7524/SkyRLMain/data/iclr_arxiv_sft_yearbalanced"
DEFAULT_OUT = "/scratch/gpfs/ZHUANGL/sk7524/SkyRLMain/data/iclr_arxiv_sft_yearbalanced_short"
DEFAULT_TOK = "/scratch/gpfs/ZHUANGL/sk7524/hf/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"

BOXED = re.compile(r"\\boxed\{(Accept|Reject)\}")


def _filter_split(src_dir: str, tok, max_turns: int, max_cum: int) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(src_dir, split="train")
    log.info(f"loaded {len(ds)} rows from {src_dir}")

    kept: List[Dict[str, Any]] = []
    n_turns_drop = 0
    n_cum_drop = 0
    n_boxed_drop = 0
    n_match_drop = 0
    for r in ds:
        msgs = r["messages"]
        n_assist = sum(1 for m in msgs if m["role"] == "assistant")
        if n_assist > max_turns:
            n_turns_drop += 1
            continue
        rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        cum = len(tok(rendered, add_special_tokens=False)["input_ids"])
        if cum > max_cum:
            n_cum_drop += 1
            continue
        last = msgs[-1]
        if last["role"] != "assistant":
            n_boxed_drop += 1
            continue
        m = BOXED.search(last["content"])
        if not m:
            n_boxed_drop += 1
            continue
        if m.group(1) != r["target"]:
            n_match_drop += 1
            continue
        # Preserve original schema; cumulative length useful for downstream sanity
        out = {k: r[k] for k in ds.column_names}
        out["_cum_tokens"] = cum
        kept.append(out)

    log.info(
        f"filter funnel: turns_drop={n_turns_drop} cum_drop={n_cum_drop} "
        f"boxed_drop={n_boxed_drop} match_drop={n_match_drop}  → kept={len(kept)}"
    )
    return kept


def _year_balance(rows: List[Dict[str, Any]], seed: int = 42) -> List[Dict[str, Any]]:
    """For each year, take min(Accept, Reject) from each side; shuffle within year."""
    rng = random.Random(seed)
    by_year_target: Dict[int, Dict[str, List[Dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in rows:
        by_year_target[r["year"]][r["target"]].append(r)

    out: List[Dict[str, Any]] = []
    log.info("per-year balance (Accept / Reject → kept-per-side):")
    for year in sorted(by_year_target.keys()):
        a = by_year_target[year].get("Accept", [])
        rj = by_year_target[year].get("Reject", [])
        n = min(len(a), len(rj))
        rng.shuffle(a)
        rng.shuffle(rj)
        out.extend(a[:n])
        out.extend(rj[:n])
        log.info(f"  {year}: A={len(a):4d} R={len(rj):4d} → {n} of each")

    rng.shuffle(out)
    return out


def _write_sharded_parquet(rows: List[Dict[str, Any]], out_dir: str, shard_size: int = 4000) -> None:
    import glob

    import pyarrow as pa
    import pyarrow.parquet as pq

    os.makedirs(out_dir, exist_ok=True)
    for old in glob.glob(os.path.join(out_dir, "*.parquet")):
        os.remove(old)
    if not rows:
        pa.Table.from_pylist([]).to_pandas().to_parquet(
            os.path.join(out_dir, "shard_000.parquet"), index=False
        )
        return
    for i in range(0, len(rows), shard_size):
        chunk = rows[i : i + shard_size]
        table = pa.Table.from_pylist(chunk)
        path = os.path.join(out_dir, f"shard_{i // shard_size:03d}.parquet")
        pq.write_table(table, path, row_group_size=len(chunk), compression="snappy")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--tokenizer", default=DEFAULT_TOK)
    ap.add_argument("--max_turns", type=int, default=7)
    ap.add_argument("--max_cum", type=int, default=32000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--splits", nargs="+", default=["train", "val"],
        help="Which splits under --src to process (each becomes a subdir of --out).",
    )
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    for split in args.splits:
        src_dir = os.path.join(args.src, split)
        if not os.path.isdir(src_dir):
            log.warning(f"skip {split}: {src_dir} not found")
            continue
        log.info(f"=== processing {split} ===")
        kept = _filter_split(src_dir, tok, args.max_turns, args.max_cum)
        balanced = _year_balance(kept, seed=args.seed)
        log.info(f"after year-balance: {len(balanced)} rows")
        out_dir = os.path.join(args.out, split)
        _write_sharded_parquet(balanced, out_dir)
        log.info(f"wrote {len(balanced)} rows to {out_dir}")


if __name__ == "__main__":
    main()
