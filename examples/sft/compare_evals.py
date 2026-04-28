"""Aggregate per-chunk eval JSONL files into a side-by-side model comparison.

Reads the dumped_evals/eval_only/iclr_arxiv.jsonl files produced by each
``gen_eval_<tag>_chunk<k>`` run, concatenates chunks per model, and prints a
markdown table with accuracy / format-score / tool-use rate per model.

Per-paper join across models (same submission_ids in every model's chunks)
also surfaces verdict-flip rates between consecutive checkpoints.

Usage:
    python examples/sft/compare_evals.py [--ckpt_root <root>] [tag1 tag2 ...]

If no tags given, scans ckpt_root for any directory matching
``gen_eval_<tag>_chunk<k>`` and groups them.
"""

import argparse
import glob
import json
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

BOXED_RE = re.compile(r"\\boxed\{(Accept|Reject)\}")
SSEARCH_RE = re.compile(r"<ssearch>(.*?)</ssearch>", re.DOTALL)
PYTHON_RE = re.compile(r"<python>(.*?)</python>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def find_eval_files(ckpt_root: str, tag: str) -> List[str]:
    # Glob both `iclr_arxiv.jsonl` (text) and `iclr_arxiv_vl.jsonl` (VL) — the
    # dumped jsonl filename mirrors the data_source field on each row, which
    # differs between text and multimodal evals.
    pattern = os.path.join(
        ckpt_root, f"gen_eval_{tag}_chunk*", "exports",
        "dumped_evals", "eval_only", "iclr_arxiv*.jsonl",
    )
    return sorted(glob.glob(pattern))


def parse_row(row: dict) -> dict:
    out = row.get("output_response", "") or ""
    extras = row.get("env_extras", {}) or {}
    extra_info = extras.get("extra_info") or {}
    reward_spec = extras.get("reward_spec") or {}
    target = (reward_spec.get("ground_truth") or {}).get("target") or [None]
    gt = target[0]

    boxed = BOXED_RE.findall(out)
    pred = boxed[-1] if boxed else None

    # Tool use: did the model emit at least one <ssearch> AND it returned
    # a real (non-empty) query? We look at the assistant's emission only.
    ssearch_queries = [q.strip() for q in SSEARCH_RE.findall(out) if q.strip()]
    python_blocks = [c.strip() for c in PYTHON_RE.findall(out) if c.strip()]

    # Format check (intermediate-style): final assistant turn must have
    # think+answer with non-empty boxed.
    has_think = bool(THINK_RE.search(out))
    has_answer = bool(ANSWER_RE.search(out))
    fmt_ok = has_think and has_answer and pred is not None

    # Parrot detection: model copying placeholder error text from env.
    parrot = ("No valid tag found" in out)

    return {
        "submission_id": extra_info.get("submission_id"),
        "year": extra_info.get("year"),
        "gt": gt,
        "pred": pred,
        "correct": pred is not None and pred == gt,
        "boxed_present": pred is not None,
        "ssearch_count": len(ssearch_queries),
        "python_count": len(python_blocks),
        "tool_use": len(ssearch_queries) > 0 or len(python_blocks) > 0,
        "fmt_ok": fmt_ok,
        "parrot": parrot,
        "stop_reason": row.get("stop_reason"),
        "out_len": len(out),
    }


def aggregate(rows: List[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "accuracy": sum(r["correct"] for r in rows) / n,
        "boxed_rate": sum(r["boxed_present"] for r in rows) / n,
        "fmt_rate": sum(r["fmt_ok"] for r in rows) / n,
        "tool_rate": sum(r["tool_use"] for r in rows) / n,
        "ssearch_rate": sum(r["ssearch_count"] > 0 for r in rows) / n,
        "python_rate": sum(r["python_count"] > 0 for r in rows) / n,
        "parrot_rate": sum(r["parrot"] for r in rows) / n,
        "stop_rate": sum(r["stop_reason"] == "stop" for r in rows) / n,
        "pred_accept": sum(r["pred"] == "Accept" for r in rows) / n,
        "pred_reject": sum(r["pred"] == "Reject" for r in rows) / n,
        "pred_none":   sum(r["pred"] is None for r in rows) / n,
        "avg_out_len": sum(r["out_len"] for r in rows) / n,
    }


def fmt_pct(x: Optional[float]) -> str:
    return "—" if x is None else f"{x*100:5.1f}%"


def fmt_int(x: Optional[float]) -> str:
    return "—" if x is None else f"{int(x)}"


def render_table(per_model: Dict[str, dict]) -> str:
    cols = ["n", "accuracy", "boxed_rate", "fmt_rate", "tool_rate",
            "ssearch_rate", "python_rate", "parrot_rate", "stop_rate",
            "pred_accept", "pred_reject", "pred_none", "avg_out_len"]
    headers = ["Model"] + cols
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for tag, agg in per_model.items():
        cells = [tag]
        for c in cols:
            v = agg.get(c)
            if c == "n":
                cells.append(fmt_int(v))
            elif c == "avg_out_len":
                cells.append("—" if v is None else f"{int(v)}")
            else:
                cells.append(fmt_pct(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def verdict_flips(per_model_rows: Dict[str, List[dict]],
                  pairs: List[Tuple[str, str]]) -> List[str]:
    out = []
    for a, b in pairs:
        if a not in per_model_rows or b not in per_model_rows:
            continue
        a_by = {r["submission_id"]: r for r in per_model_rows[a] if r["submission_id"]}
        b_by = {r["submission_id"]: r for r in per_model_rows[b] if r["submission_id"]}
        common = sorted(set(a_by.keys()) & set(b_by.keys()))
        n = len(common)
        if n == 0:
            continue
        flips = sum(1 for sid in common if a_by[sid]["pred"] != b_by[sid]["pred"])
        agree = n - flips
        a_correct = sum(1 for sid in common if a_by[sid]["correct"])
        b_correct = sum(1 for sid in common if b_by[sid]["correct"])
        # Of the flips, did B agree with GT more than A did?
        flip_to_better = sum(
            1 for sid in common
            if a_by[sid]["pred"] != b_by[sid]["pred"]
            and b_by[sid]["correct"] and not a_by[sid]["correct"]
        )
        flip_to_worse = sum(
            1 for sid in common
            if a_by[sid]["pred"] != b_by[sid]["pred"]
            and a_by[sid]["correct"] and not b_by[sid]["correct"]
        )
        out.append(
            f"- {a} → {b}: n={n} agree={agree} ({agree/n*100:.1f}%) "
            f"flips={flips} (better={flip_to_better}, worse={flip_to_worse}); "
            f"{a} acc={a_correct/n*100:.1f}% → {b} acc={b_correct/n*100:.1f}%"
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_root", default="/scratch/gpfs/ZHUANGL/sk7524/ckpts")
    ap.add_argument("--tags", nargs="*",
                    default=["base", "step206", "step412", "vero"],
                    help="Model tags to look up under gen_eval_<tag>_chunk*")
    args = ap.parse_args()

    per_model_rows: Dict[str, List[dict]] = {}
    per_model_agg: Dict[str, dict] = {}
    for tag in args.tags:
        files = find_eval_files(args.ckpt_root, tag)
        if not files:
            print(f"# {tag}: no eval files yet")
            continue
        rows = []
        seen_chunks = []
        for fp in files:
            chunk = re.search(r"gen_eval_[^/]*_chunk(\d+)", fp).group(1)
            seen_chunks.append(chunk)
            with open(fp) as f:
                for line in f:
                    if not line.strip():
                        continue
                    rows.append(parse_row(json.loads(line)))
        per_model_rows[tag] = rows
        per_model_agg[tag] = aggregate(rows)
        print(f"# {tag}: chunks={','.join(seen_chunks)} rows={len(rows)}")

    print()
    print(render_table(per_model_agg))
    print()
    flip_pairs = [
        ("base", "step206"),
        ("step206", "step412"),
        ("step412", "vero"),
        ("base", "vero"),
    ]
    flip_lines = verdict_flips(per_model_rows, flip_pairs)
    if flip_lines:
        print("## Per-paper verdict flips")
        for line in flip_lines:
            print(line)


if __name__ == "__main__":
    main()
