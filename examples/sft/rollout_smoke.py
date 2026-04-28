"""
Zero-shot rollout smoke for Qwen2.5-3B on the search_arxiv env.

Drives `SearchArxivEnv` against a live vLLM server (OpenAI-compatible API) and
a live arxiv_retriever server, captures full chat_history for N papers, and
emits side-by-side comparison with a replay example from the SFT dataset.

Purpose: eyeball the distribution shift. Does 3B zero-shot emit
``<think>…</think><ssearch>q</ssearch>`` cleanly? Does it use the python
tool? Does it hit the format check? What does it look like vs our replayed
expert trajectories?

Usage (inside the smoke sbatch):
  python examples/sft/rollout_smoke.py \
      --vllm_url http://127.0.0.1:6001/v1 \
      --model /path/to/qwen2.5-3b \
      --search_url http://127.0.0.1:8000/retrieve \
      --out /scratch/gpfs/ZHUANGL/sk7524/SkyRLMain/data/rollout_smoke/rollouts.jsonl \
      --n_rollouts 10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "examples" / "train" / "search_arxiv"))
sys.path.insert(0, str(_REPO / "skyrl-gym"))

from searchr1_arxiv_dataset import (  # noqa: E402
    ICLR_UPPER_BOUND,
    extract_title,
    strip_user_preamble,
    system_content_for_mode,
)
from skyrl_gym.envs.search_arxiv.env import SearchArxivEnv, SearchArxivEnvConfig  # noqa: E402
from skyrl_gym.envs.search_arxiv.utils import compute_format_score, compute_score  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rollout")


# ---------------------------------------------------------------------------
# Balanced paper picker
# ---------------------------------------------------------------------------


def pick_papers(source_json: str, n: int, seed: int = 0) -> List[Dict[str, Any]]:
    """Pick n papers stratified across (year, Accept/Reject) buckets."""
    log.info("loading %s", source_json)
    with open(source_json) as f:
        rows = json.load(f)
    buckets: Dict[Any, List[Dict[str, Any]]] = {}
    for r in rows:
        meta = r.get("_metadata") or {}
        year = meta.get("year")
        ans = meta.get("answer")
        if ans not in ("Accept", "Reject") or year is None:
            continue
        human = next((m["value"] for m in r.get("conversations", []) if m.get("from") == "human"), None)
        if not human:
            continue
        key = (year, ans)
        body = strip_user_preamble(human)
        buckets.setdefault(key, []).append(
            {
                "submission_id": meta.get("submission_id"),
                "year": year,
                "answer": ans,
                "paper_body": body,
                "title": extract_title(body),
            }
        )
    rng = random.Random(seed)
    keys = sorted(buckets.keys())
    per_bucket = max(1, n // max(1, len(keys)))
    out: List[Dict[str, Any]] = []
    for k in keys:
        rng.shuffle(buckets[k])
        out.extend(buckets[k][:per_bucket])
    rng.shuffle(out)
    return out[:n]


# ---------------------------------------------------------------------------
# vLLM chat completion
# ---------------------------------------------------------------------------


def vllm_chat(
    vllm_url: str,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 2048,
    temperature: float = 0.7,
    stop: Optional[List[str]] = None,
    timeout: int = 600,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if stop:
        payload["stop"] = stop
    r = requests.post(
        vllm_url.rstrip("/") + "/chat/completions",
        json=payload,
        timeout=timeout,
        headers={"Content-Type": "application/json"},
    )
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Env-driven rollout for one paper
# ---------------------------------------------------------------------------


def rollout_one(
    vllm_url: str,
    model: str,
    search_url: str,
    paper: Dict[str, Any],
    max_turns: int = 6,
    topk: int = 5,
    python_timeout: float = 10.0,
    max_tokens: int = 2048,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """Run one rollout with the real SearchArxivEnv and a real retrieval server."""
    target = paper["answer"]
    year = paper["year"]
    upper = ICLR_UPPER_BOUND.get(int(year)) if year is not None else None

    cfg = SearchArxivEnvConfig(
        search_url=search_url,
        topk=topk,
        timeout=30,
        max_turns=max_turns,
        search_enabled=True,
        format_weight=0.2,
        python_enabled=True,
        python_timeout=python_timeout,
    )
    extras = {
        "reward_spec": {"ground_truth": {"target": [target]}},
        "max_turns": max_turns,
        "search_enabled": True,
        "python_enabled": True,
        "upper_bound_datetime": upper,
        "exclude_title": paper.get("title"),
    }
    env = SearchArxivEnv(cfg, extras=extras)
    env.chat_history = [
        {"role": "system", "content": system_content_for_mode("search")},
        {"role": "user", "content": paper["paper_body"]},
    ]

    stop = ["</ssearch>", "</python>", "</answer>"]
    t0 = time.time()
    for turn in range(max_turns):
        # ask vLLM for the next assistant message
        try:
            action = vllm_chat(
                vllm_url, model, env.chat_history,
                max_tokens=max_tokens, temperature=temperature, stop=stop,
            )
        except Exception as e:
            log.warning("vllm error at turn %d: %s", turn, e)
            action = f"<think>vllm error: {e}</think>\n<answer>\\boxed{{Reject}}</answer>"

        # Restore the stop token that vLLM strips, so the env's regexes match.
        if "</ssearch>" not in action and "<ssearch>" in action:
            action += "</ssearch>"
        if "</python>" not in action and "<python>" in action:
            action += "</python>"
        if "</answer>" not in action and "<answer>" in action:
            action += "</answer>"

        out = env.step(action)
        if out["done"]:
            break

    elapsed = time.time() - t0
    fmt = compute_format_score(env.chat_history)
    chat_str = "".join(m.get("content", "") for m in env.chat_history)
    acc = compute_score(chat_str, {"target": [target]})
    return {
        "submission_id": paper["submission_id"],
        "year": year,
        "gt_target": target,
        "elapsed_sec": round(elapsed, 2),
        "format_score": fmt,
        "acc_reward": acc,
        "n_messages": len(env.chat_history),
        "n_assistant_turns": sum(1 for m in env.chat_history if m["role"] == "assistant"),
        "messages": [{"role": m["role"], "content": m["content"]} for m in env.chat_history],
    }


# ---------------------------------------------------------------------------
# Side-by-side report
# ---------------------------------------------------------------------------


def _ellipsis(s: str, n: int = 800) -> str:
    s = s.replace("\n", " \\n ")
    return s if len(s) <= n else s[:n] + "…"


def print_comparison_report(rollouts: List[Dict[str, Any]], replay_dir: Optional[str] = None) -> None:
    print("\n" + "=" * 80)
    print("ROLLOUT SUMMARY (zero-shot Qwen2.5-3B-Instruct on search_arxiv env)")
    print("=" * 80)
    fmt_sum = sum(r["format_score"] for r in rollouts)
    acc_sum = sum(r["acc_reward"] for r in rollouts)
    print(f"n rollouts         : {len(rollouts)}")
    print(f"mean format_score  : {fmt_sum/max(1,len(rollouts)):.3f}")
    print(f"mean accuracy      : {acc_sum/max(1,len(rollouts)):.3f}")
    print(f"mean n turns       : {sum(r['n_assistant_turns'] for r in rollouts)/max(1,len(rollouts)):.2f}")
    print(f"mean elapsed (sec) : {sum(r['elapsed_sec'] for r in rollouts)/max(1,len(rollouts)):.1f}")
    print()
    for r in rollouts:
        decision_ok = "✓" if r["acc_reward"] > 0.5 else "✗"
        fmt_ok = "✓" if r["format_score"] > 0.99 else "✗"
        print(f"  [{r['year']}] {r['submission_id']} gt={r['gt_target']} fmt={r['format_score']:.2f} {fmt_ok} acc={r['acc_reward']:.1f} {decision_ok} turns={r['n_assistant_turns']} t={r['elapsed_sec']}s")

    # Dump first rollout turn-by-turn
    if rollouts:
        r = rollouts[0]
        print("\n" + "=" * 80)
        print(f"FIRST ROLLOUT: {r['submission_id']} ({r['year']}, gt={r['gt_target']})")
        print("=" * 80)
        for i, m in enumerate(r["messages"]):
            print(f"\n[{i}] {m['role'].upper()} ({len(m['content'])} chars)")
            print(_ellipsis(m["content"], 600))

    # Side-by-side: show one replay example for comparison
    if replay_dir and os.path.isdir(replay_dir):
        try:
            import pyarrow.parquet as pq  # local
            import glob as _glob
            shards = sorted(_glob.glob(os.path.join(replay_dir, "*.parquet")))
            pf = pq.ParquetFile(shards[0])
            batch = next(pf.iter_batches(batch_size=32))
            msgs_col = batch.column("messages").to_pylist()
            sid_col = batch.column("submission_id").to_pylist()
            tgt_col = batch.column("target").to_pylist()
            # pick first one with >= 4 assistant turns
            for k in range(len(msgs_col)):
                if sum(1 for mm in msgs_col[k] if mm["role"] == "assistant") >= 4:
                    print("\n" + "=" * 80)
                    print(f"REPLAY EXAMPLE FOR COMPARISON: {sid_col[k]} (target={tgt_col[k]})")
                    print("=" * 80)
                    for i, m in enumerate(msgs_col[k]):
                        print(f"\n[{i}] {m['role'].upper()} ({len(m['content'])} chars)")
                        print(_ellipsis(m["content"], 600))
                    break
        except Exception as e:
            print(f"(could not load replay example: {e})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm_url", default="http://127.0.0.1:6001/v1")
    ap.add_argument("--model", required=True, help="Model name as served by vLLM (e.g. the local path).")
    ap.add_argument("--search_url", default="http://127.0.0.1:8000/retrieve")
    ap.add_argument(
        "--source_json",
        default=(
            "/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer/data/"
            "iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_train/data.json"
        ),
    )
    ap.add_argument("--n_rollouts", type=int, default=10)
    ap.add_argument("--max_turns", type=int, default=6)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--out", required=True, help="Output JSONL path.")
    ap.add_argument(
        "--replay_dir",
        default="/scratch/gpfs/ZHUANGL/sk7524/SkyRLMain/data/iclr_arxiv_sft/val",
        help="Directory of replay-dataset parquet shards for comparison.",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    papers = pick_papers(args.source_json, args.n_rollouts, seed=args.seed)
    log.info("selected %d papers (years: %s)", len(papers), sorted({p["year"] for p in papers}))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rollouts: List[Dict[str, Any]] = []
    with open(args.out, "w") as f:
        for i, paper in enumerate(papers):
            log.info("[%d/%d] %s (%s, gt=%s) ...", i + 1, len(papers), paper["submission_id"], paper["year"], paper["answer"])
            r = rollout_one(
                vllm_url=args.vllm_url,
                model=args.model,
                search_url=args.search_url,
                paper=paper,
                max_turns=args.max_turns,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            f.write(json.dumps(r) + "\n")
            f.flush()
            rollouts.append(r)
            log.info(
                "    fmt=%.2f acc=%.1f turns=%d %.1fs",
                r["format_score"], r["acc_reward"], r["n_assistant_turns"], r["elapsed_sec"],
            )

    print_comparison_report(rollouts, args.replay_dir)
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
