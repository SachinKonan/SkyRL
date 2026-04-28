#!/usr/bin/env python3
"""
Inner vLLM keepalive — load Qwen-7B-Instruct, run inference forever on ICLR
train prompts. Stdout/err go to a log file passed via env var.

Standalone (no project imports beyond vLLM). Killed externally via SIGTERM.

This script is INTENTIONALLY allowed to crash. The outer keepalive watches
the subprocess and restarts or falls back to matmul mode.
"""
import argparse
import json
import os
import random
import signal
import sys
import time
from pathlib import Path


def _ext(p, key, default=None):
    """Best-effort field extraction across data formats."""
    if isinstance(p, dict):
        return p.get(key, default)
    return default


def _load_prompts(path: str, n: int = 64) -> list[str]:
    """Pull short text prompts from a data.json. Robust to schema variations."""
    try:
        data = json.load(open(path))
    except Exception as e:
        print(f"[INNER] couldn't load {path}: {e}; using fallback prompts", flush=True)
        return [
            "Summarize the key contribution of this paper in one sentence.",
            "What is the main weakness of this method?",
            "List three follow-up experiments.",
        ] * (n // 3 + 1)

    random.seed(0)
    sample = random.sample(data, min(n, len(data)))
    out: list[str] = []
    for ex in sample:
        # Try common shapes: conversations / messages / instruction+input / prompt
        text = None
        convs = _ext(ex, "conversations") or _ext(ex, "messages")
        if isinstance(convs, list):
            for c in convs:
                if not isinstance(c, dict):
                    continue
                role = c.get("from") or c.get("role")
                val = c.get("value") or c.get("content")
                if role in ("human", "user") and isinstance(val, str):
                    text = val
                    break
        if not text:
            text = _ext(ex, "prompt") or _ext(ex, "instruction")
        if isinstance(text, str) and text.strip():
            # cap to a reasonable length to avoid one massive prompt
            out.append(text[:8000])

    if not out:
        out = ["Hello, can you describe yourself?"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts_jsonl", required=True)
    ap.add_argument("--num_prompts", type=int, default=64)
    ap.add_argument("--cutoff_len", type=int, default=24480)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    args = ap.parse_args()

    # Be polite on SIGTERM: just exit cleanly. Outer manages the lifecycle.
    def _on_sigterm(sig, frame):
        print(f"[INNER] SIGTERM received, exiting", flush=True)
        sys.exit(0)
    signal.signal(signal.SIGTERM, _on_sigterm)

    print(f"[INNER] starting vLLM with model={args.model}", flush=True)
    print(f"[INNER] prompts={args.prompts_jsonl}", flush=True)

    try:
        from vllm import LLM, SamplingParams
    except Exception as e:
        print(f"[INNER] failed to import vllm: {e}", flush=True)
        return 2

    prompts = _load_prompts(args.prompts_jsonl, n=args.num_prompts)
    print(f"[INNER] prompt pool: {len(prompts)}", flush=True)

    try:
        llm = LLM(
            model=args.model,
            trust_remote_code=True,
            max_model_len=args.cutoff_len + args.max_new_tokens,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            disable_log_stats=True,
            enforce_eager=False,
        )
    except Exception as e:
        print(f"[INNER] LLM init failed: {e}", flush=True)
        return 3

    sp = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.7, top_p=0.95)
    print(f"[INNER] vLLM engine ready, looping forever", flush=True)

    batch_size = 4
    i = 0
    t_last = time.time()
    n_done = 0
    while True:
        batch = [prompts[(i + k) % len(prompts)] for k in range(batch_size)]
        i = (i + batch_size) % len(prompts)
        try:
            llm.generate(batch, sp, use_tqdm=False)
            n_done += batch_size
        except Exception as e:
            print(f"[INNER] generate exception: {e}; sleeping 5s", flush=True)
            time.sleep(5)
            continue
        if time.time() - t_last > 300:
            rate = n_done / (time.time() - t_last)
            print(f"[INNER] alive at {time.strftime('%H:%M:%S')}, "
                  f"{n_done} prompts in last 5min ({rate:.1f}/s)", flush=True)
            t_last = time.time()
            n_done = 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException as e:
        print(f"[INNER] uncaught: {e}", flush=True)
        sys.exit(1)
