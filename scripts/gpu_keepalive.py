#!/usr/bin/env python3
"""
Fault-tolerant GPU keepalive for SLURM allocations after a training crash.

Hardcoded workload:
  - Model:    Qwen2.5-7B-Instruct (HF cache, offline)
  - Prompts:  ICLR train data.json (any text-like field is harvested)
  - Strategy: vLLM inference loop in a subprocess, with matmul fallback if
              vLLM cannot start (e.g. offline cache miss, NCCL conflict).

Public entry point:
    keepalive(model_dir)
        - never raises
        - exits cleanly only when ${model_dir}/.keepalive.release exists
        - pauses (kills vLLM, frees GPU memory) when ${model_dir}/.keepalive.pause exists
        - resumes when the pause file is removed

Workflow during a crash:
    1. Training entry-point catches BaseException and calls keepalive(MODEL_DIR)
    2. SSH to the compute node (e.g. `ssh della-i23g4`)
    3. cat ${MODEL_DIR}/.error.txt
    4. touch ${MODEL_DIR}/.keepalive.pause     # frees GPUs for debug code
    5. <run debug / fix script on the freed GPUs>
    6. (optional) rm ${MODEL_DIR}/.keepalive.pause   # back to keepalive
    7. touch ${MODEL_DIR}/.keepalive.release    # exit, release the node
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

# ----------------------------- defaults ------------------------------------

QWEN_7B_CACHE_BASE = Path(
    "/scratch/gpfs/ZHUANGL/sk7524/hf/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots"
)
ICLR_TRAIN_JSON = Path(
    "/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer/data/"
    "iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_train/data.json"
)


def _safe_print(msg: str) -> None:
    try:
        print(f"[KEEPALIVE] {msg}", flush=True)
    except Exception:
        pass


def _resolve_qwen7b_path() -> str | None:
    """Find a snapshot dir under the HF cache for Qwen2.5-7B-Instruct."""
    try:
        if not QWEN_7B_CACHE_BASE.exists():
            return None
        snapshots = [p for p in QWEN_7B_CACHE_BASE.iterdir() if p.is_dir()]
        snapshots = [s for s in snapshots if (s / "config.json").exists()]
        if not snapshots:
            return None
        # Prefer the most recently modified
        snapshots.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return str(snapshots[0])
    except Exception as e:
        _safe_print(f"resolve_qwen7b_path failed: {e!r}")
        return None


# ----------------------------- vLLM subprocess management ------------------

class _VllmProc:
    """Wraps the inner vLLM subprocess; stop()/start() with hard guarantees."""

    def __init__(self, model_path: str, prompts: str, log_path: Path,
                 tensor_parallel_size: int):
        self.model_path = model_path
        self.prompts = prompts
        self.log_path = log_path
        self.tp = tensor_parallel_size
        self.proc: subprocess.Popen | None = None
        self.consec_failures = 0

    def start(self) -> bool:
        if self.proc is not None:
            return True
        inner = Path(__file__).parent / "gpu_keepalive_inner.py"
        cmd = [
            sys.executable, str(inner),
            "--model", self.model_path,
            "--prompts_jsonl", self.prompts,
            "--tensor_parallel_size", str(self.tp),
            "--cutoff_len", "24480",
            "--max_new_tokens", "128",
            "--gpu_memory_utilization", "0.85",
        ]
        try:
            log_fh = open(self.log_path, "a")
            log_fh.write(f"\n\n=== keepalive vLLM start at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            log_fh.flush()
            self.proc = subprocess.Popen(
                cmd, stdout=log_fh, stderr=subprocess.STDOUT,
                # Make the child its own process group so we can kill cleanly
                preexec_fn=os.setsid,
                env={**os.environ, "TRANSFORMERS_OFFLINE": "1", "PYTHONUNBUFFERED": "1"},
            )
            _safe_print(f"vLLM subprocess pid={self.proc.pid}, log={self.log_path}")
            return True
        except Exception as e:
            _safe_print(f"failed to spawn vLLM subprocess: {e!r}")
            self.proc = None
            return False

    def stop(self) -> None:
        if self.proc is None:
            return
        pid = self.proc.pid
        _safe_print(f"stopping vLLM subprocess pgid={pid}")
        # SIGTERM the entire process group to catch vLLM's worker spawn
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(os.getpgid(pid), sig)
            except ProcessLookupError:
                break
            except Exception as e:
                _safe_print(f"killpg({sig}) error: {e!r}")
            try:
                self.proc.wait(timeout=20)
                break
            except subprocess.TimeoutExpired:
                continue
            except Exception:
                break
        self.proc = None

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def returncode(self) -> int | None:
        return None if self.proc is None else self.proc.poll()


# ----------------------------- matmul fallback -----------------------------

def _matmul_fallback_loop(stop_predicate, log_every: float = 600.0) -> None:
    """In-process matmul keepalive. Runs until stop_predicate() returns True.
    Wrapped so it can never propagate."""
    try:
        import torch
    except Exception as e:
        _safe_print(f"torch import failed in fallback: {e!r}; sleeping forever")
        while not stop_predicate():
            time.sleep(10)
        return

    n = 0
    try:
        n = torch.cuda.device_count()
    except Exception:
        pass
    if n == 0:
        _safe_print("matmul fallback: no GPUs visible; sleeping forever")
        while not stop_predicate():
            time.sleep(10)
        return

    bufs = []
    try:
        for i in range(n):
            bufs.append(torch.randn(2048, 2048, device=f"cuda:{i}", dtype=torch.bfloat16))
    except Exception as e:
        _safe_print(f"matmul fallback: alloc failed: {e!r}; sleeping forever")
        while not stop_predicate():
            time.sleep(10)
        return

    _safe_print(f"matmul fallback active on {n} GPUs (bf16 2048x2048)")
    last = 0.0
    while not stop_predicate():
        for x in bufs:
            try:
                _ = x @ x
                torch.cuda.synchronize(x.device)
            except Exception as e:
                _safe_print(f"matmul fallback exception {e!r}; sleeping 30s")
                time.sleep(30)
                break
        if time.time() - last > log_every:
            _safe_print("matmul fallback alive")
            last = time.time()


# ----------------------------- public API ----------------------------------

def keepalive(
    model_dir: str | Path,
    *,
    inference_model: str | None = None,
    prompts_jsonl: str | Path | None = None,
    tensor_parallel_size: int | None = None,
    log_every: float = 600.0,
) -> None:
    """Block until ${model_dir}/.keepalive.release exists. Never raises."""
    try:
        model_dir = Path(model_dir)
        try:
            model_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        pause_file = model_dir / ".keepalive.pause"
        release_file = model_dir / ".keepalive.release"
        log_path = model_dir / ".keepalive.log"

        # Resolve model path & prompts
        model_path = inference_model or _resolve_qwen7b_path()
        prompts = str(prompts_jsonl) if prompts_jsonl else str(ICLR_TRAIN_JSON)
        if model_path is None or not Path(model_path).exists():
            _safe_print(f"Qwen-7B path not found ({model_path!r}); using matmul fallback only")
        if not Path(prompts).exists():
            _safe_print(f"prompts not found at {prompts}; inner will use synthetic prompts")

        if tensor_parallel_size is None:
            try:
                import torch
                tensor_parallel_size = max(1, torch.cuda.device_count())
            except Exception:
                tensor_parallel_size = 1

        _safe_print("=" * 70)
        _safe_print("GPU KEEPALIVE active")
        _safe_print(f"  model_dir : {model_dir}")
        _safe_print(f"  vLLM model: {model_path}")
        _safe_print(f"  prompts   : {prompts}")
        _safe_print(f"  TP size   : {tensor_parallel_size}")
        _safe_print(f"  pause     : touch {pause_file}")
        _safe_print(f"  resume    : rm    {pause_file}")
        _safe_print(f"  release   : touch {release_file}")
        _safe_print("=" * 70)

        vproc: _VllmProc | None = None
        if model_path is not None:
            vproc = _VllmProc(model_path, prompts, log_path, tensor_parallel_size)

        last_log = 0.0
        FALLBACK_AFTER_FAILURES = 3
        in_fallback = False
        fallback_paused = False

        while True:
            try:
                # Release: hard exit
                if release_file.exists():
                    _safe_print("release file detected, exiting cleanly")
                    if vproc is not None:
                        vproc.stop()
                    return

                want_pause = pause_file.exists()

                # ----- normal vLLM mode -----
                if vproc is not None and not in_fallback:
                    if want_pause and vproc.alive():
                        vproc.stop()
                    elif not want_pause and not vproc.alive():
                        # If subprocess exited, restart it (with caps)
                        if vproc.returncode() is not None and vproc.returncode() != 0:
                            vproc.consec_failures += 1
                            _safe_print(f"vLLM exited rc={vproc.returncode()} "
                                        f"(consec failures={vproc.consec_failures})")
                            if vproc.consec_failures >= FALLBACK_AFTER_FAILURES:
                                _safe_print(f"falling back to matmul keepalive permanently")
                                in_fallback = True
                                vproc = None
                                continue
                            time.sleep(20)  # backoff
                        ok = vproc.start()
                        if ok:
                            vproc.consec_failures = 0
                        else:
                            vproc.consec_failures += 1
                            time.sleep(20)
                            if vproc.consec_failures >= FALLBACK_AFTER_FAILURES:
                                _safe_print(f"falling back to matmul keepalive permanently")
                                in_fallback = True
                                vproc = None
                                continue

                # ----- matmul fallback mode -----
                if in_fallback or vproc is None:
                    # We can't easily pause/resume the in-process matmul mid-call.
                    # Instead, run a single short matmul tick and re-check files each loop.
                    if not want_pause:
                        # one tick of matmul
                        try:
                            import torch
                            n_gpu = torch.cuda.device_count()
                            if n_gpu > 0 and not fallback_paused:
                                # alloc once, reuse
                                if not hasattr(keepalive, "_fb_bufs"):
                                    keepalive._fb_bufs = [
                                        torch.randn(2048, 2048,
                                                    device=f"cuda:{i}",
                                                    dtype=torch.bfloat16)
                                        for i in range(n_gpu)
                                    ]
                                for x in keepalive._fb_bufs:
                                    _ = x @ x
                                    torch.cuda.synchronize(x.device)
                                fallback_paused = False
                        except Exception as e:
                            _safe_print(f"fallback matmul tick failed: {e!r}; sleeping 30")
                            time.sleep(30)
                    else:
                        # paused: drop fallback bufs to free GPU
                        if hasattr(keepalive, "_fb_bufs"):
                            try:
                                del keepalive._fb_bufs
                                import torch, gc
                                gc.collect()
                                torch.cuda.empty_cache()
                                _safe_print("paused (matmul mode) — freed GPU bufs")
                            except Exception:
                                pass
                        fallback_paused = True

                # Heartbeat log
                if time.time() - last_log > log_every:
                    state = "PAUSED" if want_pause else (
                        "ACTIVE-VLLM" if (vproc and vproc.alive()) else
                        ("ACTIVE-MATMUL" if in_fallback else "STARTING")
                    )
                    _safe_print(f"alive ({state}) at {time.strftime('%Y-%m-%d %H:%M:%S')}")
                    last_log = time.time()

                # Don't busy-loop: short tick rate when paused, slightly longer when active
                time.sleep(2 if not in_fallback else 0.5)

            except BaseException as inner_e:
                # Outer-loop guard: log everything, never raise.
                _safe_print(f"outer-loop exception: {inner_e!r}")
                try:
                    traceback.print_exc()
                except Exception:
                    pass
                # Best-effort cleanup of subprocess
                try:
                    if vproc is not None:
                        vproc.stop()
                except Exception:
                    pass
                time.sleep(30)

    except BaseException as outermost_e:
        # Last-ditch guard. Should not be reachable.
        _safe_print(f"OUTERMOST guard caught: {outermost_e!r}")
        try:
            # Park here forever rather than letting the calling process exit.
            while True:
                if Path(model_dir, ".keepalive.release").exists():
                    return
                time.sleep(30)
        except BaseException:
            return


# ----------------------------- CLI -----------------------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--inference_model", default=None)
    ap.add_argument("--prompts_jsonl", default=None)
    ap.add_argument("--tensor_parallel_size", type=int, default=None)
    args = ap.parse_args()
    keepalive(
        args.model_dir,
        inference_model=args.inference_model,
        prompts_jsonl=args.prompts_jsonl,
        tensor_parallel_size=args.tensor_parallel_size,
    )
