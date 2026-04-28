"""
Build SFT cold-start data from Qwen 3.5-27B expert review traces.

Strategy: throw away the expert's cheat-sheet user prompt, prepend our clean
search_arxiv RL prompt, and replay the expert's assistant-side trajectory
(think → search → python → synthesis → verdict) through `SearchArxivEnv` with
the tool backends mocked to return the pre-recorded retrieval/python responses.
The resulting `env.chat_history` is the SFT training example — byte-exact
compatible with the RL env's format check.

Inputs
------
  --traces_glob   shard JSONL pattern (dr_areachair_qwen35_shard*_of_66_*.jsonl)
  --source_json   clean paper bodies (data.json from sibling LLaMA-Factory repo)
  --out_dir       writes train.parquet + val.parquet with a `messages` column

Subcommands
-----------
  build     (default) produce the parquet files
  validate  reload parquet, run compute_format_score on samples, sanity check
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from omegaconf import DictConfig

# Make sibling modules importable without installing the package.
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
from skyrl_gym.envs.search_arxiv.utils import _turn_is_well_formatted, compute_format_score  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_sft")

# ---------------------------------------------------------------------------
# Filters (5-stage funnel, ported from analyze_qwen35_blind.py)
# ---------------------------------------------------------------------------

REQ_FIELDS = [
    "strengths",
    "weaknesses",
    "critical_disputes",
    "most_important_strength",
    "most_important_weakness",
    "relation_to_other_work",
    "paper_quality",
]
MIN_FIELD_LEN = 20


def is_schema_valid(trace: Dict[str, Any]) -> bool:
    rev = trace.get("review")
    if not isinstance(rev, dict):
        return False
    if not isinstance(rev.get("accept"), bool):
        return False
    for f in REQ_FIELDS:
        v = rev.get(f)
        if not isinstance(v, str) or len(v) < MIN_FIELD_LEN:
            return False
    return True


def count_tool_errors(trace: Dict[str, Any]) -> int:
    n = 0
    for m in trace.get("messages", []):
        if m.get("role") == "user" and "Tool error" in (m.get("content") or ""):
            n += 1
    return n


def approx_token_count(trace: Dict[str, Any]) -> int:
    total = sum(len(m.get("content", "") or "") for m in trace.get("messages", []))
    return total // 4  # 4 chars per token is a loose proxy (Qwen: ~3.3 chars/tok)


# ---------------------------------------------------------------------------
# Expert trace parsing
# ---------------------------------------------------------------------------

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
_TOOL_RESPONSE_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)


def _parse_expert_tool_call(assistant_content: str) -> Optional[Dict[str, Any]]:
    """Extract {name, arguments} from the first <tool_call> JSON block."""
    m = _TOOL_CALL_RE.search(assistant_content)
    if not m:
        return None
    try:
        obj = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or "name" not in obj:
        return None
    return obj


def _parse_expert_answer(assistant_content: str) -> Optional[str]:
    m = _ANSWER_RE.search(assistant_content)
    return m.group(1) if m else None


def _extract_think_body(assistant_content: str) -> str:
    """Everything before </think> is reasoning (Qwen3 chat template puts the
    opening tag outside the assistant message). If </think> is absent, take
    everything before the first <tool_call> or <answer>."""
    if "</think>" in assistant_content:
        return assistant_content.split("</think>", 1)[0].strip()
    for marker in ("<tool_call>", "<answer>"):
        if marker in assistant_content:
            return assistant_content.split(marker, 1)[0].strip()
    return assistant_content.strip()


def _unwrap_tool_response(user_content: str) -> str:
    m = _TOOL_RESPONSE_RE.search(user_content or "")
    if m:
        return m.group(1).strip()
    return (user_content or "").strip()


# ---------------------------------------------------------------------------
# Translation: expert assistant turn -> SkyRL assistant action string
# ---------------------------------------------------------------------------


def _render_think(body: str) -> str:
    body = body.strip() or "Analyzing the paper and planning the next search."
    return f"<think>\n{body}\n</think>"


def _render_ssearch(queries: List[str]) -> str:
    inner = "\n".join(q.strip() for q in queries if q.strip())
    return f"<ssearch>\n{inner}\n</ssearch>"


def _render_python(code: str) -> str:
    # Preserve exact code spacing (it's the training signal).
    return f"<python>\n{code}\n</python>"


def _synthesize_final_answer_prose(review: Dict[str, Any]) -> str:
    sections = [
        ("Strengths", review.get("strengths", "")),
        ("Weaknesses", review.get("weaknesses", "")),
        ("Critical disputes", review.get("critical_disputes", "")),
        ("Most important strength", review.get("most_important_strength", "")),
        ("Most important weakness", review.get("most_important_weakness", "")),
        ("Relation to other work", review.get("relation_to_other_work", "")),
        ("Paper quality", review.get("paper_quality", "")),
    ]
    parts = []
    for header, body in sections:
        if body and body.strip():
            parts.append(f"**{header}.** {body.strip()}")
    verdict = "Accept" if review.get("accept") else "Reject"
    parts.append(f"\\boxed{{{verdict}}}")
    return "\n\n".join(parts)


def translate_assistant_turn(
    assistant_content: str, is_final: bool, review: Optional[Dict[str, Any]]
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Translate one expert assistant turn into a SkyRL action string.

    Returns (action_string, metadata) or None if the turn is unusable.
    metadata contains {"kind": "ssearch"|"python"|"answer", ...}.
    """
    think_body = _extract_think_body(assistant_content)
    if is_final:
        if not review:
            return None
        answer_prose = _synthesize_final_answer_prose(review)
        action = f"{_render_think(think_body)}\n<answer>\n{answer_prose}\n</answer>"
        return action, {"kind": "answer"}

    tc = _parse_expert_tool_call(assistant_content)
    if not tc:
        return None
    name = tc.get("name")
    args = tc.get("arguments") or {}
    if name == "arxiv_search":
        q = args.get("query")
        if isinstance(q, str):
            queries = [q.strip()] if q.strip() else []
        elif isinstance(q, list):
            queries = [s.strip() for s in q if isinstance(s, str) and s.strip()]
        else:
            queries = []
        if not queries:
            return None
        action = f"{_render_think(think_body)}\n{_render_ssearch(queries)}"
        return action, {"kind": "ssearch", "queries": queries}
    if name == "PythonInterpreter":
        code = args.get("code")
        if not isinstance(code, str) or not code.strip():
            return None
        action = f"{_render_think(think_body)}\n{_render_python(code)}"
        return action, {"kind": "python"}
    # Unknown tool: skip this trace
    return None


# ---------------------------------------------------------------------------
# Replay through the real env (with mocked _execute_tool)
# ---------------------------------------------------------------------------


def _make_env(target_decision: str, year: Optional[int], title: Optional[str]) -> SearchArxivEnv:
    cfg = SearchArxivEnvConfig(
        search_url="http://mocked.invalid/retrieve",
        topk=5,
        timeout=30,
        max_turns=99,
        search_enabled=True,
        format_weight=0.0,
        python_enabled=True,
        python_timeout=10.0,
    )
    upper = ICLR_UPPER_BOUND.get(int(year)) if year is not None else None
    extras = {
        "reward_spec": {"ground_truth": {"target": [target_decision]}},
        "max_turns": 99,
        "search_enabled": True,
        "python_enabled": True,
        "upper_bound_datetime": upper,
        "exclude_title": title,
    }
    return SearchArxivEnv(cfg, extras=extras)


def replay_trace(
    trace: Dict[str, Any],
    clean_paper_body: str,
    year: Optional[int],
    title: Optional[str],
) -> Optional[List[Dict[str, str]]]:
    """Replay one expert trace; return the final env.chat_history or None on failure."""
    review = trace.get("review") or {}
    target = "Accept" if review.get("accept") else "Reject"

    expert_msgs = trace.get("messages", [])
    # Expert messages layout: system, user(paper+cheatsheet), assistant, user(tool), ...
    # Collect the assistant turns + the paired tool responses (if any).
    assistant_indices = [i for i, m in enumerate(expert_msgs) if m.get("role") == "assistant"]
    if not assistant_indices:
        return None

    # For each assistant turn, determine whether it's the final one and whether it has a
    # paired tool response (the user message that immediately follows it).
    translated: List[Tuple[str, Dict[str, Any], Optional[str]]] = []
    for k, idx in enumerate(assistant_indices):
        is_final = k == len(assistant_indices) - 1
        asst_content = expert_msgs[idx].get("content", "") or ""
        res = translate_assistant_turn(asst_content, is_final=is_final, review=review)
        if res is None:
            return None
        action, meta = res
        tool_response: Optional[str] = None
        if not is_final:
            nxt = expert_msgs[idx + 1] if idx + 1 < len(expert_msgs) else None
            if nxt is None or nxt.get("role") != "user":
                return None
            tool_response = _unwrap_tool_response(nxt.get("content") or "")
        translated.append((action, meta, tool_response))

    # Build env and seed with our clean prompt.
    try:
        env = _make_env(target, year, title)
    except Exception as e:
        log.warning("env init failed: %s", e)
        return None
    env.chat_history = [
        {"role": "system", "content": system_content_for_mode("search")},
        {"role": "user", "content": clean_paper_body},
    ]

    # Monkey-patch _execute_tool to return canned responses in order.
    pending = iter((meta.get("kind"), resp) for _, meta, resp in translated if resp is not None)

    def mock_exec(self, group_name, tool_name, tool_input):
        try:
            kind, raw = next(pending)
        except StopIteration:
            raise RuntimeError("No canned tool response available")
        # Wrap to match the tool's string return, so the env's own
        # _execute_tool wrapping yields <information>{...}</information>.
        return json.dumps({"result": raw}) if kind == "ssearch" else raw

    # Bind as instance method (intercept at the super() layer)
    env_tool_exec = env._execute_tool  # noqa: F841 (retain reference if we ever want to unpatch)

    def patched_execute_tool(group_name, tool_name, tool_input):
        out = mock_exec(env, group_name, tool_name, tool_input)
        return "\n<information>" + out + "</information>\n"

    env._execute_tool = patched_execute_tool  # type: ignore[assignment]

    for action, meta, _ in translated:
        out = env.step(action)
        if out["done"]:
            break

    return list(env.chat_history)


# ---------------------------------------------------------------------------
# Shard loading + dedup
# ---------------------------------------------------------------------------


def _shard_jobid(path: str) -> int:
    m = re.search(r"_(\d+)\.jsonl$", os.path.basename(path))
    return int(m.group(1)) if m else 0


def load_traces(
    traces_glob: str,
    python_glob_order: str = "jobid_desc",
) -> Iterable[Tuple[Dict[str, Any], str]]:
    """Yield (trace, shard_file) in dedup order: newest jobid first, so the first
    occurrence of a submission_id wins."""
    paths = sorted(glob.glob(traces_glob), key=_shard_jobid, reverse=True)
    seen: set[str] = set()
    for p in paths:
        try:
            with open(p) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    sid = r.get("submission_id")
                    if not sid or sid in seen:
                        continue
                    seen.add(sid)
                    yield r, p
        except Exception as e:
            log.warning("failed to read %s: %s", p, e)


# ---------------------------------------------------------------------------
# Source data.json -> {submission_id: clean_paper_body, year, title, gt}
# ---------------------------------------------------------------------------


def index_source(source_json: str) -> Dict[str, Dict[str, Any]]:
    log.info("loading source %s", source_json)
    with open(source_json) as f:
        rows = json.load(f)
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        meta = row.get("_metadata") or {}
        sid = meta.get("submission_id")
        if not sid:
            continue
        human = next((m["value"] for m in row.get("conversations", []) if m.get("from") == "human"), None)
        if not human:
            continue
        body = strip_user_preamble(human)
        title = extract_title(body)
        out[sid] = {
            "paper_body": body,
            "title": title,
            "year": meta.get("year"),
            "answer": meta.get("answer"),
        }
    log.info("source indexed: %d submission_ids", len(out))
    return out


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_examples(
    traces_glob: str,
    source_index: Dict[str, Dict[str, Any]],
    max_traces: Optional[int],
    max_token_budget: int,
    per_turn_split: bool = False,
) -> List[Dict[str, Any]]:
    """Produce a list of SFT examples.

    Default (``per_turn_split=False``): one example per trace, containing the
    full replayed chat. Pair this with the SFT trainer's ``multi_turn_loss=true``
    to get gradient on every assistant turn in a single forward pass.

    When ``per_turn_split=True``: emit N examples per trace (one per assistant
    turn), each a prefix that ends at that turn. Use when the trainer runs
    with the default last-turn-only loss.
    """
    examples: List[Dict[str, Any]] = []
    stats: Dict[str, int] = defaultdict(int)

    for trace, _shard in load_traces(traces_glob):
        if max_traces is not None and stats["kept_traces"] >= max_traces:
            break
        stats["total"] += 1
        if not is_schema_valid(trace):
            stats["drop_schema"] += 1
            continue
        sid = trace.get("submission_id")
        src = source_index.get(sid)
        if not src:
            stats["drop_no_source"] += 1
            continue
        # Ground-truth match
        pred_accept = bool(trace["review"]["accept"])
        gt_accept = (src.get("answer") == "Accept")
        if pred_accept != gt_accept:
            stats["drop_gt_mismatch"] += 1
            continue
        if approx_token_count(trace) > max_token_budget:
            stats["drop_too_long"] += 1
            continue
        if count_tool_errors(trace) > 1:
            stats["drop_tool_errors"] += 1
            continue

        chat = replay_trace(trace, src["paper_body"], src.get("year"), src.get("title"))
        if chat is None:
            stats["drop_replay_failed"] += 1
            continue

        # Validate format on the replayed chat — must be 1.0 or we skip.
        fmt = compute_format_score(chat)
        if fmt < 0.999:
            stats["drop_format"] += 1
            continue

        assistant_positions = [i for i, m in enumerate(chat) if m.get("role") == "assistant"]
        total_turns = len(assistant_positions)
        target = "Accept" if gt_accept else "Reject"

        if per_turn_split:
            for turn_idx, end in enumerate(assistant_positions):
                messages = chat[: end + 1]
                examples.append(
                    {
                        "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
                        "submission_id": sid,
                        "year": src.get("year"),
                        "turn_idx": turn_idx,
                        "total_turns": total_turns,
                        "target": target,
                    }
                )
        else:
            # One example per trace — full chat history. Loss on every assistant
            # turn is computed by the trainer via ``multi_turn_loss=true``.
            examples.append(
                {
                    "messages": [{"role": m["role"], "content": m["content"]} for m in chat],
                    "submission_id": sid,
                    "year": src.get("year"),
                    "turn_idx": total_turns - 1,  # kept for validator compat (always final)
                    "total_turns": total_turns,
                    "target": target,
                }
            )
        stats["kept_traces"] += 1
        if stats["kept_traces"] % 250 == 0:
            log.info("  progress: %d traces -> %d examples", stats["kept_traces"], len(examples))

    log.info("stats: %s", dict(stats))
    log.info("total examples: %d", len(examples))
    return examples


def _subsample_balanced(
    examples: List[Dict[str, Any]], per_bucket: int, seed: int
) -> List[Dict[str, Any]]:
    """Subsample up to ``per_bucket`` examples from each (year, target) bucket.

    Caps — does not oversample. Buckets smaller than ``per_bucket`` contribute
    all their examples.
    """
    rng = random.Random(seed)
    buckets: Dict[Tuple[Any, str], List[Dict[str, Any]]] = defaultdict(list)
    for ex in examples:
        buckets[(ex.get("year"), ex.get("target"))].append(ex)
    out: List[Dict[str, Any]] = []
    for key in sorted(buckets.keys(), key=lambda k: (str(k[0]), str(k[1]))):
        rows = buckets[key]
        rng.shuffle(rows)
        take = rows[:per_bucket]
        out.extend(take)
        log.info("  bucket %s: %d -> %d", key, len(rows), len(take))
    rng.shuffle(out)
    log.info("balanced subsample: %d examples total (per_bucket cap=%d)", len(out), per_bucket)
    return out


def _balance_within_year(
    examples: List[Dict[str, Any]], seed: int
) -> List[Dict[str, Any]]:
    """For each year, take ``min(n_accept, n_reject)`` of each class.

    Yields a dataset that is exactly 50/50 Accept/Reject within each year, with
    no arbitrary cap — every usable example from the minority class is kept,
    and the majority class is subsampled uniformly to match.
    """
    rng = random.Random(seed)
    by_year: Dict[Any, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: {"Accept": [], "Reject": []})
    for ex in examples:
        t = ex.get("target")
        if t in ("Accept", "Reject"):
            by_year[ex.get("year")][t].append(ex)

    out: List[Dict[str, Any]] = []
    for year in sorted(by_year.keys(), key=str):
        acc = by_year[year]["Accept"]
        rej = by_year[year]["Reject"]
        k = min(len(acc), len(rej))
        rng.shuffle(acc)
        rng.shuffle(rej)
        out.extend(acc[:k])
        out.extend(rej[:k])
        log.info("  year %s: Accept %d, Reject %d -> take %d of each", year, len(acc), len(rej), k)
    rng.shuffle(out)
    log.info("balance-within-year: %d examples total", len(out))
    return out


def split_and_write(
    examples: List[Dict[str, Any]],
    out_dir: str,
    val_frac: float = 0.02,
    seed: int = 42,
) -> Tuple[int, int]:
    """Stratified val split by year, write train.parquet + val.parquet."""
    rng = random.Random(seed)
    by_year: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for ex in examples:
        by_year[ex.get("year")].append(ex)
    train, val = [], []
    for yr, rows in by_year.items():
        rng.shuffle(rows)
        n_val = max(1, int(len(rows) * val_frac))
        val.extend(rows[:n_val])
        train.extend(rows[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)

    os.makedirs(out_dir, exist_ok=True)
    train_dir = os.path.join(out_dir, "train")
    val_dir = os.path.join(out_dir, "val")
    _write_sharded_parquet(train, train_dir, shard_size=4000)
    _write_sharded_parquet(val, val_dir, shard_size=4000)
    log.info("wrote %s (%d) and %s (%d)", train_dir, len(train), val_dir, len(val))
    return len(train), len(val)


def _write_sharded_parquet(rows: List[Dict[str, Any]], out_dir: str, shard_size: int = 4000) -> None:
    """Write rows into a directory of small parquet shards. Sharding avoids a
    known pyarrow issue with list<struct> columns above a certain size
    (``ArrowNotImplementedError: Nested data conversions not implemented for
    chunked array outputs``) while letting HF ``datasets.load_dataset`` pick up
    all shards via ``load_dataset(<dir>)``.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    os.makedirs(out_dir, exist_ok=True)
    # Clean prior shards to avoid stale rows mixing in.
    for old in glob.glob(os.path.join(out_dir, "*.parquet")):
        os.remove(old)
    if not rows:
        # Write an empty shard so the dir is a valid empty dataset.
        pd.DataFrame(columns=[]).to_parquet(os.path.join(out_dir, "shard_000.parquet"), index=False)
        return
    for i in range(0, len(rows), shard_size):
        chunk = rows[i : i + shard_size]
        table = pa.Table.from_pylist(chunk)
        path = os.path.join(out_dir, f"shard_{i // shard_size:03d}.parquet")
        pq.write_table(table, path, row_group_size=len(chunk), compression="snappy")


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


def validate(parquet_path: str, n_samples: int = 100, seed: int = 0) -> Dict[str, Any]:
    """Per-example format check: the LAST assistant turn must be well-formed
    for its role — intermediate (ssearch/python) if turn_idx < total_turns-1,
    else final (answer + boxed). Also count boxed↔target match on final-turn
    rows.

    Uses pyarrow batch iteration to avoid chunked-array conversion errors on
    large parquet files with nested list<struct> columns.
    """
    import pyarrow.parquet as pq  # local import

    # parquet_path may be a single file or a directory of shards.
    if os.path.isdir(parquet_path):
        shard_paths = sorted(glob.glob(os.path.join(parquet_path, "*.parquet")))
    else:
        shard_paths = [parquet_path]

    total_rows = sum(pq.ParquetFile(p).metadata.num_rows for p in shard_paths)
    rng = random.Random(seed)
    target_idxs = set(rng.sample(range(total_rows), min(n_samples, total_rows)))

    fmt_pass = 0
    final_rows = 0
    box_match = 0
    sampled = 0
    row_cursor = 0
    for shard in shard_paths:
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(batch_size=256, columns=["messages", "turn_idx", "total_turns", "target"]):
            msgs_col = batch.column("messages").to_pylist()
            turn_col = batch.column("turn_idx").to_pylist()
            total_col = batch.column("total_turns").to_pylist()
            tgt_col = batch.column("target").to_pylist()
            for j in range(len(msgs_col)):
                if row_cursor + j not in target_idxs:
                    continue
                messages = msgs_col[j]
                if not isinstance(messages, list):
                    continue
                is_final = int(turn_col[j]) == int(total_col[j]) - 1
                last_asst = next((m for m in reversed(messages) if m["role"] == "assistant"), None)
                if last_asst is None:
                    continue
                sampled += 1
                if _turn_is_well_formatted(last_asst["content"], is_final=is_final):
                    fmt_pass += 1
                if is_final:
                    final_rows += 1
                    m = re.search(r"\\boxed\{([^}]*)\}", last_asst["content"])
                    if m and m.group(1).strip().lower() == tgt_col[j].lower():
                        box_match += 1
            row_cursor += len(msgs_col)

    report = {
        "sampled": sampled,
        "format_pass": fmt_pass,
        "format_pass_rate": fmt_pass / max(1, sampled),
        "final_rows": final_rows,
        "boxed_match_on_final": box_match,
        "boxed_match_rate": box_match / max(1, final_rows),
    }
    log.info("validate report: %s", report)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_TRACES_GLOB = (
    "/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer-deepresearch/"
    "results/reviews/dr_areachair_qwen35_shard*_of_66_*.jsonl"
)
DEFAULT_SOURCE_JSON = (
    "/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer/data/"
    "iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_train/"
    "data.json"
)
DEFAULT_OUT_DIR = "/scratch/gpfs/ZHUANGL/sk7524/SkyRLMain/data/iclr_arxiv_sft"


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=False)

    bp = sub.add_parser("build", help="build train/val parquet")
    bp.add_argument("--traces_glob", default=DEFAULT_TRACES_GLOB)
    bp.add_argument("--source_json", default=DEFAULT_SOURCE_JSON)
    bp.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    bp.add_argument("--max_traces", type=int, default=None)
    bp.add_argument("--max_token_budget", type=int, default=40000)
    bp.add_argument("--val_frac", type=float, default=0.02)
    bp.add_argument("--seed", type=int, default=42)
    bp.add_argument(
        "--per_turn_split",
        action="store_true",
        help="Emit one SFT example per assistant turn (N examples per trace). "
        "Default is one example per trace — pair with trainer "
        "multi_turn_loss=true for gradient on every assistant turn.",
    )
    bp.add_argument(
        "--balance_per_bucket",
        type=int,
        default=0,
        help="If >0, cap the number of examples per (year, target) bucket to "
        "this value after building. Small buckets contribute everything; "
        "large buckets are subsampled uniformly. Use 0 (default) to keep all.",
    )
    bp.add_argument(
        "--balance_within_year",
        action="store_true",
        help="For each year, subsample the majority class to match the minority "
        "class (min(n_accept, n_reject) of each). No arbitrary cap — every "
        "example from the minority class is kept. Mutually exclusive with "
        "--balance_per_bucket.",
    )

    vp = sub.add_parser("validate", help="sanity-check a built parquet")
    vp.add_argument("parquet")
    vp.add_argument("--n_samples", type=int, default=100)
    vp.add_argument("--seed", type=int, default=0)

    # Default to build if no subcommand
    args = ap.parse_args()
    if args.cmd in (None, "build"):
        # When no subcommand is given, argparse leaves the build-specific attrs unset.
        # Re-parse with "build" prepended so the defaults kick in.
        if args.cmd is None:
            args = ap.parse_args(["build"] + sys.argv[1:])
        src = index_source(args.source_json)
        examples = build_examples(
            traces_glob=args.traces_glob,
            source_index=src,
            max_traces=args.max_traces,
            max_token_budget=args.max_token_budget,
            per_turn_split=args.per_turn_split,
        )
        if args.balance_per_bucket > 0 and args.balance_within_year:
            raise ValueError("--balance_per_bucket and --balance_within_year are mutually exclusive")
        if args.balance_per_bucket > 0:
            examples = _subsample_balanced(examples, args.balance_per_bucket, seed=args.seed)
        elif args.balance_within_year:
            examples = _balance_within_year(examples, seed=args.seed)
        split_and_write(examples, args.out_dir, val_frac=args.val_frac, seed=args.seed)
    elif args.cmd == "validate":
        validate(args.parquet, n_samples=args.n_samples, seed=args.seed)


if __name__ == "__main__":
    main()
