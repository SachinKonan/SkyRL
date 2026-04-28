"""Chunk a SkyRL eval parquet into N pieces for parallel sbatch execution.

Two subcommands:

  text  Filter to year ∈ args.years, stratified-shuffle by Accept/Reject,
        partition into N chunks. Optionally check no submission_id overlap
        with a held-back train set.

  vl    Match the chunk shape of an existing text-chunk root: pull rows from
        the source VL parquet whose submission_id is in text chunk k, write
        them as VL chunk k. Same paper-row mapping → text and VL eval results
        align by submission_id.

Each chunk is written to ``<out>/chunk_<k>/validation.parquet`` so existing
sbatch templates (``gen_smoke_3b.sbatch`` etc.) that read
``${DATA_DIR}/validation.parquet`` work without changes.
"""

import argparse
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq


def _read_table_streaming(path: str) -> pa.Table:
    """Read a parquet via small batches so nested cells stay within a single
    chunk; nested list<struct> tables otherwise hit ArrowNotImplementedError
    on read or offset-overflow on combine_chunks (e.g. base64-encoded image
    bytes inside the prompt column)."""
    pf = pq.ParquetFile(path)
    batches = list(pf.iter_batches(batch_size=64))
    return pa.Table.from_batches(batches)


def _has_nested_columns(path: str) -> bool:
    return "images" in pq.ParquetFile(path).schema_arrow.names


def _stream_extra_info(path: str) -> List[dict]:
    """Read just extra_info as a python list. Cheap on the VL parquet because
    extra_info is a flat struct of small primitives — no nested-array issues."""
    t = pq.read_table(path, columns=["extra_info"])
    return t.combine_chunks().column("extra_info").to_pylist()


def _row_meta(row: dict) -> Dict[str, object]:
    ei = row["extra_info"] or {}
    rs = row.get("reward_spec") or {}
    target = (rs.get("ground_truth") or {}).get("target") or [None]
    return {
        "submission_id": ei.get("submission_id"),
        "year": ei.get("year"),
        "decision": target[0] if target else None,
    }


def _stratified_shuffle_indices(metas: List[dict], seed: int = 42) -> List[int]:
    """Return indices in stratified-shuffled order: alternates Accept and Reject
    so each chunk gets a balanced slice."""
    rng = random.Random(seed)
    by_dec: Dict[str, List[int]] = defaultdict(list)
    for i, m in enumerate(metas):
        by_dec[m["decision"]].append(i)
    for v in by_dec.values():
        rng.shuffle(v)
    out: List[int] = []
    cursors = {k: 0 for k in by_dec}
    keys = list(by_dec.keys())
    while any(cursors[k] < len(by_dec[k]) for k in keys):
        for k in keys:
            if cursors[k] < len(by_dec[k]):
                out.append(by_dec[k][cursors[k]])
                cursors[k] += 1
    return out


def _partition(indices: List[int], n: int) -> List[List[int]]:
    """Split list into N near-equal contiguous chunks (preserves the strat order)."""
    size = len(indices)
    base, extra = divmod(size, n)
    chunks = []
    start = 0
    for k in range(n):
        end = start + base + (1 if k < extra else 0)
        chunks.append(indices[start:end])
        start = end
    return chunks


def _write_chunk(table: pa.Table, indices: List[int], chunk_dir: str) -> None:
    os.makedirs(chunk_dir, exist_ok=True)
    sub = table.take(indices)
    out_path = os.path.join(chunk_dir, "validation.parquet")
    pq.write_table(sub, out_path)


def cmd_text(args: argparse.Namespace) -> None:
    table = _read_table_streaming(args.src)
    metas = [_row_meta(r) for r in table.to_pylist()]
    print(f"loaded {len(metas)} rows from {args.src}")

    keep_idx = [
        i for i, m in enumerate(metas)
        if (not args.years) or m["year"] in set(args.years)
    ]
    print(f"after year filter {args.years or '<all>'}: {len(keep_idx)} rows")

    if args.exclude_submission_ids:
        # Read the train parquet and collect its submission_ids; any test row
        # whose submission_id is in that set gets dropped (data-leak guard).
        train_table = _read_table_streaming(args.exclude_submission_ids)
        train_ids = set()
        for r in train_table.to_pylist():
            sid = (r.get("extra_info") or {}).get("submission_id")
            if sid:
                train_ids.add(sid)
        before = len(keep_idx)
        keep_idx = [i for i in keep_idx if metas[i]["submission_id"] not in train_ids]
        print(
            f"  excluded {before - len(keep_idx)} rows present in {args.exclude_submission_ids}"
        )

    kept_metas = [metas[i] for i in keep_idx]
    decision_counts: Dict[str, int] = defaultdict(int)
    for m in kept_metas:
        decision_counts[m["decision"]] += 1
    print(f"decision distribution: {dict(decision_counts)}")

    strat_order = _stratified_shuffle_indices(kept_metas, seed=args.seed)
    abs_order = [keep_idx[j] for j in strat_order]
    chunks = _partition(abs_order, args.num_chunks)

    os.makedirs(args.out, exist_ok=True)
    for k, chunk_idx in enumerate(chunks):
        chunk_dir = os.path.join(args.out, f"chunk_{k}")
        _write_chunk(table, chunk_idx, chunk_dir)
        chunk_dec: Dict[str, int] = defaultdict(int)
        for i in chunk_idx:
            chunk_dec[metas[i]["decision"]] += 1
        print(f"  chunk_{k}: {len(chunk_idx)} rows  {dict(chunk_dec)}  -> {chunk_dir}")

    total = sum(len(c) for c in chunks)
    union_ids = {metas[i]["submission_id"] for c in chunks for i in c}
    print(f"\ntotal across chunks: {total}; unique submission_ids: {len(union_ids)}")
    assert total == len(union_ids), "duplicate submission_ids across chunks"


def cmd_vl(args: argparse.Namespace) -> None:
    """Match VL rows to text chunks by submission_id, write per-chunk parquets.

    Streams the VL parquet batch-by-batch (avoids the int32 offset overflow
    that ``combine_chunks`` would hit on base64 image cells), and uses one
    ``ParquetWriter`` per output chunk so a row from any input batch can
    land in any output chunk.
    """
    eis = _stream_extra_info(args.src)
    by_sid = {ei["submission_id"]: i for i, ei in enumerate(eis)}
    print(f"loaded {len(eis)} VL rows (extra_info) from {args.src}")

    os.makedirs(args.out, exist_ok=True)

    # Build per-chunk: target row-set (in source-row order)
    chunk_target_sids: Dict[int, List[str]] = {}
    chunk_missing: Dict[int, List[str]] = {}
    for k in range(args.num_chunks):
        text_chunk = os.path.join(args.match_submission_ids, f"chunk_{k}", "validation.parquet")
        if not os.path.exists(text_chunk):
            print(f"  chunk_{k}: SKIP (text chunk not found at {text_chunk})")
            continue
        text_eis = _stream_extra_info(text_chunk)
        sids = [ei["submission_id"] for ei in text_eis]
        chunk_target_sids[k] = sids
        chunk_missing[k] = [sid for sid in sids if sid not in by_sid]

    # Map source-row -> target-chunk (a row may belong to at most one chunk).
    row_to_chunk: Dict[int, int] = {}
    for k, sids in chunk_target_sids.items():
        for sid in sids:
            if sid in by_sid:
                row_to_chunk[by_sid[sid]] = k

    # Stream VL parquet, distribute rows to chunk writers.
    pf = pq.ParquetFile(args.src)
    writers: Dict[int, pq.ParquetWriter] = {}
    chunk_paths: Dict[int, str] = {}
    chunk_counts: Dict[int, int] = {k: 0 for k in chunk_target_sids}

    schema = pf.schema_arrow
    for k in chunk_target_sids:
        chunk_dir = os.path.join(args.out, f"chunk_{k}")
        os.makedirs(chunk_dir, exist_ok=True)
        path = os.path.join(chunk_dir, "validation.parquet")
        chunk_paths[k] = path
        writers[k] = pq.ParquetWriter(path, schema)

    global_row = 0
    for batch in pf.iter_batches(batch_size=64):
        # For each chunk, pick batch-local indices for rows that belong here.
        per_chunk_idx: Dict[int, List[int]] = defaultdict(list)
        for local_i in range(batch.num_rows):
            tgt = row_to_chunk.get(global_row + local_i)
            if tgt is not None:
                per_chunk_idx[tgt].append(local_i)
        for k, idx in per_chunk_idx.items():
            sub = batch.take(pa.array(idx, type=pa.int64()))
            writers[k].write_batch(sub)
            chunk_counts[k] += len(idx)
        global_row += batch.num_rows

    for w in writers.values():
        w.close()

    for k, path in chunk_paths.items():
        msg = f"  chunk_{k}: {chunk_counts[k]} rows -> {path}"
        if chunk_missing.get(k):
            miss = chunk_missing[k]
            msg += f"  ({len(miss)} text submission_ids missing in VL: {miss[:3]}...)"
        print(msg)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_text = sub.add_parser("text")
    p_text.add_argument("--src", required=True)
    p_text.add_argument("--out", required=True)
    p_text.add_argument("--years", type=int, nargs="*", default=None)
    p_text.add_argument("--exclude_submission_ids", default=None,
                        help="Path to a train parquet whose submission_ids to drop.")
    p_text.add_argument("--num_chunks", type=int, default=7)
    p_text.add_argument("--seed", type=int, default=42)
    p_text.set_defaults(func=cmd_text)

    p_vl = sub.add_parser("vl")
    p_vl.add_argument("--src", required=True)
    p_vl.add_argument("--out", required=True)
    p_vl.add_argument("--match_submission_ids", required=True,
                      help="Root of an existing text-chunk dir to mirror.")
    p_vl.add_argument("--num_chunks", type=int, default=7)
    p_vl.set_defaults(func=cmd_vl)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
