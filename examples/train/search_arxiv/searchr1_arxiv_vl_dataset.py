"""
Build the VL (qwen2_vl / qwen3_vl template) parquet for `search_arxiv` training
from the LLaMA-Factory ICLR vision JSONs.

Differences from the text loader:
  - Rows carry an `images` field (list of absolute image paths, one per <image>
    placeholder in the user message).
  - The prompt is a list of chat messages where the user turn is a list of
    content parts mixing {"type":"image","image":<path>} and {"type":"text", ...}.

The VLM generator (skyrl.train.generators.skyrl_vlm_generator) and the shared
PromptDataset loader just carry the `prompt` column through as-is; the tokenizer's
chat template is responsible for handling the image parts, so we stick to the
content-parts convention already used by Qwen2.5-VL / Qwen3-VL chat templates.
"""

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from searchr1_arxiv_dataset import (  # noqa: E402
    ICLR_UPPER_BOUND,
    SYSTEM_CONTENT,
    extract_title,
    normalize_answer,
    parse_authors,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def split_on_image_tokens(text: str) -> List[str]:
    """Split `text` on '<image>', returning the interleaved non-image chunks."""
    return text.split("<image>")


def _image_part(path: str) -> Dict[str, Any]:
    """vLLM's chat endpoint accepts `image_url` with a file:// URI for local images."""
    url = path if "://" in path else f"file://{path}"
    return {"type": "image_url", "image_url": {"url": url}}


def build_user_parts(text: str, images_abs: List[str]) -> List[Dict[str, Any]]:
    """
    Given human text with N <image> placeholders and N image paths, build the
    Qwen-VL content-parts list using the OpenAI-compatible schema vLLM expects:
        [{"type":"image_url","image_url":{"url":"file://..."}},
         {"type":"text","text":chunk}, ...]
    """
    chunks = split_on_image_tokens(text)
    n_placeholders = len(chunks) - 1
    if n_placeholders != len(images_abs):
        parts: List[Dict[str, Any]] = [_image_part(p) for p in images_abs]
        if text:
            parts.append({"type": "text", "text": text})
        return parts

    parts = []
    if chunks[0]:
        parts.append({"type": "text", "text": chunks[0]})
    for img_path, next_chunk in zip(images_abs, chunks[1:]):
        parts.append(_image_part(img_path))
        if next_chunk:
            parts.append({"type": "text", "text": next_chunk})
    return parts


def process_row(
    row: Dict[str, Any],
    split: str,
    index: int,
    image_root: str,
) -> Optional[Dict[str, Any]]:
    conv = row.get("conversations") or []
    meta = row.get("_metadata") or {}
    images = row.get("images") or []

    target = normalize_answer(meta.get("answer"))
    if target is None:
        return None

    human_turn = next((m["value"] for m in conv if m.get("from") == "human"), None)
    if not human_turn:
        return None

    images_abs = [os.path.join(image_root, p) if not os.path.isabs(p) else p for p in images]
    title = extract_title(human_turn)
    year = meta.get("year")
    upper = ICLR_UPPER_BOUND.get(int(year)) if isinstance(year, (int, float, str)) and str(year).isdigit() else None

    # Arrow needs a single schema per nested column, so use content-parts for both turns.
    prompt = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_CONTENT}]},
        {"role": "user", "content": build_user_parts(human_turn, images_abs)},
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
        "num_images": len(images_abs),
    }

    return {
        "data_source": "iclr_arxiv_vl",
        "prompt": prompt,
        "ability": "search_arxiv",
        "env_class": "search_arxiv",
        "reward_spec": reward_spec,
        "extra_info": extra_info,
        "metadata": meta,
        "images": images_abs,  # mirror LLaMA-Factory convention; generator may use it
    }


DEFAULT_INPUT_BASE = (
    "/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer/data/"
    "iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480"
)
DEFAULT_IMAGE_ROOT = "/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer"
DEFAULT_OUTPUT_DIR = "/scratch/gpfs/ZHUANGL/sk7524/SkyRLMain/data/iclr_arxiv_vl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_base", default=DEFAULT_INPUT_BASE)
    ap.add_argument("--image_root", default=DEFAULT_IMAGE_ROOT, help="Root to resolve relative image paths.")
    ap.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    ap.add_argument("--max_rows", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for split in args.splits:
        in_path = f"{args.input_base}_{split}/data.json"
        out_path = os.path.join(args.output_dir, f"{split}.parquet")
        if not os.path.exists(in_path):
            logger.warning(f"skipping {split}: {in_path} not found")
            continue
        logger.info(f"Loading {in_path}")
        with open(in_path, "r") as f:
            data = json.load(f)
        if args.max_rows is not None:
            data = data[: args.max_rows]

        rows = []
        dropped = 0
        for i, r in enumerate(data):
            out = process_row(r, split, i, args.image_root)
            if out is None:
                dropped += 1
                continue
            rows.append(out)
        logger.info(f"  kept {len(rows)}, dropped {dropped}")
        pd.DataFrame(rows).to_parquet(out_path, index=False)
        logger.info(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
