"""
Preprocess the AIME2025 dataset to parquet format for SkyRL evaluation.

Usage:
    python examples/aime2025/aime2025_dataset.py --output_dir data/aime2025
"""

import argparse
import json
import os
from pathlib import Path

import datasets


# Path to AIME2025 dataset in HF cache
AIME2025_CACHE_PATH = "/scratch/gpfs/ZHUANGL/sk7524/hf/hub/datasets--opencompass--AIME2025/snapshots/a6ad95f611d72cf628a80b58bd0432ef6638f958"


def load_jsonl(file_path: str) -> list[dict]:
    """Load a JSONL file and return a list of dictionaries."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/aime2025")
    parser.add_argument("--aime_cache_path", default=AIME2025_CACHE_PATH)
    args = parser.parse_args()

    args.output_dir = os.path.expanduser(args.output_dir)

    data_source = "opencompass/AIME2025"

    # Load both AIME2025 JSONL files
    aime_i_path = Path(args.aime_cache_path) / "aime2025-I.jsonl"
    aime_ii_path = Path(args.aime_cache_path) / "aime2025-II.jsonl"

    aime_i_data = load_jsonl(aime_i_path)
    aime_ii_data = load_jsonl(aime_ii_path)

    print(f"Loaded {len(aime_i_data)} problems from AIME2025-I")
    print(f"Loaded {len(aime_ii_data)} problems from AIME2025-II")

    # Combine datasets with source file info
    all_data = []
    for item in aime_i_data:
        item["source_file"] = "aime2025-I.jsonl"
        all_data.append(item)
    for item in aime_ii_data:
        item["source_file"] = "aime2025-II.jsonl"
        all_data.append(item)

    print(f"Total problems: {len(all_data)}")

    # Instruction for AIME problems - prompt for boxed format
    instruction = "Solve this problem step by step. Put your final answer in \\boxed{}."

    def process_fn(example: dict, idx: int) -> dict:
        question_raw = example["question"]
        answer = example["answer"]
        source_file = example["source_file"]

        question = question_raw + "\n\n" + instruction

        data = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "user",
                    "content": question,
                }
            ],
            "env_class": "aime",
            # AIME env expects reward_model.ground_truth (not reward_spec)
            "reward_model": {
                "ground_truth": str(answer),
            },
            "extra_info": {
                "split": "test",
                "index": idx,
                "question": question_raw,
                "answer": answer,
                "source_file": source_file,
            },
        }
        return data

    # Process all examples
    processed_data = [process_fn(example, idx) for idx, example in enumerate(all_data)]

    # Create HuggingFace dataset
    val_dataset = datasets.Dataset.from_list(processed_data)

    print(f"Processed dataset size: {len(val_dataset)}")
    print(f"Sample entry: {val_dataset[0]}")

    # Save to parquet
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    val_dataset.to_parquet(os.path.join(output_dir, "validation.parquet"))

    print(f"Saved validation.parquet to {output_dir}")
