"""
Preprocess the SearchR1 dataset (NQ + HotpotQA) to parquet format for SkyRL evaluation.

Downloads from HuggingFace: PeterJinGo/nq_hotpotqa_train

Usage:
    python examples/searchr1/searchr1_dataset.py --output_dir data/searchr1
"""

import argparse
import os
from typing import Any

import datasets
import pandas as pd


# System prompt for Search-R1 tasks
SEARCH_SYSTEM_PROMPT = """You are a helpful assistant that can search Wikipedia for information.

You have access to a search tool that takes a query and returns relevant Wikipedia documents.

To search, use: <search>your query here</search>
To provide your final answer, use: <answer>your answer here</answer>

Think step by step:
1. Analyze what information you need
2. Search for relevant information
3. Analyze the search results
4. If needed, search again with refined queries
5. Provide your final answer

Keep your answer concise (1-5 words typically)."""


def process_answer(answer: Any) -> list[str]:
    """Convert answer to list of strings, handling various formats."""
    if isinstance(answer, str):
        return [answer]
    elif isinstance(answer, list):
        # Flatten and convert to strings
        result = []
        for a in answer:
            if isinstance(a, str):
                result.append(a)
            elif isinstance(a, (list, tuple)):
                result.extend(str(x) for x in a)
            else:
                result.append(str(a))
        return result if result else [""]
    elif hasattr(answer, "tolist"):  # numpy array
        return [str(x) for x in answer.tolist()]
    else:
        return [str(answer)]


def process_fn(example: dict, idx: int, data_source: str) -> dict:
    """Process a single example into SkyRL format."""
    question = example.get("question", example.get("query", ""))

    # Handle different answer field names
    if "golden_answers" in example:
        answers = process_answer(example["golden_answers"])
    elif "answer" in example:
        answers = process_answer(example["answer"])
    elif "answers" in example:
        answers = process_answer(example["answers"])
    else:
        answers = [""]

    # Build the prompt with system message
    prompt_content = f"{SEARCH_SYSTEM_PROMPT}\n\nQuestion: {question}"

    data = {
        "data_source": data_source,
        "prompt": [
            {
                "role": "user",
                "content": prompt_content,
            }
        ],
        "env_class": "searchr1embeddings",
        "reward_model": {
            "ground_truth": {"target": answers},
        },
        "extra_info": {
            "split": example.get("split", "unknown"),
            "index": idx,
            "question": question,
            "answers": answers,
            "data_source": example.get("data_source", data_source),
        },
    }
    return data


if __name__ == "__main__":
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/searchr1")
    parser.add_argument("--dataset_name", default="PeterJinGo/nq_hotpotqa_train")
    parser.add_argument("--parquet_dir", default="/scratch/gpfs/ZHUANGL/sk7524/hf/hub/datasets--PeterJinGo--nq_hotpotqa_train/snapshots/b7d80abfee334a7a91cb377544f09180d58b34f6",
                        help="Directory with pre-downloaded parquet files")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Max training samples (for testing)")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Max validation samples (for testing)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load from pre-downloaded parquet files (avoids HF datasets schema issues)
    print(f"Loading from parquet files in: {args.parquet_dir}")
    train_df = pd.read_parquet(os.path.join(args.parquet_dir, "train.parquet"))
    test_df = pd.read_parquet(os.path.join(args.parquet_dir, "test.parquet"))
    print(f"Loaded train: {len(train_df)}, test: {len(test_df)}")

    # Shuffle
    train_df = train_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # Limit samples if requested
    if args.max_train_samples:
        train_df = train_df.head(args.max_train_samples)
    if args.max_val_samples:
        test_df = test_df.head(args.max_val_samples)

    # Convert to list of dicts
    train_dataset = train_df.to_dict("records")
    test_dataset = test_df.to_dict("records")

    # Print sample
    print("\nSample from train dataset:")
    print(train_dataset[0])

    # Process training data
    print(f"\nProcessing {len(train_dataset)} training examples...")
    train_processed = [
        process_fn(example, idx, args.dataset_name)
        for idx, example in enumerate(train_dataset)
    ]
    train_ds = datasets.Dataset.from_list(train_processed)
    train_ds.to_parquet(os.path.join(args.output_dir, "train.parquet"))
    print(f"Saved train.parquet ({len(train_ds)} examples)")

    # Process validation/test data
    print(f"Processing {len(test_dataset)} validation examples...")
    val_processed = [
        process_fn(example, idx, args.dataset_name)
        for idx, example in enumerate(test_dataset)
    ]
    val_ds = datasets.Dataset.from_list(val_processed)
    val_ds.to_parquet(os.path.join(args.output_dir, "validation.parquet"))
    print(f"Saved validation.parquet ({len(val_ds)} examples)")

    print(f"\nDataset saved to {args.output_dir}")
    print(f"Sample processed entry:")
    print(train_processed[0])
