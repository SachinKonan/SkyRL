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
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/searchr1")
    parser.add_argument("--dataset_name", default="PeterJinGo/nq_hotpotqa_train")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Max training samples (for testing)")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Max validation samples (for testing)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading dataset: {args.dataset_name}")

    try:
        # Try loading with train/test splits
        train_dataset = datasets.load_dataset(args.dataset_name, split="train")
        test_dataset = datasets.load_dataset(args.dataset_name, split="test")
        print(f"Loaded train: {len(train_dataset)}, test: {len(test_dataset)}")
    except Exception as e:
        print(f"Could not load with splits, trying without: {e}")
        # Try loading without splits
        full_dataset = datasets.load_dataset(args.dataset_name)
        if "train" in full_dataset:
            train_dataset = full_dataset["train"]
            test_dataset = full_dataset.get("test", full_dataset.get("validation", None))
        else:
            # Just use all data as train
            train_dataset = full_dataset
            test_dataset = None

    # Shuffle and limit samples if requested
    train_dataset = train_dataset.shuffle(seed=args.seed)
    if args.max_train_samples:
        train_dataset = train_dataset.select(range(min(args.max_train_samples, len(train_dataset))))

    if test_dataset:
        test_dataset = test_dataset.shuffle(seed=args.seed)
        if args.max_val_samples:
            test_dataset = test_dataset.select(range(min(args.max_val_samples, len(test_dataset))))

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
    if test_dataset:
        print(f"Processing {len(test_dataset)} validation examples...")
        val_processed = [
            process_fn(example, idx, args.dataset_name)
            for idx, example in enumerate(test_dataset)
        ]
        val_ds = datasets.Dataset.from_list(val_processed)
        val_ds.to_parquet(os.path.join(args.output_dir, "validation.parquet"))
        print(f"Saved validation.parquet ({len(val_ds)} examples)")
    else:
        # Create a small validation set from training data
        print("No test split found, creating validation from train...")
        val_size = min(500, len(train_processed) // 10)
        val_processed = train_processed[:val_size]
        val_ds = datasets.Dataset.from_list(val_processed)
        val_ds.to_parquet(os.path.join(args.output_dir, "validation.parquet"))
        print(f"Saved validation.parquet ({len(val_ds)} examples)")

    print(f"\nDataset saved to {args.output_dir}")
    print(f"Sample processed entry:")
    print(train_processed[0])
