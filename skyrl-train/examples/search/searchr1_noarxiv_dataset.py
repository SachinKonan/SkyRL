# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Dataset generator for ICLR paper acceptance prediction without arXiv search.

Converts ShareGPT-format ICLR data to training parquet for the search_arxiv environment.

Input format (ShareGPT JSON):
{
  "conversations": [
    {"from": "system", "value": "You are an expert academic reviewer..."},
    {"from": "human", "value": "I am giving you a paper...# Title\n# Abstract\n..."},
    {"from": "gpt", "value": "Outcome: \\boxed{Accept}"}
  ],
  "_metadata": {"submission_id": "...", "answer": "Accept"}
}

Output format (Parquet):
{
    "data_source": "searchR1_noarxiv_iclr",
    "prompt": [{"role": "system", "content": ...}, {"role": "user", "content": ...}],
    "env_class": "search_arxiv",
    "reward_model": {"ground_truth": {"target": ["Accept"]}},
    "extra_info": {...}
}
"""

import argparse
import json
import logging
import os
from typing import List, Dict, Any

import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_SYSTEM_CONTENT = """You are an expert academic reviewer for the ICLR conference, tasked with evaluating research papers. Your task is to predict whether a paper will be accepted or rejected.
 - Note: ICLR generally has a ~30% acceptance rate

Think CONCISELY about things like:
1. The content of the paper
2. Whether the investigation is executed well, and the paper is high quality.
3. Whether the paper is novel w.r.t to others, in terms of things such as (but not limited to) ideas, execution, or results.

When you are ready to provide your final answer, use: <answer> \\boxed{Accept|Reject} </answer>
- Do NOT restate things you have already said.
"""


def load_sharegpt_json(file_path: str) -> List[Dict[str, Any]]:
    """Load ShareGPT format JSON file."""
    logger.info(f"Loading ShareGPT JSON from {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} examples")
    return data


def extract_question_from_conversations(conversations: List[Dict[str, str]]) -> str:
    """Extract the human question/paper content from conversations."""
    for conv in conversations:
        if conv.get("from") == "human":
            return conv.get("value", "")
    return ""


def extract_answer_from_metadata(item: Dict[str, Any]) -> str:
    """Extract ground truth answer from _metadata."""
    metadata = item.get("_metadata", {})
    answer = metadata.get("answer", "")
    return answer


def process_single_row(
    item: Dict[str, Any],
    row_index: int,
    current_split_name: str,
    system_content: str,
    user_content_prefix: str
) -> Dict[str, Any]:
    """
    Process a single ShareGPT item into the training format.

    Args:
        item: ShareGPT format item with conversations and _metadata
        row_index: Index of the row
        current_split_name: Name of the current split (train/validation)
        system_content: System prompt content
        user_content_prefix: Prefix for user content with instructions

    Returns:
        Processed row data in the required format
    """
    conversations = item.get("conversations", [])
    question = extract_question_from_conversations(conversations)
    ground_truth_answer = extract_answer_from_metadata(item)
    submission_id = item.get("_metadata", {}).get("submission_id", f"unknown_{row_index}")

    # Build prompt structure
    user_content = question.replace("I am giving you a paper. I want to predict its acceptance outcome at ICLR.\n - Your answer will either be: \\boxed{Accept} or \\boxed{Reject}\n - Note: ICLR generally has a ~30% acceptance rate\n\n", "")
    prompt = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]

    # Build ground truth for reward calculation
    # Normalize answer (Accept/Reject)
    if ground_truth_answer.lower().strip() in ["accept", "accepted"]:
        target = ["Accept"]
    elif ground_truth_answer.lower().strip() in ["reject", "rejected"]:
        target = ["Reject"]
    else:
        target = [ground_truth_answer]  # Keep original if not standard

    reward_model = {
        "ground_truth": {
            "target": target
        }
    }

    # Build data source tag
    data_source_tagged = "searchR1_noarxiv_iclr"

    # Build extra info
    year = item.get("_metadata", {}).get("year", None)
    extra_info = {
        "index": row_index,
        "split": current_split_name,
        "submission_id": submission_id,
        "ground_truth_answer": ground_truth_answer,
        "year": year,
    }

    return {
        "data_source": data_source_tagged,
        "prompt": prompt,
        "env_class": "search_arxiv",
        "reward_model": reward_model,
        "extra_info": extra_info,
    }


def process_dataset(
    data: List[Dict[str, Any]],
    split_name: str,
    system_content: str,
    user_content_prefix: str
) -> pd.DataFrame:
    """Process all items in the dataset."""
    processed_rows = []
    for idx, item in enumerate(data):
        try:
            row = process_single_row(
                item=item,
                row_index=idx,
                current_split_name=split_name,
                system_content=system_content,
                user_content_prefix=user_content_prefix
            )
            processed_rows.append(row)
        except Exception as e:
            logger.warning(f"Error processing row {idx}: {e}")
            continue

    return pd.DataFrame(processed_rows)


def main():
    parser = argparse.ArgumentParser(
        description="Convert ShareGPT-format ICLR data to training parquet for pure thinking, no access to search environment."
    )
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="Path to input ShareGPT JSON file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/scratch/gpfs/ZHUANGL/sk7524/SkyRL/skyrl-train/data/noarxiv_iclr",
        help="Output directory for parquet files."
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Ratio of data to use for training (rest for validation)."
    )
    parser.add_argument(
        "--system_content",
        type=str,
        default=None,
        help="Custom system content. Defaults to academic reviewer prompt."
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle data before splitting."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling."
    )

    args = parser.parse_args()

    # Set content
    system_content = args.system_content if args.system_content else DEFAULT_SYSTEM_CONTENT
    user_content_prefix = ""

    # Load data
    data = load_sharegpt_json(args.input_json)

    # Shuffle if requested
    if args.shuffle:
        import random
        random.seed(args.seed)
        random.shuffle(data)
        logger.info(f"Shuffled data with seed {args.seed}")

    # Split data
    split_idx = int(len(data) * args.train_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    logger.info(f"Split: {len(train_data)} train, {len(val_data)} validation")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process and save train data
    if train_data:
        train_df = process_dataset(train_data, "train", system_content, user_content_prefix)
        train_path = os.path.join(args.output_dir, "train.parquet")
        train_df.to_parquet(train_path, index=False)
        logger.info(f"Saved {len(train_df)} training examples to {train_path}")

    # Process and save validation data
    if val_data:
        val_df = process_dataset(val_data, "validation", system_content, user_content_prefix)
        val_path = os.path.join(args.output_dir, "validation.parquet")
        val_df.to_parquet(val_path, index=False)
        logger.info(f"Saved {len(val_df)} validation examples to {val_path}")

    logger.info(f"Dataset generation complete. Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
