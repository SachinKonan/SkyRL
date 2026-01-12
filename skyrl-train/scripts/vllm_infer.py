#!/usr/bin/env python3
"""
vLLM inference using SkyRL infrastructure.

Follows existing conventions: @hydra.main, @ray.remote, BasePPOExp.
Converts ShareGPT JSON to temp parquet, uses PromptDataset and SkyRLGymGenerator.

Usage:
    python scripts/vllm_infer.py \
        trainer.policy.model.path=/path/to/model \
        generator.backend=vllm \
        generator.eval_n_samples_per_prompt=8 \
        generator.max_turns=3 \
        +infer_input_file=/path/to/data.json \
        +infer_output_file=/path/to/output.jsonl \
        +infer_use_arxiv=true
"""

import asyncio
import json
import os
import tempfile
import shutil
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any, List, Dict, Optional

import hydra
import ray
import pandas as pd
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from skyrl_train.entrypoints.main_base import (
    BasePPOExp,
    config_dir,
    create_ray_wrapped_inference_engines_from_config,
    create_remote_inference_engines_from_config,
)
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.utils.utils import validate_generator_cfg, initialize_ray
from skyrl_train.utils.trainer_utils import build_dataloader
from skyrl_train.dataset import PromptDataset
from skyrl_train.generators.utils import (
    prepare_generator_input,
    parse_prediction_from_response,
)
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend


# System prompts (same as training datasets)
ARXIV_SYSTEM_PROMPT = """You are an expert academic reviewer for the ICLR conference, tasked with evaluating research papers. Your task is to predict whether a paper will be accepted or rejected.
 - Note: ICLR generally has a ~30% acceptance rate

You have access to a arxiv search tool that allows you to semantically search for other papers.

To semantic search, use: <ssearch> your search query here</ssearch>
- this works best if you are descriptive and dont use abbreviations, for example: <ssearch> the universality of graph neural networks and depth versus width </ssearch>

To provide your final answer, use: <answer> \\boxed{Accept|Reject} </answer>

Think CONCISELY about things like:
1. The content of the paper
2. Whether the investigation is executed well, and the paper is high quality.
3. Whether the paper is novel w.r.t to others, in terms of things such as (but not limited to) ideas, execution, or results.

Do NOT restate things you have already said.

Every response must end with <ssearch> [query] </ssearch> or <answer> \\boxed{Accept|Reject} </answer>
"""

NOARXIV_SYSTEM_PROMPT = """You are an expert academic reviewer for the ICLR conference, tasked with evaluating research papers. Your task is to predict whether a paper will be accepted or rejected.
 - Note: ICLR generally has a ~30% acceptance rate

Think CONCISELY about things like:
1. The content of the paper
2. Whether the investigation is executed well, and the paper is high quality.
3. Whether the paper is novel w.r.t to others, in terms of things such as (but not limited to) ideas, execution, or results.

When you are ready to provide your final answer, use: <answer> \\boxed{Accept|Reject} </answer>
- Do NOT restate things you have already said.
"""


def extract_user_content(conversations: List[Dict]) -> str:
    """Extract human content from ShareGPT conversations."""
    for conv in conversations:
        if conv.get("from") == "human":
            return conv.get("value", "")
    return ""


def convert_sharegpt_to_parquet(
    input_file: str,
    output_dir: str,
    system_prompt: str,
    env_class: str,
    limit: Optional[int] = None,
) -> str:
    """Convert ShareGPT JSON to skyrl parquet format."""
    logger.info(f"Loading ShareGPT JSON from {input_file}")
    with open(input_file) as f:
        data = json.load(f)

    if limit:
        data = data[:limit]
        logger.info(f"Limited to {limit} examples")

    rows = []
    for idx, item in enumerate(data):
        paper_content = extract_user_content(item["conversations"])
        metadata = item.get("_metadata", {})
        ground_truth_answer = metadata.get("answer", "")
        year = metadata.get("year")

        # Normalize ground truth to Accept/Reject
        if ground_truth_answer.lower().strip() in ["accept", "accepted"]:
            target = ["Accept"]
        elif ground_truth_answer.lower().strip() in ["reject", "rejected"]:
            target = ["Reject"]
        else:
            target = [ground_truth_answer] if ground_truth_answer else ["Unknown"]

        # Format reward_model as expected by search_arxiv env
        reward_model = {
            "ground_truth": {
                "target": target
            }
        }

        # Format extra_info with year for temporal filtering
        extra_info = {
            "index": idx,
            "submission_id": metadata.get("submission_id", str(idx)),
            "ground_truth_answer": ground_truth_answer,
            "year": year,
        }

        rows.append({
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": paper_content},
            ],
            "env_class": env_class,
            "uid": metadata.get("submission_id", str(idx)),
            "reward_model": reward_model,
            "extra_info": extra_info,
            "data_source": "sharegpt_eval",
            # Store original for output formatting
            "_original_conversations": json.dumps(item["conversations"]),
            "_original_metadata": json.dumps(metadata),
        })

    df = pd.DataFrame(rows)
    output_path = os.path.join(output_dir, "eval_data.parquet")
    df.to_parquet(output_path)
    logger.info(f"Converted {len(rows)} items to {output_path}")
    return output_path


def compute_majority_vote(responses: List[str]) -> Optional[str]:
    """Compute majority vote from responses (reuses existing parse function)."""
    predictions = [parse_prediction_from_response(r) for r in responses]
    predictions = [p for p in predictions if p]
    if not predictions:
        return None
    return Counter(predictions).most_common(1)[0][0]


def format_prompt_for_output(conversations: List[Dict]) -> str:
    """Format conversations for output."""
    parts = []
    for conv in conversations:
        if conv.get("from") == "gpt":
            continue
        parts.append(f"[{conv.get('from', 'unknown')}]: {conv.get('value', '')}")
    return "\n\n".join(parts)


class ShareGPTInferEntrypoint(BasePPOExp):
    """Entrypoint for ShareGPT inference, following EvalOnlyEntrypoint pattern."""

    def __init__(self, cfg: DictConfig, temp_dir: str):
        self.temp_dir = temp_dir
        super().__init__(cfg)

    def get_train_dataset(self):
        """No training dataset needed."""
        return None

    def get_eval_dataset(self):
        """Load the converted parquet as eval dataset."""
        return PromptDataset(
            datasets=self.cfg.data.val_data,
            tokenizer=self.tokenizer,
            max_prompt_length=self.cfg.trainer.max_prompt_length,
            num_workers=8,
        )

    async def run(self) -> List[Dict[str, Any]]:
        """Run inference and return results in LlamaFactory format."""
        assert self.eval_dataset is not None

        tokenizer = self.tokenizer

        # Create inference engines (same as EvalOnlyEntrypoint)
        if self.cfg.generator.run_engines_locally:
            inference_engines = create_ray_wrapped_inference_engines_from_config(
                self.cfg, self.colocate_pg, tokenizer
            )
        else:
            inference_engines = create_remote_inference_engines_from_config(self.cfg, tokenizer)

        inference_engine_client = InferenceEngineClient(inference_engines, tokenizer, self.cfg)
        await inference_engine_client.wake_up()
        generator = self.get_generator(self.cfg, tokenizer, inference_engine_client)

        # Build dataloader
        eval_dataloader = build_dataloader(self.cfg, self.eval_dataset, is_train=False)

        # Collect all generator outputs for post-processing
        all_outputs = []
        all_extras = []

        sampling_params = self.cfg.generator.eval_sampling_params

        for batch_idx, prompts in enumerate(tqdm(eval_dataloader, desc="Inference")):
            generator_input, uids = prepare_generator_input(
                prompts,
                self.cfg.generator.eval_n_samples_per_prompt,
                get_sampling_params_for_backend(self.cfg.generator.backend, sampling_params),
                self.cfg.environment.env_class,
                "eval",
                None,
            )

            generator_output = await generator.generate(generator_input)

            # Store outputs with extras for post-processing
            for i, uid in enumerate(uids):
                response_ids = generator_output["response_ids"][i]
                response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
                all_outputs.append({
                    "uid": uid,
                    "response": response_text,
                })

            # Store extras from prompts (only once per unique uid)
            for prompt in prompts:
                all_extras.append({
                    "uid": prompt["uid"],
                    "env_extras": prompt["env_extras"],
                })

        # Post-process: group by uid and compute majority vote
        return self._format_results(all_outputs, all_extras)

    def _format_results(self, all_outputs, all_extras) -> List[Dict]:
        """Format results in LlamaFactory format with majority voting."""
        # Group responses by uid
        uid_to_responses = defaultdict(list)
        for output in all_outputs:
            uid_to_responses[output["uid"]].append(output["response"])

        # Get extras by uid (first occurrence)
        uid_to_extras = {}
        for extra in all_extras:
            if extra["uid"] not in uid_to_extras:
                uid_to_extras[extra["uid"]] = extra["env_extras"]

        results = []
        for uid, responses in uid_to_responses.items():
            extras = uid_to_extras.get(uid, {})

            # Parse original data
            original_convs = json.loads(extras.get("_original_conversations", "[]"))
            original_meta = json.loads(extras.get("_original_metadata", "{}"))

            # Get ground truth from extra_info (stored there during conversion)
            extra_info = extras.get("extra_info", {})
            ground_truth = extra_info.get("ground_truth_answer", "")

            # Compute majority vote
            majority = compute_majority_vote(responses)

            results.append({
                "prompt": format_prompt_for_output(original_convs),
                "generations": responses,
                "predict": f"Outcome: \\boxed{{{majority}}}" if majority else responses[0],
                "label": f"Outcome: \\boxed{{{ground_truth}}}",
                "_metadata": original_meta,
            })

        return results


@ray.remote(num_cpus=1)
def infer_entrypoint(cfg: DictConfig, temp_dir: str) -> List[Dict]:
    """Ray remote entrypoint."""
    exp = ShareGPTInferEntrypoint(cfg, temp_dir)
    return asyncio.run(exp.run())


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main entry point with Hydra config."""
    # Get custom args from config overrides
    input_file = cfg.get("infer_input_file")
    output_file = cfg.get("infer_output_file")
    use_arxiv = cfg.get("infer_use_arxiv", True)
    limit = cfg.get("infer_limit", None)

    assert input_file, "Must provide +infer_input_file=/path/to/data.json"
    assert output_file, "Must provide +infer_output_file=/path/to/output.jsonl"

    # Create output directory if needed
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Create temp directory for converted data
    temp_dir = tempfile.mkdtemp(prefix="vllm_infer_")
    logger.info(f"Using temp directory: {temp_dir}")

    try:
        # Convert ShareGPT to parquet
        system_prompt = ARXIV_SYSTEM_PROMPT if use_arxiv else NOARXIV_SYSTEM_PROMPT
        env_class = "search_arxiv"

        parquet_path = convert_sharegpt_to_parquet(
            input_file, temp_dir, system_prompt, env_class, limit
        )

        # Update config to use converted data
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        cfg_dict["data"]["val_data"] = [parquet_path]
        cfg = OmegaConf.create(cfg_dict)

        # Validate and run
        validate_generator_cfg(cfg)
        initialize_ray(cfg)

        results = ray.get(infer_entrypoint.remote(cfg, temp_dir))

        # Write results
        logger.info(f"Writing {len(results)} results to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

        logger.info("Done!")

    finally:
        # Cleanup temp directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temp directory: {temp_dir}")


if __name__ == "__main__":
    main()
