"""
Branched SkyRL Gym Generator - implements progressive branched rollouts.

This generator starts with fewer root trajectories (src_trajectories) and spawns
branches from completed trajectories until reaching the target group_size.

Inspired by tinker-cookbook's do_branched_group_rollout pattern.
"""

import asyncio
import copy
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import skyrl_gym
from loguru import logger
from omegaconf import DictConfig
from tqdm.asyncio import tqdm

from skyrl_train.generators.base import (
    BranchedTrajectoryID,
    GeneratorInput,
    GeneratorInterface,
    GeneratorOutput,
    TrajectoryID,
)
from skyrl_train.generators.skyrl_gym_generator import AgentLoopOutput, SkyRLGymGenerator
from skyrl_train.generators.utils import get_rollout_metrics
from skyrl_train.inference_engines.base import ConversationType, InferenceEngineInput
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient


@dataclass
class BranchPoint:
    """Information about where to branch from a completed trajectory."""

    branch_turn: int  # Which turn (0-indexed) to branch from
    branch_token_idx: int  # Token index within response_ids to split at
    prefix_input_ids: List[int]  # prompt_ids + response_ids[:branch_token_idx]
    prefix_loss_mask: List[int]  # loss_mask[:branch_token_idx]
    prefix_logprobs: Optional[List[float]]  # logprobs[:branch_token_idx]
    initial_prompt: ConversationType  # Original prompt for creating new env
    original_prompt_length: int  # Length of original prompt_ids (for separating prompt from response)


@dataclass
class CompletedTrajectory:
    """A completed trajectory with its metadata."""

    output: AgentLoopOutput
    trajectory_id: BranchedTrajectoryID
    prompt: ConversationType
    env_class: str
    env_extras: Dict[str, Any]


class BranchedSkyRLGymGenerator(SkyRLGymGenerator):
    """
    Generator with branched group rollout support.

    Key features:
    - Progressive execution: Start src_trajectories, spawn branches as they complete
    - Token-level branching: Branch at random positions within assistant turns
    - First-completed strategy: Process trajectories as they finish
    - Automatic cancellation: Cancel excess tasks once group_size is reached

    Always enforces multi-turn mode for proper branching support.
    """

    def __init__(
        self,
        generator_cfg: DictConfig,
        skyrl_gym_cfg: DictConfig,
        inference_engine_client: InferenceEngineClient,
        tokenizer,
        model_name: str,
    ):
        # Force multi-turn mode for branching
        generator_cfg = copy.deepcopy(generator_cfg)
        generator_cfg.use_conversation_multi_turn = True

        super().__init__(
            generator_cfg=generator_cfg,
            skyrl_gym_cfg=skyrl_gym_cfg,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            model_name=model_name,
        )

        # Branching configuration
        branching_cfg = getattr(generator_cfg, "branching", DictConfig({}))
        self.branching_enabled = getattr(branching_cfg, "enabled", True)
        self.src_trajectories = getattr(branching_cfg, "src_trajectories", 2)
        self.num_branches = getattr(branching_cfg, "num_branches", 2)

        # Random number generator for reproducible branching
        self.rng = random.Random()

    def _find_turn_boundaries(self, loss_mask: List[int]) -> List[Tuple[int, int]]:
        """Find (start, end) indices for each assistant turn based on loss_mask.

        Loss mask values:
        - 1 = model-generated tokens (assistant response)
        - 0 = environment/system tokens (observations, prompts)

        Returns a list of (start_idx, end_idx) tuples for each assistant turn.
        """
        if not loss_mask:
            return []

        boundaries = []
        in_turn = False
        turn_start = 0

        for i, mask in enumerate(loss_mask):
            if mask == 1 and not in_turn:
                turn_start = i
                in_turn = True
            elif mask == 0 and in_turn:
                boundaries.append((turn_start, i))
                in_turn = False

        # Handle case where last turn extends to end of sequence
        if in_turn:
            boundaries.append((turn_start, len(loss_mask)))

        return boundaries

    def _sample_branch_point(
        self,
        output: AgentLoopOutput,
        prompt: ConversationType,
    ) -> Optional[BranchPoint]:
        """Sample a branch point from a completed trajectory.

        Uses output.steps to find turn boundaries and picks a random position
        within the first 50% of a randomly selected turn.

        Returns None if no valid branch point can be found.
        """
        # Find model steps (assistant turns)
        model_steps = [s for s in output.steps if s["type"] == "model"]

        if not model_steps:
            logger.warning("No model steps found in trajectory, cannot branch")
            return None

        # Pick a random turn
        turn_idx = self.rng.randint(0, len(model_steps) - 1)
        step = model_steps[turn_idx]

        turn_start = step["start_ix"]
        turn_end = step["end_ix"]
        turn_length = turn_end - turn_start

        if turn_length < 2:
            logger.warning(f"Turn {turn_idx} too short ({turn_length} tokens), cannot branch")
            return None

        # Pick token position < 50% of the turn's tokens
        max_token_in_turn = max(1, int(turn_length * 0.5))
        token_offset_in_turn = self.rng.randint(1, max_token_in_turn)

        branch_token_idx = turn_start + token_offset_in_turn

        # Build prefix
        prefix_input_ids = list(output.prompt_ids) + list(output.response_ids[:branch_token_idx])
        prefix_loss_mask = list(output.loss_mask[:branch_token_idx])
        prefix_logprobs = (
            list(output.rollout_logprobs[:branch_token_idx]) if output.rollout_logprobs else None
        )

        return BranchPoint(
            branch_turn=turn_idx,
            branch_token_idx=branch_token_idx,
            prefix_input_ids=prefix_input_ids,
            prefix_loss_mask=prefix_loss_mask,
            prefix_logprobs=prefix_logprobs,
            initial_prompt=prompt,
            original_prompt_length=len(output.prompt_ids),
        )

    async def agent_loop_from_prefix(
        self,
        prompt: ConversationType,
        env_class: str,
        env_extras: Dict[str, Any],
        max_tokens: int,
        max_input_length: int,
        prefix_input_ids: List[int],
        prefix_loss_mask: List[int],
        prefix_logprobs: Optional[List[float]],
        start_turn: int,
        original_prompt_length: int,
        sampling_params: Optional[Dict[str, Any]] = None,
        trajectory_id: Optional[TrajectoryID] = None,
    ) -> AgentLoopOutput:
        """
        Run agent loop from a prefix point (for branched rollouts).

        Key differences from agent_loop:
        1. input_ids starts as prefix_input_ids (includes prefix response)
        2. loss_mask starts as prefix_loss_mask
        3. rollout_logprobs starts as prefix_logprobs
        4. turn counter starts at start_turn

        The model generates a continuation from the prefix, and the environment
        processes the FULL response (prefix + continuation).
        """
        # Create environment
        env_extras = copy.deepcopy(env_extras)
        env_extras["max_turns"] = self.max_turns
        env_config = self.skyrl_gym_cfg.get(env_class, DictConfig({}))
        env = skyrl_gym.make(env_class, env_config=env_config, extras=env_extras)

        # Set turn counter for branching (env's chat_history is not used for reward)
        env.set_history([], turn=start_turn)

        session_id = (
            f"{trajectory_id.to_string()}" if trajectory_id is not None else uuid4().hex
        )

        # Initialize from prefix
        # prefix_input_ids = prompt_ids + response_ids[:branch_token_idx]
        # prefix_loss_mask = loss_mask[:branch_token_idx] (loss_mask is for response only)
        input_ids = list(prefix_input_ids)
        loss_mask = list(prefix_loss_mask)
        rollout_logprobs = list(prefix_logprobs) if prefix_logprobs else None

        # Per-step rewards and timing
        per_step_rewards: List[Tuple[float, Optional[int]]] = []
        steps: List[Dict[str, Any]] = []

        done = False
        turn = start_turn
        stop_reason = "stop"  # Default

        while not done and turn < self.max_turns:
            if len(input_ids) > max_input_length:
                stop_reason = "length"
                break

            # Track token index before generation (relative to response_ids)
            model_start_ix = len(input_ids) - original_prompt_length
            llm_start_time = time.time()

            # Generate continuation from prefix (token-in-token-out)
            engine_input = InferenceEngineInput(
                prompt_token_ids=[input_ids],
                session_ids=[session_id],
                sampling_params=sampling_params,
            )
            engine_output = await self.inference_engine_client.generate(engine_input)
            output = engine_output["responses"][0]
            output_ids = engine_output["response_ids"][0]
            stop_reason = engine_output["stop_reasons"][0]

            llm_elapsed = time.time() - llm_start_time

            # Handle EOS appending for stop strings
            current_sampling_params = (
                sampling_params if sampling_params is not None else self.generator_cfg.sampling_params
            )
            stop_strs = current_sampling_params.get("stop", None)
            added_eos = False
            if (
                stop_strs is not None
                and self.generator_cfg.append_eos_token_after_stop_str_in_multi_turn
            ):
                if output.endswith(tuple(stop_strs)) and output_ids[-1] != self.tokenizer.eos_token_id:
                    output_ids.append(self.tokenizer.eos_token_id)
                    added_eos = True

            # Environment step
            env_start_time = time.time()
            env_step_output = await self._run_in_executor_if_available(env.step, output)
            env_elapsed = time.time() - env_start_time

            new_obs = env_step_output["observations"]
            step_reward = env_step_output["reward"]
            done = env_step_output["done"]

            # Update input_ids
            input_ids += output_ids

            # Set loss_mask for generated tokens
            if added_eos:
                loss_mask += [1] * (len(output_ids) - 1) + [0]
            else:
                loss_mask += [1] * len(output_ids)

            # Update logprobs
            if rollout_logprobs is not None:
                gen_logprobs = engine_output.get("response_logprobs", [None])[0]
                if gen_logprobs:
                    rollout_logprobs += gen_logprobs
                else:
                    rollout_logprobs += [0.0] * len(output_ids)

            # Record reward at response end (relative to response_ids)
            response_end_idx = len(input_ids) - original_prompt_length - 1
            per_step_rewards.append((step_reward, response_end_idx))

            # Record step timing
            model_end_ix = model_start_ix + len(output_ids)
            steps.append({
                "start_ix": model_start_ix,
                "end_ix": model_end_ix,
                "type": "model",
                "time_elapsed_s": llm_elapsed,
            })

            # Handle observations
            if len(new_obs) > 0 and not done:
                observation_ids = self.tokenizer.apply_chat_template(
                    [*self.base_conversation, *new_obs],
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.generator_cfg.chat_template_kwargs,
                )[len(self.base_conversation_token_ids):]

                input_ids += observation_ids
                loss_mask += [0] * len(observation_ids)
                if rollout_logprobs is not None:
                    rollout_logprobs += [0.0] * len(observation_ids)

                # Record env step
                current_response_len = len(input_ids) - original_prompt_length
                steps.append({
                    "start_ix": model_end_ix,
                    "end_ix": current_response_len,
                    "type": "env",
                    "time_elapsed_s": env_elapsed,
                })

            turn += 1

        # Get environment metrics and close
        env_metrics = env.get_metrics()
        await self._run_in_executor_if_available(env.close)

        # Extract final prompt_ids and response_ids
        prompt_ids = input_ids[:original_prompt_length]
        response_ids = input_ids[original_prompt_length:]

        # Ensure loss_mask matches response_ids length
        assert len(loss_mask) == len(response_ids), (
            f"loss_mask length ({len(loss_mask)}) != response_ids length ({len(response_ids)})"
        )

        # Build reward output (token-level rewards)
        token_level_rewards: List[float] = [0.0] * len(response_ids)
        for step_reward, idx in per_step_rewards:
            if idx is not None and 0 <= idx < len(token_level_rewards):
                token_level_rewards[idx] += step_reward

        return AgentLoopOutput(
            response_ids=response_ids,
            reward=token_level_rewards,
            stop_reason=stop_reason,
            loss_mask=loss_mask,
            prompt_ids=prompt_ids,
            rollout_logprobs=rollout_logprobs,
            env_metrics=env_metrics,
            steps=steps,
        )

    async def _generate_branched_for_uid(
        self,
        uid: str,
        prompts: List[ConversationType],
        env_classes: List[str],
        env_extras_list: List[Dict[str, Any]],
        max_tokens: int,
        max_input_length: int,
        sampling_params: Optional[Dict[str, Any]],
        group_size: int,
    ) -> List[CompletedTrajectory]:
        """Generate branched trajectories for a single UID.

        Uses FIRST_COMPLETED strategy to process trajectories as they finish
        and spawn branches until reaching group_size.
        """
        completed: List[CompletedTrajectory] = []
        active_tasks: Dict[asyncio.Task, Tuple[ConversationType, str, Dict[str, Any], BranchedTrajectoryID]] = {}

        # Use the first prompt/env_class/env_extras as template
        prompt = prompts[0]
        env_class = env_classes[0]
        env_extras = env_extras_list[0]

        next_rep_id = 0

        # Launch root trajectories
        num_roots = min(self.src_trajectories, group_size)
        logger.info(f"[{uid}] Launching {num_roots} root trajectories (target: {group_size})")

        for i in range(num_roots):
            traj_id = BranchedTrajectoryID(
                instance_id=uid,
                repetition_id=next_rep_id,
                source_repetition_id=None,
                branch_turn=None,
                branch_token_idx=None,
            )
            next_rep_id += 1

            task = asyncio.create_task(
                self.agent_loop(
                    prompt=prompt,
                    env_class=env_class,
                    env_extras=env_extras,
                    max_tokens=max_tokens,
                    max_input_length=max_input_length,
                    sampling_params=sampling_params,
                    trajectory_id=traj_id,
                )
            )
            active_tasks[task] = (prompt, env_class, env_extras, traj_id)

        # Process trajectories as they complete
        while len(completed) < group_size and active_tasks:
            done_tasks, _ = await asyncio.wait(
                active_tasks.keys(),
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in done_tasks:
                prompt, env_class, env_extras, traj_id = active_tasks.pop(task)

                try:
                    output = await task
                except Exception as e:
                    logger.error(f"[{uid}] Trajectory {traj_id.to_string()} failed: {e}")
                    continue

                # Store completed trajectory
                completed.append(CompletedTrajectory(
                    output=output,
                    trajectory_id=traj_id,
                    prompt=prompt,
                    env_class=env_class,
                    env_extras=env_extras,
                ))

                logger.debug(
                    f"[{uid}] Completed trajectory {len(completed)}/{group_size} "
                    f"(rep_id={traj_id.repetition_id}, root={traj_id.is_root()})"
                )

                # Spawn branches if room
                remaining_slots = group_size - len(completed) - len(active_tasks)
                if remaining_slots > 0:
                    branches_to_spawn = min(self.num_branches, remaining_slots)

                    for _ in range(branches_to_spawn):
                        # Sample branch point
                        branch_point = self._sample_branch_point(output, prompt)
                        if branch_point is None:
                            continue

                        branch_traj_id = BranchedTrajectoryID(
                            instance_id=uid,
                            repetition_id=next_rep_id,
                            source_repetition_id=traj_id.repetition_id,
                            branch_turn=branch_point.branch_turn,
                            branch_token_idx=branch_point.branch_token_idx,
                        )
                        next_rep_id += 1

                        branch_task = asyncio.create_task(
                            self.agent_loop_from_prefix(
                                prompt=prompt,
                                env_class=env_class,
                                env_extras=env_extras,
                                max_tokens=max_tokens,
                                max_input_length=max_input_length,
                                prefix_input_ids=branch_point.prefix_input_ids,
                                prefix_loss_mask=branch_point.prefix_loss_mask,
                                prefix_logprobs=branch_point.prefix_logprobs,
                                start_turn=branch_point.branch_turn,
                                original_prompt_length=branch_point.original_prompt_length,
                                sampling_params=sampling_params,
                                trajectory_id=branch_traj_id,
                            )
                        )
                        active_tasks[branch_task] = (prompt, env_class, env_extras, branch_traj_id)

                        logger.debug(
                            f"[{uid}] Spawned branch from rep_id={traj_id.repetition_id} "
                            f"at turn={branch_point.branch_turn}, token={branch_point.branch_token_idx}"
                        )

        # Cancel excess tasks
        if active_tasks:
            logger.info(f"[{uid}] Cancelling {len(active_tasks)} excess tasks")
            for task in active_tasks:
                task.cancel()
            # Wait for cancellation to complete
            await asyncio.gather(*active_tasks.keys(), return_exceptions=True)

        return completed[:group_size]  # Ensure we don't exceed group_size

    async def generate(self, input_batch: GeneratorInput, disable_tqdm: bool = False) -> GeneratorOutput:
        """Generate trajectories using branched rollouts.

        Groups prompts by UID and runs branched generation for each group.
        """
        if not self.branching_enabled:
            # Fall back to parent implementation
            return await super().generate(input_batch, disable_tqdm)

        prompts = input_batch["prompts"]
        env_classes = input_batch["env_classes"]
        env_extras = input_batch["env_extras"]
        trajectory_ids = input_batch.get("trajectory_ids", None)
        sampling_params = input_batch.get("sampling_params", None)
        max_tokens = self.generator_cfg.sampling_params.max_generate_length
        max_input_length = self.generator_cfg.max_input_length

        # Group by UID
        uid_to_indices: Dict[str, List[int]] = {}
        for i, traj_id in enumerate(trajectory_ids or []):
            uid = traj_id.instance_id if traj_id else f"uid_{i}"
            if uid not in uid_to_indices:
                uid_to_indices[uid] = []
            uid_to_indices[uid].append(i)

        # Process each UID group
        all_completed: List[CompletedTrajectory] = []

        uid_tasks = []
        for uid, indices in uid_to_indices.items():
            group_size = len(indices)
            uid_prompts = [prompts[i] for i in indices]
            uid_env_classes = [env_classes[i] for i in indices]
            uid_env_extras = [env_extras[i] for i in indices]

            uid_tasks.append(
                self._generate_branched_for_uid(
                    uid=uid,
                    prompts=uid_prompts,
                    env_classes=uid_env_classes,
                    env_extras_list=uid_env_extras,
                    max_tokens=max_tokens,
                    max_input_length=max_input_length,
                    sampling_params=sampling_params,
                    group_size=group_size,
                )
            )

        # Run all UID groups in parallel
        uid_results = await tqdm.gather(
            *uid_tasks,
            desc="Generating Branched Trajectories",
            disable=disable_tqdm,
        )

        for completed_list in uid_results:
            all_completed.extend(completed_list)

        # Build output
        responses = [ct.output.response_ids for ct in all_completed]
        rewards = [ct.output.reward for ct in all_completed]
        stop_reasons = [ct.output.stop_reason for ct in all_completed]
        loss_masks = [ct.output.loss_mask for ct in all_completed]
        prompt_token_ids = [ct.output.prompt_ids for ct in all_completed]
        env_metrics = [ct.output.env_metrics for ct in all_completed]
        all_steps = [ct.output.steps for ct in all_completed]
        output_trajectory_ids = [ct.trajectory_id for ct in all_completed]

        # Get logprobs if available
        get_logprobs = (
            sampling_params.get("logprobs", None) is not None
            if sampling_params
            else self.generator_cfg.sampling_params.logprobs is not None
        )
        if get_logprobs:
            rollout_logprobs = [ct.output.rollout_logprobs for ct in all_completed]
        else:
            rollout_logprobs = None

        # Compute rollout metrics
        rollout_metrics = get_rollout_metrics(
            responses, rewards, env_metrics, env_classes, all_steps
        )

        # Add branching-specific metrics
        num_roots = sum(1 for ct in all_completed if ct.trajectory_id.is_root())
        num_branches = len(all_completed) - num_roots
        rollout_metrics["branching/num_roots"] = num_roots
        rollout_metrics["branching/num_branches"] = num_branches
        rollout_metrics["branching/branch_ratio"] = num_branches / max(num_roots, 1)

        generator_output: GeneratorOutput = {
            "prompt_token_ids": prompt_token_ids,
            "response_ids": responses,
            "rewards": rewards,
            "loss_masks": loss_masks,
            "stop_reasons": stop_reasons,
            "rollout_metrics": rollout_metrics,
            "rollout_logprobs": rollout_logprobs,
            "trajectory_ids": output_trajectory_ids,
            "steps": all_steps,
        }

        return generator_output
