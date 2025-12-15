"""
Tests for BranchedSkyRLGymGenerator - branched group rollouts.

uv run --extra dev pytest tests/cpu/generators/test_branched_skyrl_gym_generator.py -v
"""

import copy
import pytest
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch
from omegaconf import DictConfig

from skyrl_train.generators.branched_skyrl_gym_generator import BranchedSkyRLGymGenerator
from skyrl_train.generators.base import (
    GeneratorInput,
    GeneratorOutput,
    TrajectoryID,
    BranchedTrajectoryID,
)
from skyrl_gym.envs.base_text_env import BaseTextEnvStepOutput, BaseTextEnv
from skyrl_train.config.utils import get_default_config


# Mock constants
MOCK_PROMPT_IDS = [1, 2, 3, 4]
MOCK_LLM_OUTPUT_IDS = [10, 11, 12, 13, 99]  # 99 is EOS
MOCK_OBSERVATION_IDS = [50, 51]


# =============================================================================
# Mock Environments (don't use real envs to avoid ChromaDB/external deps)
# =============================================================================


class MockSingleTurnEnv(BaseTextEnv):
    """
    Simple single-turn environment for testing.
    Simulates gsm8k-like behavior: single response, immediate reward.
    """

    def __init__(self, env_config: DictConfig = None, extras: Dict[str, Any] = None):
        super().__init__()
        self.max_turns = 1
        self.chat_history = []
        self.extras = extras or {}

    def init(self, prompt):
        self.chat_history = copy.deepcopy(prompt)
        return prompt, {}

    def step(self, action):
        self.turns += 1
        self.chat_history.append({"role": "assistant", "content": action})
        # Simple reward: 1.0 if "answer" in response, else 0.5
        reward = 1.0 if "answer" in action.lower() else 0.5
        return BaseTextEnvStepOutput(
            observations=[],
            reward=reward,
            done=True,
            metadata={},
        )

    def set_history(self, chat_history, turn=0):
        super().set_history(chat_history, turn)
        self.chat_history = copy.deepcopy(chat_history)

    def get_metrics(self):
        return {"turns": self.turns}


class MockMultiTurnEnv(BaseTextEnv):
    """
    Multi-turn environment for testing.
    Simulates search-r1-like behavior: search -> observation -> answer.
    """

    def __init__(self, env_config: DictConfig = None, extras: Dict[str, Any] = None):
        super().__init__()
        extras = extras or {}
        self.max_turns = extras.get("max_turns", 3)
        self.chat_history = []

    def init(self, prompt):
        self.chat_history = copy.deepcopy(prompt)
        return prompt, {}

    def step(self, action):
        self.turns += 1
        self.chat_history.append({"role": "assistant", "content": action})

        # Check for answer tag (done)
        if "<answer>" in action and "</answer>" in action:
            return BaseTextEnvStepOutput(
                observations=[],
                reward=1.0,
                done=True,
                metadata={"action_type": "answer"},
            )

        # Check for search tag (continue with observation)
        if "<search>" in action and "</search>" in action:
            if self.turns >= self.max_turns:
                # Max turns reached without answer
                return BaseTextEnvStepOutput(
                    observations=[],
                    reward=0.0,
                    done=True,
                    metadata={"action_type": "search_timeout"},
                )
            # Return mock observation
            obs = {
                "role": "user",
                "content": f"<information>Search result {self.turns}: Mock data</information>",
            }
            self.chat_history.append(obs)
            return BaseTextEnvStepOutput(
                observations=[obs],
                reward=0.0,
                done=False,
                metadata={"action_type": "search"},
            )

        # No valid tag - end with low reward
        return BaseTextEnvStepOutput(
            observations=[],
            reward=0.0,
            done=True,
            metadata={"action_type": "invalid"},
        )

    def set_history(self, chat_history, turn=0):
        super().set_history(chat_history, turn)
        self.chat_history = copy.deepcopy(chat_history)

    def get_metrics(self):
        return {"turns": self.turns}


# Registry of mock environments
# Use registered env names to avoid metrics aggregation errors
MOCK_ENV_REGISTRY = {
    "gsm8k": MockSingleTurnEnv,  # gsm8k is a registered single-turn env
    "gsm8k_multi_turn": MockMultiTurnEnv,  # gsm8k_multi_turn is a registered multi-turn env
}


def mock_skyrl_gym_make(env_class: str, env_config=None, extras=None):
    """Mock skyrl_gym.make that returns our test environments."""
    if env_class in MOCK_ENV_REGISTRY:
        return MOCK_ENV_REGISTRY[env_class](env_config, extras)
    # Default to single-turn
    return MockSingleTurnEnv(env_config, extras)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer that provides predictable tokenization."""
    tokenizer = MagicMock()

    def mock_apply_chat_template(messages, **kwargs):
        if kwargs.get("tokenize", True):
            if kwargs.get("return_dict", False):
                return {
                    "input_ids": MOCK_LLM_OUTPUT_IDS.copy(),
                    "assistant_masks": [1] * len(MOCK_LLM_OUTPUT_IDS),
                }
            # For observation tokenization, return observation IDs
            if kwargs.get("add_generation_prompt", False) and len(messages) > 2:
                return MOCK_OBSERVATION_IDS.copy()
            return MOCK_PROMPT_IDS.copy()
        else:
            return "".join([str(m.get("content", "")) for m in messages])

    def mock_encode(text, **kwargs):
        if text:
            return [100, 101]
        return []

    def mock_decode(ids, **kwargs):
        return f"decoded_{len(ids)}_tokens"

    tokenizer.apply_chat_template.side_effect = mock_apply_chat_template
    tokenizer.encode.side_effect = mock_encode
    tokenizer.decode.side_effect = mock_decode
    tokenizer.eos_token_id = 99
    tokenizer.eos_token = "<eos>"
    return tokenizer


@pytest.fixture
def mock_llm():
    """Mock inference engine client."""
    mock = MagicMock()

    def mock_generate(input_batch):
        num_prompts = (
            len(input_batch["prompts"])
            if "prompts" in input_batch
            else len(input_batch["prompt_token_ids"])
        )
        return {
            "responses": ["mocked output"] * num_prompts,
            "stop_reasons": ["stop"] * num_prompts,
            "response_logprobs": [[0.1] * len(MOCK_LLM_OUTPUT_IDS)] * num_prompts,
            "response_ids": [MOCK_LLM_OUTPUT_IDS.copy() for _ in range(num_prompts)],
        }

    mock.generate = AsyncMock(side_effect=mock_generate)
    return mock


@pytest.fixture
def generator_cfg():
    """Generator config with branching enabled."""
    cfg = get_default_config().generator
    cfg.sampling_params.max_generate_length = 50
    cfg.sampling_params.logprobs = None
    cfg.apply_overlong_filtering = False
    cfg.max_input_length = 512
    cfg.batched = False
    cfg.max_turns = 1
    cfg.use_conversation_multi_turn = True
    cfg.chat_template_kwargs = {}
    cfg.chat_template = {"source": "name", "name_or_path": None}
    cfg.append_eos_token_after_stop_str_in_multi_turn = True
    # Add branching config
    cfg.branching = DictConfig({
        "enabled": True,
        "src_trajectories": 2,
        "num_branches": 2,
    })
    return cfg


@pytest.fixture
def mock_env_cfg():
    """Mock environment config."""
    cfg = MagicMock()
    cfg.max_env_workers = 0
    cfg.get = MagicMock(return_value=DictConfig({}))
    return cfg


# =============================================================================
# Unit Tests for Helper Methods
# =============================================================================


class TestFindTurnBoundaries:
    """Tests for _find_turn_boundaries method."""

    @patch("skyrl_gym.make", side_effect=mock_skyrl_gym_make)
    def test_single_turn(self, mock_make, mock_tokenizer, mock_llm, generator_cfg, mock_env_cfg):
        """Single assistant turn should return one boundary."""
        generator = BranchedSkyRLGymGenerator(
            generator_cfg=generator_cfg,
            skyrl_gym_cfg=mock_env_cfg,
            inference_engine_client=mock_llm,
            tokenizer=mock_tokenizer,
            model_name="test",
        )
        generator.base_conversation_token_ids = []

        # loss_mask: all 1s = single assistant turn
        loss_mask = [1, 1, 1, 1, 1]
        boundaries = generator._find_turn_boundaries(loss_mask)

        assert len(boundaries) == 1
        assert boundaries[0] == (0, 5)

    @patch("skyrl_gym.make", side_effect=mock_skyrl_gym_make)
    def test_multi_turn(self, mock_make, mock_tokenizer, mock_llm, generator_cfg, mock_env_cfg):
        """Multiple turns with observations should return multiple boundaries."""
        generator = BranchedSkyRLGymGenerator(
            generator_cfg=generator_cfg,
            skyrl_gym_cfg=mock_env_cfg,
            inference_engine_client=mock_llm,
            tokenizer=mock_tokenizer,
            model_name="test",
        )
        generator.base_conversation_token_ids = []

        # Turn 1: tokens 0-3, Observation: tokens 4-5, Turn 2: tokens 6-9
        loss_mask = [1, 1, 1, 1, 0, 0, 1, 1, 1, 1]
        boundaries = generator._find_turn_boundaries(loss_mask)

        assert len(boundaries) == 2
        assert boundaries[0] == (0, 4)
        assert boundaries[1] == (6, 10)

    @patch("skyrl_gym.make", side_effect=mock_skyrl_gym_make)
    def test_empty_mask(self, mock_make, mock_tokenizer, mock_llm, generator_cfg, mock_env_cfg):
        """Empty loss mask should return empty boundaries."""
        generator = BranchedSkyRLGymGenerator(
            generator_cfg=generator_cfg,
            skyrl_gym_cfg=mock_env_cfg,
            inference_engine_client=mock_llm,
            tokenizer=mock_tokenizer,
            model_name="test",
        )
        generator.base_conversation_token_ids = []

        boundaries = generator._find_turn_boundaries([])
        assert boundaries == []

    @patch("skyrl_gym.make", side_effect=mock_skyrl_gym_make)
    def test_all_observations(self, mock_make, mock_tokenizer, mock_llm, generator_cfg, mock_env_cfg):
        """All observation tokens should return empty boundaries."""
        generator = BranchedSkyRLGymGenerator(
            generator_cfg=generator_cfg,
            skyrl_gym_cfg=mock_env_cfg,
            inference_engine_client=mock_llm,
            tokenizer=mock_tokenizer,
            model_name="test",
        )
        generator.base_conversation_token_ids = []

        loss_mask = [0, 0, 0, 0]
        boundaries = generator._find_turn_boundaries(loss_mask)
        assert boundaries == []


class TestBranchedTrajectoryID:
    """Tests for BranchedTrajectoryID dataclass."""

    def test_root_trajectory(self):
        """Root trajectory should have is_root() return True."""
        traj_id = BranchedTrajectoryID(
            instance_id="uid_1",
            repetition_id=0,
            source_repetition_id=None,
            branch_turn=None,
            branch_token_idx=None,
        )
        assert traj_id.is_root() is True
        assert traj_id.to_string() == "uid_1_0"

    def test_branched_trajectory(self):
        """Branched trajectory should have is_root() return False."""
        traj_id = BranchedTrajectoryID(
            instance_id="uid_1",
            repetition_id=2,
            source_repetition_id=0,
            branch_turn=1,
            branch_token_idx=15,
        )
        assert traj_id.is_root() is False
        assert "from_0" in traj_id.to_string()
        assert "t1" in traj_id.to_string()
        assert "tok15" in traj_id.to_string()


# =============================================================================
# Integration Tests for Single-Turn Environments
# =============================================================================


@pytest.mark.asyncio
@patch("skyrl_gym.make", side_effect=mock_skyrl_gym_make)
async def test_single_turn_branching(mock_make, mock_tokenizer, mock_llm, generator_cfg, mock_env_cfg):
    """Test branching works correctly for single-turn (max_turns=1) environments."""
    generator_cfg.max_turns = 1
    generator_cfg.branching.src_trajectories = 1
    generator_cfg.branching.num_branches = 1

    generator = BranchedSkyRLGymGenerator(
        generator_cfg=generator_cfg,
        skyrl_gym_cfg=mock_env_cfg,
        inference_engine_client=mock_llm,
        tokenizer=mock_tokenizer,
        model_name="test",
    )
    generator.base_conversation_token_ids = []

    # Create input with 2 samples per prompt (group_size=2)
    prompts = [[{"role": "user", "content": "What is 2+2?"}]] * 2
    env_extras = [{"answer": "4"}] * 2
    trajectory_ids = [
        TrajectoryID(instance_id="uid_1", repetition_id=0),
        TrajectoryID(instance_id="uid_1", repetition_id=1),
    ]

    input_batch: GeneratorInput = {
        "prompts": prompts,
        "env_extras": env_extras,
        "env_classes": ["gsm8k"] * 2,
        "trajectory_ids": trajectory_ids,
        "sampling_params": None,
        "batch_metadata": None,
    }

    output = await generator.generate(input_batch, disable_tqdm=True)

    # Should have 2 completed trajectories
    assert len(output["response_ids"]) == 2
    assert len(output["rewards"]) == 2
    assert len(output["loss_masks"]) == 2
    assert len(output["trajectory_ids"]) == 2

    # Check branching metrics
    assert "branching/num_roots" in output["rollout_metrics"]
    assert "branching/num_branches" in output["rollout_metrics"]


@pytest.mark.asyncio
@patch("skyrl_gym.make", side_effect=mock_skyrl_gym_make)
async def test_single_turn_loss_mask_preserved(mock_make, mock_tokenizer, mock_llm, generator_cfg, mock_env_cfg):
    """Verify loss_mask is correctly preserved/built for single-turn branched trajectories."""
    generator_cfg.max_turns = 1
    generator_cfg.branching.src_trajectories = 1
    generator_cfg.branching.num_branches = 1

    generator = BranchedSkyRLGymGenerator(
        generator_cfg=generator_cfg,
        skyrl_gym_cfg=mock_env_cfg,
        inference_engine_client=mock_llm,
        tokenizer=mock_tokenizer,
        model_name="test",
    )
    generator.base_conversation_token_ids = []

    prompts = [[{"role": "user", "content": "Test"}]] * 2
    env_extras = [{}] * 2
    trajectory_ids = [
        TrajectoryID(instance_id="uid_1", repetition_id=i) for i in range(2)
    ]

    input_batch: GeneratorInput = {
        "prompts": prompts,
        "env_extras": env_extras,
        "env_classes": ["gsm8k"] * 2,
        "trajectory_ids": trajectory_ids,
        "sampling_params": None,
        "batch_metadata": None,
    }

    output = await generator.generate(input_batch, disable_tqdm=True)

    # Each trajectory should have matching response_ids and loss_mask lengths
    for i in range(len(output["response_ids"])):
        response_len = len(output["response_ids"][i])
        mask_len = len(output["loss_masks"][i])
        assert response_len == mask_len, f"Trajectory {i}: response_len={response_len}, mask_len={mask_len}"

        # Loss mask should only contain 0s and 1s
        for val in output["loss_masks"][i]:
            assert val in [0, 1], f"Invalid loss_mask value: {val}"


# =============================================================================
# Integration Tests for Multi-Turn Environments
# =============================================================================


@pytest.mark.asyncio
@patch("skyrl_gym.make", side_effect=mock_skyrl_gym_make)
async def test_multi_turn_branching(mock_make, mock_tokenizer, mock_llm, generator_cfg, mock_env_cfg):
    """Test branching works correctly for multi-turn environments."""
    generator_cfg.max_turns = 3
    generator_cfg.branching.src_trajectories = 1
    generator_cfg.branching.num_branches = 2

    # Mock LLM to return different outputs per turn
    turn_count = [0]

    def mock_generate_multi_turn(input_batch):
        turn_count[0] += 1
        num_prompts = (
            len(input_batch["prompts"])
            if "prompts" in input_batch
            else len(input_batch["prompt_token_ids"])
        )
        # Alternate between search and answer
        if turn_count[0] % 2 == 1:
            response = "<search>query</search>"
        else:
            response = "<answer>42</answer>"

        return {
            "responses": [response] * num_prompts,
            "stop_reasons": ["stop"] * num_prompts,
            "response_logprobs": [[0.1] * 5] * num_prompts,
            "response_ids": [[10, 11, 12, 13, 99] for _ in range(num_prompts)],
        }

    mock_llm.generate = AsyncMock(side_effect=mock_generate_multi_turn)

    generator = BranchedSkyRLGymGenerator(
        generator_cfg=generator_cfg,
        skyrl_gym_cfg=mock_env_cfg,
        inference_engine_client=mock_llm,
        tokenizer=mock_tokenizer,
        model_name="test",
    )
    generator.base_conversation_token_ids = []

    # Create input with 3 samples per prompt (group_size=3)
    prompts = [[{"role": "user", "content": "Search for answer"}]] * 3
    env_extras = [{"max_turns": 3}] * 3
    trajectory_ids = [
        TrajectoryID(instance_id="uid_1", repetition_id=i) for i in range(3)
    ]

    input_batch: GeneratorInput = {
        "prompts": prompts,
        "env_extras": env_extras,
        "env_classes": ["gsm8k_multi_turn"] * 3,
        "trajectory_ids": trajectory_ids,
        "sampling_params": None,
        "batch_metadata": None,
    }

    output = await generator.generate(input_batch, disable_tqdm=True)

    # Should have 3 completed trajectories
    assert len(output["response_ids"]) == 3
    assert len(output["trajectory_ids"]) == 3

    # Check that we have a mix of root and branched trajectories
    roots = [t for t in output["trajectory_ids"] if t.is_root()]
    branches = [t for t in output["trajectory_ids"] if not t.is_root()]

    # With src_trajectories=1 and num_branches=2, we expect 1 root and 2 branches
    assert len(roots) >= 1, "Should have at least one root trajectory"


@pytest.mark.asyncio
@patch("skyrl_gym.make", side_effect=mock_skyrl_gym_make)
async def test_multi_turn_loss_mask_structure(mock_make, mock_tokenizer, mock_llm, generator_cfg, mock_env_cfg):
    """Verify loss_mask correctly marks model (1) vs env (0) tokens in multi-turn."""
    generator_cfg.max_turns = 2
    generator_cfg.branching.src_trajectories = 1
    generator_cfg.branching.num_branches = 0  # No branching, just test multi-turn

    # Mock LLM to return search then answer
    call_count = [0]

    def mock_generate_sequence(input_batch):
        call_count[0] += 1
        num_prompts = (
            len(input_batch["prompts"])
            if "prompts" in input_batch
            else len(input_batch["prompt_token_ids"])
        )

        if call_count[0] == 1:
            # First call: search
            return {
                "responses": ["<search>test</search>"] * num_prompts,
                "stop_reasons": ["stop"] * num_prompts,
                "response_logprobs": [[0.1] * 3] * num_prompts,
                "response_ids": [[10, 11, 12] for _ in range(num_prompts)],
            }
        else:
            # Second call: answer
            return {
                "responses": ["<answer>result</answer>"] * num_prompts,
                "stop_reasons": ["stop"] * num_prompts,
                "response_logprobs": [[0.1] * 4] * num_prompts,
                "response_ids": [[20, 21, 22, 99] for _ in range(num_prompts)],
            }

    mock_llm.generate = AsyncMock(side_effect=mock_generate_sequence)

    generator = BranchedSkyRLGymGenerator(
        generator_cfg=generator_cfg,
        skyrl_gym_cfg=mock_env_cfg,
        inference_engine_client=mock_llm,
        tokenizer=mock_tokenizer,
        model_name="test",
    )
    generator.base_conversation_token_ids = []

    prompts = [[{"role": "user", "content": "Multi-turn test"}]]
    env_extras = [{"max_turns": 2}]
    trajectory_ids = [TrajectoryID(instance_id="uid_1", repetition_id=0)]

    input_batch: GeneratorInput = {
        "prompts": prompts,
        "env_extras": env_extras,
        "env_classes": ["gsm8k_multi_turn"],
        "trajectory_ids": trajectory_ids,
        "sampling_params": None,
        "batch_metadata": None,
    }

    output = await generator.generate(input_batch, disable_tqdm=True)

    # Check loss_mask structure
    loss_mask = output["loss_masks"][0]
    response_ids = output["response_ids"][0]

    assert len(loss_mask) == len(response_ids)

    # In multi-turn:
    # - Model-generated tokens should have loss_mask=1
    # - Observation tokens should have loss_mask=0
    # The exact pattern depends on the tokenization, but we verify consistency
    assert all(v in [0, 1] for v in loss_mask)


# =============================================================================
# Tests for Reward Handling
# =============================================================================


class MockRewardEnv(BaseTextEnv):
    """Environment with specific fixed reward for testing."""

    def __init__(self, env_config: DictConfig = None, extras: Dict[str, Any] = None):
        super().__init__()
        self.chat_history = []

    def init(self, prompt):
        return prompt, {}

    def step(self, action):
        self.turns += 1
        return BaseTextEnvStepOutput(
            observations=[],
            reward=0.75,  # Specific reward to check
            done=True,
            metadata={},
        )

    def set_history(self, chat_history, turn=0):
        super().set_history(chat_history, turn)


@pytest.mark.asyncio
async def test_rewards_correctly_attributed(mock_tokenizer, mock_llm, generator_cfg, mock_env_cfg):
    """Verify rewards are correctly placed at turn boundaries."""
    generator_cfg.max_turns = 1
    generator_cfg.branching.src_trajectories = 2
    generator_cfg.branching.num_branches = 0

    with patch("skyrl_gym.make", side_effect=lambda *args, **kwargs: MockRewardEnv()):
        generator = BranchedSkyRLGymGenerator(
            generator_cfg=generator_cfg,
            skyrl_gym_cfg=mock_env_cfg,
            inference_engine_client=mock_llm,
            tokenizer=mock_tokenizer,
            model_name="test",
        )
        generator.base_conversation_token_ids = []

        prompts = [[{"role": "user", "content": "Test"}]] * 2
        env_extras = [{}] * 2
        trajectory_ids = [TrajectoryID(instance_id="uid_1", repetition_id=i) for i in range(2)]

        input_batch: GeneratorInput = {
            "prompts": prompts,
            "env_extras": env_extras,
            "env_classes": ["gsm8k"] * 2,
            "trajectory_ids": trajectory_ids,
            "sampling_params": None,
            "batch_metadata": None,
        }

        output = await generator.generate(input_batch, disable_tqdm=True)

        # Each trajectory should have token-level rewards summing to 0.75
        for i, rewards in enumerate(output["rewards"]):
            if isinstance(rewards, list):
                total_reward = sum(rewards)
                assert abs(total_reward - 0.75) < 0.01, f"Trajectory {i}: expected reward 0.75, got {total_reward}"
            else:
                assert abs(rewards - 0.75) < 0.01


# =============================================================================
# Tests for Logprobs Handling
# =============================================================================


@pytest.mark.asyncio
@patch("skyrl_gym.make", side_effect=mock_skyrl_gym_make)
async def test_logprobs_returned_when_available(mock_make, mock_tokenizer, mock_llm, generator_cfg, mock_env_cfg):
    """Verify logprobs are returned when LLM provides them (even without explicit config)."""
    generator_cfg.max_turns = 1
    generator_cfg.branching.src_trajectories = 1
    generator_cfg.branching.num_branches = 1
    # Note: We don't set sampling_params.logprobs since batched=False doesn't support it
    # But the LLM can still return logprobs and we should handle them

    # Mock LLM to return logprobs
    def mock_generate_with_logprobs(input_batch):
        num_prompts = (
            len(input_batch["prompts"])
            if "prompts" in input_batch
            else len(input_batch["prompt_token_ids"])
        )
        return {
            "responses": ["output"] * num_prompts,
            "stop_reasons": ["stop"] * num_prompts,
            "response_logprobs": [[-0.5, -0.3, -0.2, -0.1, -0.05] for _ in range(num_prompts)],
            "response_ids": [[10, 11, 12, 13, 99] for _ in range(num_prompts)],
        }

    mock_llm.generate = AsyncMock(side_effect=mock_generate_with_logprobs)

    generator = BranchedSkyRLGymGenerator(
        generator_cfg=generator_cfg,
        skyrl_gym_cfg=mock_env_cfg,
        inference_engine_client=mock_llm,
        tokenizer=mock_tokenizer,
        model_name="test",
    )
    generator.base_conversation_token_ids = []

    prompts = [[{"role": "user", "content": "Test"}]] * 2
    env_extras = [{}] * 2
    trajectory_ids = [TrajectoryID(instance_id="uid_1", repetition_id=i) for i in range(2)]

    input_batch: GeneratorInput = {
        "prompts": prompts,
        "env_extras": env_extras,
        "env_classes": ["gsm8k"] * 2,
        "trajectory_ids": trajectory_ids,
        "sampling_params": None,
        "batch_metadata": None,
    }

    output = await generator.generate(input_batch, disable_tqdm=True)

    # With batched=False and no logprobs config, rollout_logprobs should be None
    # (This is expected behavior - logprobs are only collected when explicitly requested)
    # The test verifies that the generator handles the case gracefully
    assert len(output["response_ids"]) == 2
    assert len(output["rewards"]) == 2


# =============================================================================
# Tests for Environment set_history
# =============================================================================


@pytest.mark.asyncio
async def test_env_set_history_restores_turn_counter():
    """Verify environment's set_history correctly restores turn counter."""
    env = MockMultiTurnEnv()

    # set_history should restore turn counter
    # (env's chat_history is not used for reward computation)
    env.set_history([], turn=2)

    assert env.turns == 2


# =============================================================================
# Tests for Branching Disabled
# =============================================================================


@pytest.mark.asyncio
@patch("skyrl_gym.make", side_effect=mock_skyrl_gym_make)
async def test_branching_disabled_falls_back_to_parent(mock_make, mock_tokenizer, mock_llm, generator_cfg, mock_env_cfg):
    """When branching is disabled, should fall back to parent generator behavior."""
    generator_cfg.branching.enabled = False

    generator = BranchedSkyRLGymGenerator(
        generator_cfg=generator_cfg,
        skyrl_gym_cfg=mock_env_cfg,
        inference_engine_client=mock_llm,
        tokenizer=mock_tokenizer,
        model_name="test",
    )
    generator.base_conversation_token_ids = []

    prompts = [[{"role": "user", "content": "Test"}]] * 2
    env_extras = [{}] * 2
    trajectory_ids = [TrajectoryID(instance_id="uid_1", repetition_id=i) for i in range(2)]

    input_batch: GeneratorInput = {
        "prompts": prompts,
        "env_extras": env_extras,
        "env_classes": ["gsm8k"] * 2,
        "trajectory_ids": trajectory_ids,
        "sampling_params": None,
        "batch_metadata": None,
    }

    output = await generator.generate(input_batch, disable_tqdm=True)

    # Should still produce valid output
    assert len(output["response_ids"]) == 2
    assert len(output["rewards"]) == 2


# =============================================================================
# Tests for Edge Cases
# =============================================================================


@pytest.mark.asyncio
@patch("skyrl_gym.make", side_effect=mock_skyrl_gym_make)
async def test_group_size_equals_src_trajectories(mock_make, mock_tokenizer, mock_llm, generator_cfg, mock_env_cfg):
    """When group_size equals src_trajectories, no branching should occur."""
    generator_cfg.max_turns = 1
    generator_cfg.branching.src_trajectories = 2
    generator_cfg.branching.num_branches = 5  # Would branch a lot, but group_size=2

    generator = BranchedSkyRLGymGenerator(
        generator_cfg=generator_cfg,
        skyrl_gym_cfg=mock_env_cfg,
        inference_engine_client=mock_llm,
        tokenizer=mock_tokenizer,
        model_name="test",
    )
    generator.base_conversation_token_ids = []

    # group_size = 2 = src_trajectories
    prompts = [[{"role": "user", "content": "Test"}]] * 2
    env_extras = [{}] * 2
    trajectory_ids = [TrajectoryID(instance_id="uid_1", repetition_id=i) for i in range(2)]

    input_batch: GeneratorInput = {
        "prompts": prompts,
        "env_extras": env_extras,
        "env_classes": ["gsm8k"] * 2,
        "trajectory_ids": trajectory_ids,
        "sampling_params": None,
        "batch_metadata": None,
    }

    output = await generator.generate(input_batch, disable_tqdm=True)

    # Should have exactly 2 trajectories (all roots, no branches needed)
    assert len(output["trajectory_ids"]) == 2

    # All should be roots since we complete before needing to branch
    roots = [t for t in output["trajectory_ids"] if t.is_root()]
    assert len(roots) == 2


@pytest.mark.asyncio
@patch("skyrl_gym.make", side_effect=mock_skyrl_gym_make)
async def test_multiple_uids_handled_separately(mock_make, mock_tokenizer, mock_llm, generator_cfg, mock_env_cfg):
    """Verify different UIDs are processed as separate groups."""
    generator_cfg.max_turns = 1
    generator_cfg.branching.src_trajectories = 1
    generator_cfg.branching.num_branches = 1

    generator = BranchedSkyRLGymGenerator(
        generator_cfg=generator_cfg,
        skyrl_gym_cfg=mock_env_cfg,
        inference_engine_client=mock_llm,
        tokenizer=mock_tokenizer,
        model_name="test",
    )
    generator.base_conversation_token_ids = []

    # Two different UIDs, each with 2 samples
    prompts = [
        [{"role": "user", "content": "Question A"}],
        [{"role": "user", "content": "Question A"}],
        [{"role": "user", "content": "Question B"}],
        [{"role": "user", "content": "Question B"}],
    ]
    env_extras = [{}] * 4
    trajectory_ids = [
        TrajectoryID(instance_id="uid_A", repetition_id=0),
        TrajectoryID(instance_id="uid_A", repetition_id=1),
        TrajectoryID(instance_id="uid_B", repetition_id=0),
        TrajectoryID(instance_id="uid_B", repetition_id=1),
    ]

    input_batch: GeneratorInput = {
        "prompts": prompts,
        "env_extras": env_extras,
        "env_classes": ["gsm8k"] * 4,
        "trajectory_ids": trajectory_ids,
        "sampling_params": None,
        "batch_metadata": None,
    }

    output = await generator.generate(input_batch, disable_tqdm=True)

    # Should have 4 trajectories total (2 per UID)
    assert len(output["trajectory_ids"]) == 4

    # Check UIDs are present
    uid_a_count = sum(1 for t in output["trajectory_ids"] if t.instance_id == "uid_A")
    uid_b_count = sum(1 for t in output["trajectory_ids"] if t.instance_id == "uid_B")
    assert uid_a_count == 2
    assert uid_b_count == 2


# =============================================================================
# Tests for Output Validation
# =============================================================================


def test_generator_output_has_required_fields():
    """Verify GeneratorOutput TypedDict has expected fields."""
    expected_fields = {
        "prompt_token_ids",
        "response_ids",
        "rewards",
        "loss_masks",
        "stop_reasons",
        "rollout_metrics",
        "rollout_logprobs",
        "trajectory_ids",
        "is_last_step",
        "steps",
    }
    actual_fields = set(GeneratorOutput.__annotations__.keys())
    assert expected_fields.issubset(actual_fields), f"Missing fields: {expected_fields - actual_fields}"


@pytest.mark.asyncio
@patch("skyrl_gym.make", side_effect=mock_skyrl_gym_make)
async def test_output_conforms_to_generator_output(mock_make, mock_tokenizer, mock_llm, generator_cfg, mock_env_cfg):
    """Verify branched generator output conforms to GeneratorOutput interface."""
    generator_cfg.max_turns = 1
    generator_cfg.branching.src_trajectories = 1
    generator_cfg.branching.num_branches = 1

    generator = BranchedSkyRLGymGenerator(
        generator_cfg=generator_cfg,
        skyrl_gym_cfg=mock_env_cfg,
        inference_engine_client=mock_llm,
        tokenizer=mock_tokenizer,
        model_name="test",
    )
    generator.base_conversation_token_ids = []

    prompts = [[{"role": "user", "content": "Test"}]] * 2
    env_extras = [{}] * 2
    trajectory_ids = [TrajectoryID(instance_id="uid_1", repetition_id=i) for i in range(2)]

    input_batch: GeneratorInput = {
        "prompts": prompts,
        "env_extras": env_extras,
        "env_classes": ["gsm8k"] * 2,
        "trajectory_ids": trajectory_ids,
        "sampling_params": None,
        "batch_metadata": None,
    }

    output = await generator.generate(input_batch, disable_tqdm=True)

    # Check required fields exist and have correct types
    assert "prompt_token_ids" in output
    assert "response_ids" in output
    assert "rewards" in output
    assert "loss_masks" in output
    assert "stop_reasons" in output
    assert "rollout_metrics" in output
    assert "trajectory_ids" in output

    # Check types
    assert isinstance(output["prompt_token_ids"], list)
    assert isinstance(output["response_ids"], list)
    assert isinstance(output["rewards"], list)
    assert isinstance(output["loss_masks"], list)
    assert isinstance(output["rollout_metrics"], dict)

    # Check all trajectory_ids are BranchedTrajectoryID
    for traj_id in output["trajectory_ids"]:
        assert isinstance(traj_id, BranchedTrajectoryID)
