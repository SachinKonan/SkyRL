"""
Tests for Gemini-based branching point selection.

uv run --extra dev pytest tests/cpu/generators/test_gemini_branching.py -v
"""

import pytest
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch
from omegaconf import DictConfig

from skyrl_train.generators.branched_skyrl_gym_generator import (
    BranchedSkyRLGymGenerator,
    BranchPointMetadata,
)
from skyrl_train.generators.base import TrajectoryID
from skyrl_train.generators.skyrl_gym_generator import AgentLoopOutput
from skyrl_train.config.utils import get_default_config


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer that provides predictable tokenization."""
    tokenizer = MagicMock()

    # Token -> string mapping for decode
    token_strings = {
        10: "Hello",
        11: " world",
        12: ",",
        13: " I",
        14: " am",
        15: " searching",
        16: " for",
        17: " the",
        18: " answer",
        19: ".",
        99: "<eos>",
    }

    def mock_decode(ids, **kwargs):
        return "".join(token_strings.get(i, f"[{i}]") for i in ids)

    def mock_encode(text, **kwargs):
        # Simple reverse mapping for testing
        if text == "Hello world":
            return [10, 11]
        if text == " searching for":
            return [15, 16]
        if text == " the answer":
            return [17, 18]
        if text == "NOT_FOUND":
            return [999, 998]  # Tokens that don't exist
        return [100, 101]  # Default

    def mock_apply_chat_template(messages, **kwargs):
        if kwargs.get("tokenize", True):
            return [1, 2, 3, 4]
        return "mock_chat"

    tokenizer.decode.side_effect = mock_decode
    tokenizer.encode.side_effect = mock_encode
    tokenizer.apply_chat_template.side_effect = mock_apply_chat_template
    tokenizer.eos_token_id = 99
    tokenizer.eos_token = "<eos>"
    return tokenizer


@pytest.fixture
def mock_llm():
    """Mock inference engine client."""
    mock = MagicMock()

    def mock_generate(input_batch):
        num_prompts = len(input_batch.get("prompt_token_ids", [[]]))
        return {
            "responses": ["mocked output"] * num_prompts,
            "stop_reasons": ["stop"] * num_prompts,
            "response_logprobs": [[0.1] * 5] * num_prompts,
            "response_ids": [[10, 11, 12, 13, 99] for _ in range(num_prompts)],
        }

    mock.generate = AsyncMock(side_effect=mock_generate)
    return mock


@pytest.fixture
def generator_cfg_gemini():
    """Generator config with Gemini branching strategy."""
    cfg = get_default_config().generator
    cfg.sampling_params.max_generate_length = 50
    cfg.sampling_params.logprobs = None
    cfg.apply_overlong_filtering = False
    cfg.max_input_length = 512
    cfg.batched = False
    cfg.max_turns = 3
    cfg.use_conversation_multi_turn = True
    cfg.chat_template_kwargs = {}
    cfg.chat_template = {"source": "name", "name_or_path": None}
    cfg.append_eos_token_after_stop_str_in_multi_turn = True
    # Gemini branching config
    cfg.branching = DictConfig({
        "enabled": True,
        "src_trajectories": 2,
        "num_branches": 2,
        "strategy": "gemini",
        "gemini": DictConfig({
            "model": "gemini-2.5-flash",
            "timeout": 30.0,
            "max_retries": 3,
        }),
    })
    return cfg


@pytest.fixture
def mock_env_cfg():
    """Mock environment config."""
    cfg = MagicMock()
    cfg.max_env_workers = 0
    cfg.get = MagicMock(return_value=DictConfig({}))
    return cfg


@pytest.fixture
def mock_agent_output():
    """Mock AgentLoopOutput for testing."""
    return AgentLoopOutput(
        response_ids=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        reward=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        stop_reason="stop",
        loss_mask=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # All assistant tokens
        prompt_ids=[1, 2, 3, 4],
        rollout_logprobs=[-0.1] * 10,
        env_metrics={},
        steps=[
            {"start_ix": 0, "end_ix": 5, "type": "model", "time_elapsed_s": 0.5},
            {"start_ix": 5, "end_ix": 10, "type": "model", "time_elapsed_s": 0.3},
        ],
    )


# =============================================================================
# Tests for _find_token_sequence_position
# =============================================================================


class TestFindTokenSequencePosition:
    """Tests for the token sequence matching method."""

    @patch("skyrl_gym.make")
    def test_exact_match_found(self, mock_make, mock_tokenizer, mock_llm, generator_cfg_gemini, mock_env_cfg):
        """Test finding an exact token sequence match."""
        generator = BranchedSkyRLGymGenerator(
            generator_cfg=generator_cfg_gemini,
            skyrl_gym_cfg=mock_env_cfg,
            inference_engine_client=mock_llm,
            tokenizer=mock_tokenizer,
            model_name="test",
        )
        generator.base_conversation_token_ids = []

        # response_ids contains [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        # "Hello world" encodes to [10, 11]
        response_ids = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        position = generator._find_token_sequence_position("Hello world", response_ids)

        # Should find at position 2 (AFTER [10, 11])
        assert position == 2

    @patch("skyrl_gym.make")
    def test_match_in_middle(self, mock_make, mock_tokenizer, mock_llm, generator_cfg_gemini, mock_env_cfg):
        """Test finding a match in the middle of the sequence."""
        generator = BranchedSkyRLGymGenerator(
            generator_cfg=generator_cfg_gemini,
            skyrl_gym_cfg=mock_env_cfg,
            inference_engine_client=mock_llm,
            tokenizer=mock_tokenizer,
            model_name="test",
        )
        generator.base_conversation_token_ids = []

        # " searching for" encodes to [15, 16]
        response_ids = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        position = generator._find_token_sequence_position(" searching for", response_ids)

        # Should find at position 7 (AFTER [15, 16] which are at indices 5, 6)
        assert position == 7

    @patch("skyrl_gym.make")
    def test_no_match_returns_none(self, mock_make, mock_tokenizer, mock_llm, generator_cfg_gemini, mock_env_cfg):
        """Test that non-matching substring returns None."""
        generator = BranchedSkyRLGymGenerator(
            generator_cfg=generator_cfg_gemini,
            skyrl_gym_cfg=mock_env_cfg,
            inference_engine_client=mock_llm,
            tokenizer=mock_tokenizer,
            model_name="test",
        )
        generator.base_conversation_token_ids = []

        # "NOT_FOUND" encodes to [999, 998] which aren't in response_ids
        response_ids = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        position = generator._find_token_sequence_position("NOT_FOUND", response_ids)

        assert position is None

    @patch("skyrl_gym.make")
    def test_multiple_matches_returns_later(self, mock_make, mock_tokenizer, mock_llm, generator_cfg_gemini, mock_env_cfg):
        """Test that multiple matches return the later one."""
        generator = BranchedSkyRLGymGenerator(
            generator_cfg=generator_cfg_gemini,
            skyrl_gym_cfg=mock_env_cfg,
            inference_engine_client=mock_llm,
            tokenizer=mock_tokenizer,
            model_name="test",
        )
        generator.base_conversation_token_ids = []

        # Create response with duplicate sequence [10, 11] appearing twice
        response_ids = [10, 11, 5, 6, 10, 11, 7, 8]
        # "Hello world" encodes to [10, 11]
        position = generator._find_token_sequence_position("Hello world", response_ids)

        # Should return position after the LATER match (index 6)
        assert position == 6


# =============================================================================
# Tests for _validate_branch_position
# =============================================================================


class TestValidateBranchPosition:
    """Tests for validating branch positions."""

    @patch("skyrl_gym.make")
    def test_valid_position_in_assistant_turn(self, mock_make, mock_tokenizer, mock_llm, generator_cfg_gemini, mock_env_cfg):
        """Test that position in assistant turn (loss_mask=1) is valid."""
        generator = BranchedSkyRLGymGenerator(
            generator_cfg=generator_cfg_gemini,
            skyrl_gym_cfg=mock_env_cfg,
            inference_engine_client=mock_llm,
            tokenizer=mock_tokenizer,
            model_name="test",
        )
        generator.base_conversation_token_ids = []

        loss_mask = [1, 1, 1, 1, 1]
        steps = [{"start_ix": 0, "end_ix": 5, "type": "model"}]

        assert generator._validate_branch_position(2, loss_mask, steps) is True

    @patch("skyrl_gym.make")
    def test_invalid_position_in_observation(self, mock_make, mock_tokenizer, mock_llm, generator_cfg_gemini, mock_env_cfg):
        """Test that position in observation (loss_mask=0) is invalid."""
        generator = BranchedSkyRLGymGenerator(
            generator_cfg=generator_cfg_gemini,
            skyrl_gym_cfg=mock_env_cfg,
            inference_engine_client=mock_llm,
            tokenizer=mock_tokenizer,
            model_name="test",
        )
        generator.base_conversation_token_ids = []

        # loss_mask with observation region
        loss_mask = [1, 1, 0, 0, 1, 1]
        steps = [
            {"start_ix": 0, "end_ix": 2, "type": "model"},
            {"start_ix": 2, "end_ix": 4, "type": "env"},
            {"start_ix": 4, "end_ix": 6, "type": "model"},
        ]

        # Position 2 is in observation (loss_mask=0)
        assert generator._validate_branch_position(2, loss_mask, steps) is False

    @patch("skyrl_gym.make")
    def test_invalid_position_at_start(self, mock_make, mock_tokenizer, mock_llm, generator_cfg_gemini, mock_env_cfg):
        """Test that position at very start is invalid."""
        generator = BranchedSkyRLGymGenerator(
            generator_cfg=generator_cfg_gemini,
            skyrl_gym_cfg=mock_env_cfg,
            inference_engine_client=mock_llm,
            tokenizer=mock_tokenizer,
            model_name="test",
        )
        generator.base_conversation_token_ids = []

        loss_mask = [1, 1, 1, 1, 1]
        steps = [{"start_ix": 0, "end_ix": 5, "type": "model"}]

        # Position 0 is too early
        assert generator._validate_branch_position(0, loss_mask, steps) is False

    @patch("skyrl_gym.make")
    def test_invalid_position_at_end(self, mock_make, mock_tokenizer, mock_llm, generator_cfg_gemini, mock_env_cfg):
        """Test that position at very end is invalid."""
        generator = BranchedSkyRLGymGenerator(
            generator_cfg=generator_cfg_gemini,
            skyrl_gym_cfg=mock_env_cfg,
            inference_engine_client=mock_llm,
            tokenizer=mock_tokenizer,
            model_name="test",
        )
        generator.base_conversation_token_ids = []

        loss_mask = [1, 1, 1, 1, 1]
        steps = [{"start_ix": 0, "end_ix": 5, "type": "model"}]

        # Position 4 is at end (len-1)
        assert generator._validate_branch_position(4, loss_mask, steps) is False


# =============================================================================
# Tests for Gemini Branch Point Selection
# =============================================================================


class TestGeminiBranchPointSelection:
    """Tests for the full Gemini branch point selection flow."""

    @patch("skyrl_gym.make")
    @patch("skyrl_train.generators.branched_skyrl_gym_generator.BranchedSkyRLGymGenerator._get_gemini_client")
    def test_gemini_success(
        self, mock_get_client, mock_make, mock_tokenizer, mock_llm, generator_cfg_gemini, mock_env_cfg, mock_agent_output
    ):
        """Test successful Gemini-based branching."""
        # Setup mock Gemini client
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.suggested_substring = " searching for"  # Encodes to [15, 16]
        mock_result.raw_response = " searching for"
        mock_result.fallback_reason = None
        mock_client.select_branch_point.return_value = mock_result
        mock_get_client.return_value = mock_client

        generator = BranchedSkyRLGymGenerator(
            generator_cfg=generator_cfg_gemini,
            skyrl_gym_cfg=mock_env_cfg,
            inference_engine_client=mock_llm,
            tokenizer=mock_tokenizer,
            model_name="test",
        )
        generator.base_conversation_token_ids = []

        prompt = [{"role": "user", "content": "test"}]
        branch_point = generator._sample_branch_point_gemini(mock_agent_output, prompt)

        assert branch_point is not None
        assert branch_point.metadata is not None
        assert branch_point.metadata.strategy == "gemini"
        assert branch_point.metadata.fallback_occurred is False
        assert branch_point.metadata.gemini_suggested_substring == " searching for"

    @patch("skyrl_gym.make")
    @patch("skyrl_train.generators.branched_skyrl_gym_generator.BranchedSkyRLGymGenerator._get_gemini_client")
    def test_gemini_fallback_on_api_error(
        self, mock_get_client, mock_make, mock_tokenizer, mock_llm, generator_cfg_gemini, mock_env_cfg, mock_agent_output
    ):
        """Test fallback to random when Gemini API fails."""
        # Setup mock to raise exception
        mock_get_client.side_effect = Exception("API Error")

        generator = BranchedSkyRLGymGenerator(
            generator_cfg=generator_cfg_gemini,
            skyrl_gym_cfg=mock_env_cfg,
            inference_engine_client=mock_llm,
            tokenizer=mock_tokenizer,
            model_name="test",
        )
        generator.base_conversation_token_ids = []

        prompt = [{"role": "user", "content": "test"}]
        branch_point = generator._sample_branch_point_gemini(mock_agent_output, prompt)

        assert branch_point is not None
        assert branch_point.metadata.fallback_occurred is True
        assert "client_error" in branch_point.metadata.fallback_reason

    @patch("skyrl_gym.make")
    @patch("skyrl_train.generators.branched_skyrl_gym_generator.BranchedSkyRLGymGenerator._get_gemini_client")
    def test_gemini_fallback_on_token_not_found(
        self, mock_get_client, mock_make, mock_tokenizer, mock_llm, generator_cfg_gemini, mock_env_cfg, mock_agent_output
    ):
        """Test fallback when Gemini substring tokens not found in response."""
        # Setup mock Gemini client to return substring that doesn't exist
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.suggested_substring = "NOT_FOUND"  # Encodes to [999, 998]
        mock_result.raw_response = "NOT_FOUND"
        mock_result.fallback_reason = None
        mock_client.select_branch_point.return_value = mock_result
        mock_get_client.return_value = mock_client

        generator = BranchedSkyRLGymGenerator(
            generator_cfg=generator_cfg_gemini,
            skyrl_gym_cfg=mock_env_cfg,
            inference_engine_client=mock_llm,
            tokenizer=mock_tokenizer,
            model_name="test",
        )
        generator.base_conversation_token_ids = []

        prompt = [{"role": "user", "content": "test"}]
        branch_point = generator._sample_branch_point_gemini(mock_agent_output, prompt)

        assert branch_point is not None
        assert branch_point.metadata.fallback_occurred is True
        assert branch_point.metadata.fallback_reason == "token_sequence_not_found"

    @patch("skyrl_gym.make")
    @patch("skyrl_train.generators.branched_skyrl_gym_generator.BranchedSkyRLGymGenerator._get_gemini_client")
    def test_gemini_fallback_on_empty_response(
        self, mock_get_client, mock_make, mock_tokenizer, mock_llm, generator_cfg_gemini, mock_env_cfg, mock_agent_output
    ):
        """Test fallback when Gemini returns empty/failed response."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.suggested_substring = None
        mock_result.raw_response = ""
        mock_result.fallback_reason = "empty_response"
        mock_client.select_branch_point.return_value = mock_result
        mock_get_client.return_value = mock_client

        generator = BranchedSkyRLGymGenerator(
            generator_cfg=generator_cfg_gemini,
            skyrl_gym_cfg=mock_env_cfg,
            inference_engine_client=mock_llm,
            tokenizer=mock_tokenizer,
            model_name="test",
        )
        generator.base_conversation_token_ids = []

        prompt = [{"role": "user", "content": "test"}]
        branch_point = generator._sample_branch_point_gemini(mock_agent_output, prompt)

        assert branch_point is not None
        assert branch_point.metadata.fallback_occurred is True
        assert branch_point.metadata.fallback_reason == "empty_response"


# =============================================================================
# Tests for Strategy Dispatcher
# =============================================================================


class TestBranchingStrategyDispatcher:
    """Tests for the branching strategy dispatcher."""

    @patch("skyrl_gym.make")
    def test_random_strategy_selected(self, mock_make, mock_tokenizer, mock_llm, mock_env_cfg, mock_agent_output):
        """Test that random strategy is used when configured."""
        cfg = get_default_config().generator
        cfg.sampling_params.max_generate_length = 50
        cfg.max_input_length = 512
        cfg.batched = False
        cfg.max_turns = 3
        cfg.use_conversation_multi_turn = True
        cfg.chat_template_kwargs = {}
        cfg.chat_template = {"source": "name", "name_or_path": None}
        cfg.append_eos_token_after_stop_str_in_multi_turn = True
        cfg.branching = DictConfig({
            "enabled": True,
            "src_trajectories": 2,
            "num_branches": 2,
            "strategy": "random",  # Explicitly random
        })

        generator = BranchedSkyRLGymGenerator(
            generator_cfg=cfg,
            skyrl_gym_cfg=mock_env_cfg,
            inference_engine_client=mock_llm,
            tokenizer=mock_tokenizer,
            model_name="test",
        )
        generator.base_conversation_token_ids = []

        prompt = [{"role": "user", "content": "test"}]
        branch_point = generator._sample_branch_point(mock_agent_output, prompt)

        assert branch_point is not None
        assert branch_point.metadata.strategy == "random"

    @patch("skyrl_gym.make")
    @patch("skyrl_train.generators.branched_skyrl_gym_generator.BranchedSkyRLGymGenerator._sample_branch_point_gemini")
    def test_gemini_strategy_selected(
        self, mock_gemini_method, mock_make, mock_tokenizer, mock_llm, generator_cfg_gemini, mock_env_cfg, mock_agent_output
    ):
        """Test that gemini strategy is used when configured."""
        # Setup mock to return a valid branch point
        mock_branch_point = MagicMock()
        mock_branch_point.metadata = BranchPointMetadata(strategy="gemini")
        mock_gemini_method.return_value = mock_branch_point

        generator = BranchedSkyRLGymGenerator(
            generator_cfg=generator_cfg_gemini,
            skyrl_gym_cfg=mock_env_cfg,
            inference_engine_client=mock_llm,
            tokenizer=mock_tokenizer,
            model_name="test",
        )
        generator.base_conversation_token_ids = []

        prompt = [{"role": "user", "content": "test"}]
        branch_point = generator._sample_branch_point(mock_agent_output, prompt)

        # Verify gemini method was called
        mock_gemini_method.assert_called_once()
        assert branch_point.metadata.strategy == "gemini"


# =============================================================================
# Tests for BranchPointMetadata
# =============================================================================


class TestBranchPointMetadata:
    """Tests for BranchPointMetadata dataclass."""

    def test_random_metadata_defaults(self):
        """Test default values for random strategy metadata."""
        metadata = BranchPointMetadata(strategy="random")

        assert metadata.strategy == "random"
        assert metadata.gemini_raw_response is None
        assert metadata.gemini_suggested_substring is None
        assert metadata.fallback_occurred is False
        assert metadata.fallback_reason is None
        assert metadata.token_position_found is None

    def test_gemini_success_metadata(self):
        """Test metadata for successful Gemini selection."""
        metadata = BranchPointMetadata(
            strategy="gemini",
            gemini_raw_response="search for",
            gemini_suggested_substring="search for",
            fallback_occurred=False,
            fallback_reason=None,
            token_position_found=15,
        )

        assert metadata.strategy == "gemini"
        assert metadata.fallback_occurred is False
        assert metadata.token_position_found == 15

    def test_gemini_fallback_metadata(self):
        """Test metadata for Gemini fallback."""
        metadata = BranchPointMetadata(
            strategy="gemini",
            gemini_raw_response="invalid response",
            gemini_suggested_substring="invalid",
            fallback_occurred=True,
            fallback_reason="token_sequence_not_found",
            token_position_found=None,
        )

        assert metadata.strategy == "gemini"
        assert metadata.fallback_occurred is True
        assert metadata.fallback_reason == "token_sequence_not_found"
