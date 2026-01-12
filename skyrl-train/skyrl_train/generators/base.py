from typing import List, Dict, Any, TypedDict, Optional, Union, Literal
from abc import ABC, abstractmethod
from dataclasses import dataclass
from skyrl_train.inference_engines.base import ConversationType

TrainingPhase = Literal["train", "eval"]


@dataclass
class TrajectoryID:
    instance_id: str  # Unique identifier for the instance in the dataset
    repetition_id: int  # Which sample/repetition for this UID (0, 1, 2... for GRPO)

    def to_string(self) -> str:
        return f"{self.instance_id}_{self.repetition_id}"


@dataclass
class BranchedTrajectoryID(TrajectoryID):
    """Extended TrajectoryID for branched trajectories.

    Tracks the lineage of a branched trajectory back to its source.
    For root trajectories, source_repetition_id is None.
    """

    source_repetition_id: Optional[int] = None  # Parent's repetition_id (None for root)
    branch_turn: Optional[int] = None  # Which turn we branched from (0-indexed)
    branch_token_idx: Optional[int] = None  # Token index within response_ids where branch occurred

    def to_string(self) -> str:
        base = f"{self.instance_id}_{self.repetition_id}"
        if self.source_repetition_id is not None:
            return f"{base}_from_{self.source_repetition_id}_t{self.branch_turn}_tok{self.branch_token_idx}"
        return base

    def is_root(self) -> bool:
        """Returns True if this is a root trajectory (not branched)."""
        return self.source_repetition_id is None


@dataclass
class BatchMetadata:
    global_step: int
    training_phase: TrainingPhase


class GeneratorInput(TypedDict):
    prompts: List[ConversationType]
    env_classes: List[str]
    env_extras: Optional[List[Dict[str, Any]]]
    sampling_params: Optional[Dict[str, Any]]
    trajectory_ids: Optional[List[TrajectoryID]]
    batch_metadata: Optional[BatchMetadata]


class GeneratorOutput(TypedDict):
    prompt_token_ids: List[List[int]]
    response_ids: List[List[int]]
    rewards: Union[List[float], List[List[float]]]
    loss_masks: List[List[int]]
    stop_reasons: Optional[List[str]]
    rollout_metrics: Optional[Dict[str, Any]]
    rollout_logprobs: Optional[List[List[float]]]
    trajectory_ids: Optional[List[TrajectoryID]]
    # Applicable only for step-wise training
    is_last_step: Optional[List[bool]]
    # Per-step timing with token indices: [{start_ix, end_ix, type, time_elapsed_s}, ...]
    steps: Optional[List[List[Dict[str, Any]]]]
    # Environment metrics from each trajectory (e.g., prediction, ground_truth, num_search_calls)
    env_metrics: Optional[List[Dict[str, Any]]]


class GeneratorInterface(ABC):
    @abstractmethod
    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        """Generate trajectories for the input batch.

        Returns outputs in the same order as the input batch.

        Args:
            input_batch (GeneratorInput): Input batch
        Returns:
            GeneratorOutput: Generated trajectories
        """
        raise NotImplementedError()
