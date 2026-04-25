"""Cosine-decayed gamma curriculum sampler for difficulty-mixed training.

Used by the train-prediction RL setup to bias early batches toward "easy"
prompts (clear-Accept / clear-Reject from the dataset's `extra_info.is_easy`
tag) and gradually introduce borderline prompts as training proceeds.

Usage flow:
  1. dataloader.sampler is a `CurriculumSampler` constructed with the
     `easy_indices` (rows with extra_info.is_easy=True), `full_indices`
     (all rows), and `batch_size` (the train-batch-size in prompts).
  2. Trainer calls `sampler.set_step(global_step)` before each batch.
  3. For each batch of `batch_size` slots, the sampler computes
     `n_easy = round(batch_size * gamma(step))` and samples that many distinct
     indices from `easy_indices` without replacement, then samples
     `batch_size - n_easy` distinct indices from `full_indices \\ easy_picks`
     (also without replacement). Indices are unique within each batch so the
     downstream trainer's per-batch uniqueness assertion holds.

`gamma(step)` follows a cosine half-cycle from `gamma_start` at step 0 to
`gamma_end` at `decay_steps`. After `decay_steps`, gamma stays at `gamma_end`.
"""

import math
import random
from typing import Iterator, Sequence

from torch.utils.data import Sampler


class CurriculumSampler(Sampler[int]):
    def __init__(
        self,
        full_indices: Sequence[int],
        easy_indices: Sequence[int],
        num_samples: int,
        batch_size: int,
        gamma_start: float = 1.0,
        gamma_end: float = 0.0,
        decay_steps: int = 50,
        seed: int = 42,
    ) -> None:
        if not 0.0 <= gamma_start <= 1.0:
            raise ValueError(f"gamma_start must be in [0, 1], got {gamma_start}")
        if not 0.0 <= gamma_end <= 1.0:
            raise ValueError(f"gamma_end must be in [0, 1], got {gamma_end}")
        if decay_steps < 1:
            raise ValueError(f"decay_steps must be >= 1, got {decay_steps}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        self._full = list(full_indices)
        self._easy = list(easy_indices)
        if not self._full:
            raise ValueError("full_indices must be non-empty")
        if not self._easy:
            # Degenerate: no easy rows tagged. Fall back to uniform from full.
            # The sampler still works (gamma path simply never triggers).
            self._easy = list(self._full)

        self._num_samples = int(num_samples)
        self._batch_size = int(batch_size)
        self.gamma_start = float(gamma_start)
        self.gamma_end = float(gamma_end)
        self.decay_steps = int(decay_steps)
        self._current_step = 0
        self._rng = random.Random(int(seed))

    def set_step(self, step: int) -> None:
        """Trainer hook: update the schedule cursor before each batch dispatch."""
        self._current_step = max(0, int(step))

    def gamma(self) -> float:
        """Current sampling probability for the easy pool. Cosine half-cycle."""
        t = min(self._current_step / self.decay_steps, 1.0)
        return self.gamma_end + 0.5 * (self.gamma_start - self.gamma_end) * (1.0 + math.cos(math.pi * t))

    def _sample_batch(self, batch_size: int) -> list[int]:
        """Sample `batch_size` distinct indices for one batch, biased by gamma."""
        g = self.gamma()
        n_easy_target = int(round(batch_size * g))
        # Cap by the easy pool size; rest comes from full.
        n_easy = min(n_easy_target, len(self._easy), batch_size)
        easy_picks = self._rng.sample(self._easy, n_easy) if n_easy > 0 else []

        n_full = batch_size - n_easy
        if n_full > 0:
            picked = set(easy_picks)
            avail_full = [i for i in self._full if i not in picked]
            # Fall back: if avail_full < n_full (small dataset), sample with
            # replacement from full as a last resort. In practice batch_size
            # is much smaller than len(full) so this never triggers.
            if len(avail_full) >= n_full:
                full_picks = self._rng.sample(avail_full, n_full)
            else:
                full_picks = list(avail_full) + self._rng.choices(self._full, k=n_full - len(avail_full))
        else:
            full_picks = []

        batch = easy_picks + full_picks
        self._rng.shuffle(batch)
        return batch

    def __iter__(self) -> Iterator[int]:
        remaining = self._num_samples
        while remaining > 0:
            this_batch = min(self._batch_size, remaining)
            for idx in self._sample_batch(this_batch):
                yield idx
            remaining -= this_batch

    def __len__(self) -> int:
        return self._num_samples
