"""
Main entrypoint for evaluation-only.

Mirrors the training path's weight-sync dance at the top of ``trainer.train()``
(see skyrl/train/trainer.py:181-196) so that eval-only works under
``trainer.placement.colocate_all=true`` without running gibberish. Before the
fix, this entrypoint only called ``inference_engine_client.wake_up()``, which
with sleep level=2 allocates empty GPU buffers but never restores weights —
vLLM would then generate from randomised values. We now:

  1. Build the full trainer (``_setup_trainer()``) to get policy FSDP workers
     plus dispatch / inference_engine_client / generator, just like training.
  2. Initialize the NCCL communicator between policy FSDP and vLLM
     (``trainer.init_weight_sync_state()``).
  3. Broadcast fresh policy weights FSDP → vLLM
     (``trainer.dispatch.save_weights_for_sampler()``). For colocate_all=true
     this also wakes the weight region, then the KV-cache region.
  4. Defensively reset the prefix cache (vllm issue #17103) under colocate_all.
  5. Run the eval loop against the trainer's generator.
"""

import asyncio
import sys
from typing import Any

import ray
from loguru import logger

from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import (
    BasePPOExp,
)
from skyrl.train.evaluate import evaluate
from skyrl.train.utils.trainer_utils import build_dataloader
from skyrl.train.utils.utils import initialize_ray, validate_generator_cfg


class EvalOnlyEntrypoint(BasePPOExp):
    def get_train_dataset(self):
        """Override to avoid requiring a train dataset for eval-only runs."""
        return None

    def run_sync(self) -> dict[str, Any]:
        """Entrypoint. Builds the trainer synchronously (``_setup_trainer`` calls
        ``get_inference_client`` which does its own ``asyncio.run`` for
        colocate-mode sleep setup, so we must stay outside an outer
        ``asyncio.run`` until after that returns), then delegates the async
        weight-sync + eval loop to ``_run_async``."""
        trainer = self._setup_trainer()
        return asyncio.run(self._run_async(trainer))

    async def _run_async(self, trainer) -> dict[str, Any]:
        assert self.eval_dataset is not None, "The evaluation only entrypoint requires an eval dataset"

        # 1. NCCL communicator between policy FSDP and vLLM.
        trainer.init_weight_sync_state()

        # 2. Broadcast FSDP weights to vLLM. In colocate_all=true this also
        #    handles wake_up(weights) -> broadcast -> finish -> wake_up(kv_cache).
        #    In colocate_all=false it pauses/resumes generation around the
        #    broadcast (vLLM was never slept, weights are already loaded from
        #    disk; this just ensures FSDP and vLLM agree).
        await trainer.dispatch.save_weights_for_sampler()

        # 3. Prefix cache can contain placeholder entries from the pre-broadcast
        #    wake; clear it defensively under colocate_all=true (vllm #17103).
        if self.cfg.trainer.placement.colocate_all:
            await trainer.inference_engine_client.reset_prefix_cache()

        # 4. Run the eval loop.
        results: dict[str, Any] = await evaluate(
            eval_dataloader=build_dataloader(self.cfg, self.eval_dataset, is_train=False),
            generator=trainer.generator,
            cfg=self.cfg,
            global_step=None,
            tokenizer=self.tokenizer,
        )

        tracker = self.get_tracker()
        tracker.log(results, step=0, commit=True)

        return results


@ray.remote(num_cpus=1)
def eval_entrypoint(cfg: SkyRLTrainConfig) -> dict:
    exp = EvalOnlyEntrypoint(cfg)
    return exp.run_sync()


def main() -> None:
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_generator_cfg(cfg)
    initialize_ray(cfg)
    metrics = ray.get(eval_entrypoint.remote(cfg))
    logger.info(f"Metrics from eval only run: {metrics}")


if __name__ == "__main__":
    main()
