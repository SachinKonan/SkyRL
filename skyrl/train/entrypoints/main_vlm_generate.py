"""
VL eval-only entrypoint.

Same shape as ``main_generate.EvalOnlyEntrypoint`` (FSDP→vLLM weight sync
before the eval loop, defensive prefix-cache reset under colocate_all),
but constructs ``SkyRLVLMGymGenerator`` instead of the text generator so
image tokens flow through the multi-modal path.
"""

import sys
from typing import Any

import ray
from loguru import logger

from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_generate import EvalOnlyEntrypoint
from skyrl.train.utils.utils import initialize_ray, validate_generator_cfg


class VLMEvalOnlyEntrypoint(EvalOnlyEntrypoint):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        from skyrl.train.generators.skyrl_vlm_generator import SkyRLVLMGymGenerator

        return SkyRLVLMGymGenerator(
            generator_cfg=cfg.generator,
            skyrl_gym_cfg=cfg.environment.skyrl_gym,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
        )


@ray.remote(num_cpus=1)
def vlm_eval_entrypoint(cfg: SkyRLTrainConfig) -> dict:
    exp = VLMEvalOnlyEntrypoint(cfg)
    return exp.run_sync()


def main() -> None:
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_generator_cfg(cfg)
    initialize_ray(cfg)
    metrics = ray.get(vlm_eval_entrypoint.remote(cfg))
    logger.info(f"Metrics from VL eval-only run: {metrics}")


if __name__ == "__main__":
    main()
