"""
VLM entrypoint for search_arxiv training. Swaps the default SkyRLGymGenerator
for SkyRLVLMGymGenerator so image observations flow through the multi-modal
path.

NOTE: as of the `vision_language_generator` config flag, the equivalent runtime
behavior is achievable by calling main_base.py with
``generator.vision_language_generator=true`` -- BasePPOExp.get_generator now
dispatches on that flag. This module is kept for backward compatibility with
the working VL smoke sbatch; new sbatches can prefer the main_base.py route.
"""

import sys

import ray

from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp, _apply_slurm_job_overrides, validate_cfg
from skyrl.train.utils import initialize_ray


class VLMPPOExp(BasePPOExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        from skyrl.train.generators.skyrl_vlm_generator import SkyRLVLMGymGenerator

        return SkyRLVLMGymGenerator(
            generator_cfg=cfg.generator,
            skyrl_gym_cfg=cfg.environment.skyrl_gym,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
        )


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: SkyRLTrainConfig):
    exp = VLMPPOExp(cfg)
    exp.run()


def main() -> None:
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    _apply_slurm_job_overrides(cfg)
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
