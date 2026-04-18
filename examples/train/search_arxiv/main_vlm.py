"""
VLM entrypoint for search_arxiv training. Swaps the default SkyRLGymGenerator
for SkyRLVLMGymGenerator so image observations flow through the multi-modal
path.
"""

import sys

import ray

from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp, validate_cfg
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
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
