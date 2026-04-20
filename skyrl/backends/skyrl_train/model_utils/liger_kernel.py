"""Liger Kernel dispatch.

Applies liger-kernel's fused kernels (RMSNorm / RoPE / SwiGLU / etc.) to a
loaded HF model by model family, based on ``config.model_type``.

For RL training (``require_logits=True``), fused_linear_cross_entropy is
disabled so logits remain materialized for the policy-gradient loss.

Families covered include text LLMs and multi-modal (VL) model families that
liger-kernel currently supports (qwen2_5_vl, qwen3_vl, internvl, llava, ...).
"""

import inspect
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from transformers import PretrainedConfig


# model_type (from HF config) → name of the apply_liger_kernel_to_<X> function
# exposed by `liger_kernel.transformers`. Lazy-imported per model_type.
LIGER_FAMILIES = {
    # Text LLMs
    "llama": "apply_liger_kernel_to_llama",
    "llama4": "apply_liger_kernel_to_llama4",
    "mistral": "apply_liger_kernel_to_mistral",
    "mixtral": "apply_liger_kernel_to_mixtral",
    "phi3": "apply_liger_kernel_to_phi3",
    "gemma": "apply_liger_kernel_to_gemma",
    "gemma2": "apply_liger_kernel_to_gemma2",
    "gemma3_text": "apply_liger_kernel_to_gemma3_text",
    "qwen2": "apply_liger_kernel_to_qwen2",
    "qwen3": "apply_liger_kernel_to_qwen3",
    "qwen3_moe": "apply_liger_kernel_to_qwen3_moe",
    "glm4": "apply_liger_kernel_to_glm4",
    "olmo2": "apply_liger_kernel_to_olmo2",
    "olmo3": "apply_liger_kernel_to_olmo3",
    "granite": "apply_liger_kernel_to_granite",
    "gpt_oss": "apply_liger_kernel_to_gpt_oss",
    # Multi-modal / VL families
    "qwen2_vl": "apply_liger_kernel_to_qwen2_vl",
    "qwen2_5_vl": "apply_liger_kernel_to_qwen2_5_vl",
    "qwen3_vl": "apply_liger_kernel_to_qwen3_vl",
    "qwen3_vl_moe": "apply_liger_kernel_to_qwen3_vl_moe",
    "gemma3": "apply_liger_kernel_to_gemma3",
    "glm4v": "apply_liger_kernel_to_glm4v",
    "glm4v_moe": "apply_liger_kernel_to_glm4v_moe",
    "internvl": "apply_liger_kernel_to_internvl",
    "llava": "apply_liger_kernel_to_llava",
    "mllama": "apply_liger_kernel_to_mllama",
    "paligemma": "apply_liger_kernel_to_paligemma",
    "smolvlm": "apply_liger_kernel_to_smolvlm",
}


def apply_liger_kernel(
    config: "PretrainedConfig",
    is_trainable: bool,
    require_logits: bool = True,
) -> None:
    """Apply liger-kernel optimizations based on ``config.model_type``.

    Must be called BEFORE model loading (after loading the HF config).

    Args:
        config: Model config from ``AutoConfig.from_pretrained(...)``.
        is_trainable: Whether the model will be trained. Inference-only models
            skip the patch (no gain for forward-only workloads in this pipeline).
        require_logits: RL training requires logits for the policy gradient.
            When True and the selected apply_fn accepts ``fused_linear_cross_entropy``,
            disable that fusion so logits remain materialized.
    """
    if not is_trainable:
        return

    model_type = getattr(config, "model_type", None)
    if model_type is None:
        logger.warning("Liger: config.model_type is None; skipping.")
        return

    apply_fn_name = LIGER_FAMILIES.get(model_type)
    if apply_fn_name is None:
        logger.warning(
            f"Liger: model_type='{model_type}' not in the dispatch table; skipping. "
            f"Add it to LIGER_FAMILIES if liger-kernel supports it."
        )
        return

    try:
        apply_fn = getattr(
            __import__("liger_kernel.transformers", fromlist=[apply_fn_name]),
            apply_fn_name,
        )
    except (ImportError, AttributeError) as e:
        logger.warning(
            f"Liger: failed to import {apply_fn_name} from liger_kernel.transformers "
            f"(model_type='{model_type}'): {e}. Skipping."
        )
        return

    # For RL, we need logits for the policy gradient — disable fused linear CE
    # if the apply_fn supports it. Text and VL variants may differ; inspect at runtime.
    kwargs = {}
    if require_logits and "fused_linear_cross_entropy" in inspect.signature(apply_fn).parameters:
        logger.info(
            f"Liger: RL training requires logits — disabling fused_linear_cross_entropy "
            f"for model_type='{model_type}'."
        )
        kwargs = {"fused_linear_cross_entropy": False, "cross_entropy": True}

    apply_fn(**kwargs)
    logger.info(f"Liger kernel applied for model_type='{model_type}'")
