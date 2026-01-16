"""Liger Kernel support following LlamaFactory pattern."""

import inspect
from typing import TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from transformers import PretrainedConfig


def apply_liger_kernel(
    config: "PretrainedConfig",
    is_trainable: bool,
    require_logits: bool = True,  # Always True for RL
) -> None:
    """
    Apply Liger kernel optimizations based on model type.

    Must be called BEFORE model loading (after loading config).

    Args:
        config: Model config from AutoConfig.from_pretrained()
        is_trainable: Whether model will be trained
        require_logits: Whether logits are needed (True for RL, disables fused CE)
    """
    if not is_trainable:
        return

    model_type = getattr(config, "model_type", None)

    # Map model_type to liger kernel apply function
    if model_type == "gemma":
        from liger_kernel.transformers import apply_liger_kernel_to_gemma as apply_fn
    elif model_type == "gemma2":
        from liger_kernel.transformers import apply_liger_kernel_to_gemma2 as apply_fn
    elif model_type == "llama":
        from liger_kernel.transformers import apply_liger_kernel_to_llama as apply_fn
    elif model_type == "mistral":
        from liger_kernel.transformers import apply_liger_kernel_to_mistral as apply_fn
    elif model_type == "mixtral":
        from liger_kernel.transformers import apply_liger_kernel_to_mixtral as apply_fn
    elif model_type == "phi3":
        from liger_kernel.transformers import apply_liger_kernel_to_phi3 as apply_fn
    elif model_type == "qwen2":
        from liger_kernel.transformers import apply_liger_kernel_to_qwen2 as apply_fn
    elif model_type == "qwen3":
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3 as apply_fn
    elif model_type == "qwen3_moe":
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3_moe as apply_fn
    elif model_type == "gpt_oss":
        try:
            from liger_kernel.transformers import apply_liger_kernel_to_gpt_oss as apply_fn
        except ImportError:
            logger.warning("Liger kernel for gpt_oss requires custom liger-kernel installation.")
            return
    else:
        logger.warning(f"Model type '{model_type}' does not support liger kernel. Skipping.")
        return

    # For RL, we need logits for policy gradient, so disable fused_linear_cross_entropy
    kwargs = {}
    if require_logits and "fused_linear_cross_entropy" in inspect.signature(apply_fn).parameters:
        logger.info("RL training requires logits - disabling fused_linear_cross_entropy.")
        kwargs = {"fused_linear_cross_entropy": False, "cross_entropy": True}

    apply_fn(**kwargs)
    logger.info(f"Liger kernel applied for model_type='{model_type}'")
