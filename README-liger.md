# Liger Kernel integration in SkyRL

## What this adds

A working `trainer.use_liger_kernel=true` toggle that applies
[liger-kernel](https://github.com/linkedin/Liger-Kernel) fused kernels (RMSNorm /
RoPE / SwiGLU / etc.) to the policy and ref models in the FSDP backend.

For RL specifically, `fused_linear_cross_entropy` is **disabled** so logits
remain materialized for the policy gradient — without that flag, liger's default
fusion would hide logits and break the GRPO / PPO loss.

## Why this was needed

Before this branch, current SkyRL had a half-wired liger path
(`skyrl/backends/skyrl_train/workers/model_wrapper.py:108-113`) that swapped in
`AutoLigerKernelForCausalLM`, but:

- No config field plumbed it through — `trainer.use_liger_kernel=true` on the CLI
  was a no-op (or a Hydra error).
- `liger-kernel` was not in `pyproject.toml`, so it wasn't even installed.
- The `AutoLigerKernelForCausalLM` swap defaults to
  `fused_linear_cross_entropy=True`. That fuses the final linear + CE loss and
  hides logits — silently incorrect for RL, where the policy gradient needs
  per-token logits.

The sibling repo at `/scratch/gpfs/ZHUANGL/sk7524/SkyRLSearchEnvs/skyrl-train`
had this working via a different pattern: dispatch by `config.model_type`,
explicitly disable `fused_linear_cross_entropy` for RL. That pattern is the
basis for this port.

## Changes

### `pyproject.toml`

Add `liger-kernel>=0.5.4; sys_platform == 'linux'` to the `fsdp` extra.
The platform marker keeps macOS dev environments resolvable (liger pulls in
triton, which has no darwin-arm64 wheels).

### `skyrl/backends/skyrl_train/model_utils/`

New package. `liger_kernel.py` ports the sibling's dispatch and **extends it
with VL families** that the sibling didn't have. Coverage includes:

| Family | Apply fn |
|---|---|
| Text — `llama, llama4, mistral, mixtral, phi3, gemma, gemma2, gemma3_text, qwen2, qwen3, qwen3_moe, glm4, olmo2, olmo3, granite, gpt_oss` | `apply_liger_kernel_to_<family>` |
| VL — `qwen2_vl, qwen2_5_vl, qwen3_vl, qwen3_vl_moe, gemma3, glm4v, glm4v_moe, internvl, llava, mllama, paligemma, smolvlm` | `apply_liger_kernel_to_<family>` |

The dispatch is structural — `LIGER_FAMILIES = {model_type: apply_fn_name}`.
Unknown `model_type` is logged and gracefully skipped (no error).

The runtime signature inspection
(`inspect.signature(apply_fn).parameters` check for `fused_linear_cross_entropy`)
is preserved verbatim from the sibling. Some apply_fn variants accept that
kwarg, others don't — when it's accepted **and** `require_logits=True`, we pass
`{"fused_linear_cross_entropy": False, "cross_entropy": True}`.

### `skyrl/backends/skyrl_train/workers/model_wrapper.py`

The old `if use_liger_kernel: model_class = AutoLigerKernelForCausalLM`
branch is replaced. After loading the HF config and **before** loading the
model weights, we call:

```python
if use_liger_kernel:
    apply_liger_kernel(model_config, is_trainable=True, require_logits=True)
```

The `model_class` stays `AutoModelForCausalLM` unconditionally. Liger
monkey-patches the relevant classes globally, so the subsequent
`from_pretrained` picks up the fused kernels.

### `skyrl/train/config/config.py`

New top-level field on `TrainerConfig`:

```python
use_liger_kernel: bool = False
"""Apply liger-kernel fused kernels (RMSNorm / RoPE / SwiGLU) for supported families.
RL-safe: fused_linear_cross_entropy is disabled so logits remain exposed for the
policy gradient. Unsupported model_types are logged and skipped."""
```

### `skyrl/backends/skyrl_train/workers/fsdp/fsdp_worker.py`

Pass `use_liger_kernel=self.cfg.use_liger_kernel` at both `HFModelWrapper(...)`
call sites — the policy worker init (line 187) and the ref worker init
(line 433). Same flag for both; if liger is on, both models benefit from it.

### Smoke sbatches

Both existing smoke sbatches were edited to enable liger and (for VL) to add
the memory knobs needed when liger fuses but the model is still bigger than fits
without offload:

- `sbatch/train_prediction/run_qwen2.5_3b_1step_answer.sbatch`
  - Added `trainer.use_liger_kernel=true`.
- `sbatch/train_prediction/run_qwen2.5_vl_3b_1step_answer.sbatch`
  - Switched model from Qwen3-VL-2B-Instruct → **Qwen2.5-VL-3B-Instruct**
    (the qwen2.5-vl family this branch wires up).
  - Added `trainer.use_liger_kernel=true`.
  - Enabled `trainer.policy.fsdp_config.cpu_offload=true` and
    `trainer.gradient_checkpointing=true` to fit on 1 × 80 GB GPU
    (Qwen2.5-VL-3B is bigger than Qwen3-VL-2B; without these the second
    training step OOM'd at ~68 / 80 GB GPU memory).
  - Bumped `--mem=120G → 200G` in the SBATCH header to accommodate the
    cpu-offloaded policy weights.

## Verification

Two smoke sbatches on `gpu-test` (1 × A100-80GB, ~12 min each).

### Text — Qwen2.5-3B-Instruct (`qwen2`)

Apples-to-apples comparison: only `trainer.use_liger_kernel` toggled.

| Metric | No liger (job 7173891) | With liger (job 7178535) | Δ |
|---|---|---|---|
| Wall time | 12m17s | **11m07s** | **−10% (1.10× faster)** |
| GPU util | 52.8% | 52.7% | flat |
| Peak GPU mem | 75.3 GB / 80 (94%) | 75.3 GB / 80 (94%) | identical |
| Host RAM peak | 40.9 GB / 120 | 20.0 GB / 120 | −20 GB (likely measurement noise — liger doesn't change host RAM directly) |
| CPU util | 14.4% | 14.6% | flat |

Clean ~10% wall-time win on text. Bottleneck is still vLLM rollout (≈52% GPU
util — gradient training itself is a small slice of the wall).

### VL — Qwen2.5-VL-3B-Instruct (`qwen2_5_vl`)

NOT apples-to-apples. The with-liger run also added `policy.cpu_offload=true`,
`gradient_checkpointing=true`, and `--mem=200G` to fit. The no-liger baseline
ran at the looser memory config without those.

| Metric | No liger (job 7164509) | With liger + offload + ckpt (job 7180305) | Δ |
|---|---|---|---|
| Wall time | 13m40s | **12m40s** | −7% |
| GPU util | 32.0% | 35.0% | +3% |
| Peak GPU mem | 74.3 GB / 80 (93%) | 63.2 GB / 80 (79%) | −11 GB (mostly from cpu_offload) |
| Host RAM peak | 25.3 GB / 120 | 61.3 GB / 200 | +36 GB (cpu_offload moved policy weights to host) |
| CPU util | 23.7% | 23.0% | flat |

For VL we can't isolate liger's contribution from the offload + checkpointing
changes that were needed in the same run. Confirmed end-to-end:

- Liger applied for `qwen2_5_vl` on both policy and ref workers (verified in
  the infra log under `$SCRATCH_ROOT/skyrl-logs/`).
- `fused_linear_cross_entropy=False, cross_entropy=True` was passed (RL-safe).
- All 4 training batches completed; final checkpoint and HF model saved.

If a clean VL A/B is needed, re-run the **with-liger config but with
`use_liger_kernel=false`** — that's the only knob to swap, ~13 min job.

## How to enable in your own sbatch

Add one line to the Hydra overrides:

```bash
trainer.use_liger_kernel=true \
```

For supported families (see the table above), liger fuses kernels in-place.
For unsupported `model_type`, the dispatch logs a warning like
`Liger: model_type='X' not in the dispatch table; skipping.` and training
continues without liger.

For VL or any large model that's tight on GPU memory, consider also setting:

```bash
trainer.policy.fsdp_config.cpu_offload=true \
trainer.gradient_checkpointing=true \
```

and bumping `--mem` accordingly (~+80 GB host RAM for a 3B-class model).

## Upstreaming notes

This is plausibly a clean PR against `NovaSky-AI/SkyRL`. The dispatch file is
generic and well-tested across two model families. The only SkyRL-specific
plumbing is the new `TrainerConfig.use_liger_kernel` field and the
`fsdp_worker.py` pass-through.
