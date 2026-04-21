#!/bin/bash
# Production: Qwen2.5-VL-7B-Instruct GRPO, 1-step direct answer (no search).
# FULL-NODE on ailab: 8 x H200 + 1300G RAM + 64 CPU cores, 24h wall.
# Weights-only checkpoints. Batch-size fallback 256 -> 128 -> 64 on OOM.
#
# Prereq (one-time on login node):
#   .venv/bin/python examples/train/search_arxiv/searchr1_arxiv_vl_dataset.py \
#       --output_dir /scratch/gpfs/ZHUANGL/sk7524/SkyRLMain/data/iclr_arxiv_vl \
#       --splits train validation test
#
#SBATCH --job-name=tp_vl_7b_1step_grpo
#SBATCH --partition=ailab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=1300G
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_prediction/vl_1step_%j.out
#SBATCH --error=logs/train_prediction/vl_1step_%j.err

set -uo pipefail  # not -e: we want to handle failures via the retry loop

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$WORKDIR"
mkdir -p logs/train_prediction

SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/gpfs/ZHUANGL/sk7524}"
export HF_HOME="${HF_HOME:-${SCRATCH_ROOT}/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${SCRATCH_ROOT}/hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${SCRATCH_ROOT}/hf/datasets}"
export TORCH_HOME="${TORCH_HOME:-${SCRATCH_ROOT}/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SCRATCH_ROOT}/.cache}"
export WANDB_DIR="${WANDB_DIR:-${SCRATCH_ROOT}/wandb}"
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export LOGURU_LEVEL=INFO
export _SKYRL_USE_NEW_INFERENCE=1   # required by SkyRLVLMGymGenerator
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$WANDB_DIR"

NUM_GPUS="${NUM_GPUS:-8}"
DATA_DIR="${DATA_DIR_VL:-${SCRATCH_ROOT}/SkyRLMain/data/iclr_arxiv_vl}"
MODEL_PATH="${MODEL_PATH:-${SCRATCH_ROOT}/hf/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5}"
RUN_NAME="${RUN_NAME:-tp_vl_7b_1step_grpo_${SLURM_JOB_ID}}"
RUN_LOG_DIR="${SCRATCH_ROOT}/skyrl-logs/${RUN_NAME}"
mkdir -p "$RUN_LOG_DIR"

echo "=============================================="
echo "Job:         $SLURM_JOB_ID"
echo "Host:        $(hostname)"
echo "Data:        $DATA_DIR"
echo "Model:       $MODEL_PATH"
echo "Run name:    $RUN_NAME"
echo "NUM_GPUS:    $NUM_GPUS"
echo "=============================================="
nvidia-smi

# TODO(prompt-length): keep in sync with the half-node variant.
# 26480 = LLaMA-Factory filtered24480 cutoff + ~2k for our system prompt.
MAX_PROMPT_LEN=26480
MAX_INPUT_LEN=$MAX_PROMPT_LEN
echo "Prompt length budget: max_prompt_length=$MAX_PROMPT_LEN, max_input_length=$MAX_INPUT_LEN"

source .venv/bin/activate

# Batch-size fallback: if BS=256 OOMs we retry at 128, then 64. Mini-batch
# scales with train batch (mini = bz/2). n_samples_per_prompt stays at 8
# (GRPO needs >= 2 samples for variance estimation; 8 is the sweet spot).
# Non-memory failures do NOT retry (infinite-loop hazard on config bugs).
BS_LIST=(256 128 64)
MINI_LIST=(128 64 32)

for i in "${!BS_LIST[@]}"; do
    BS="${BS_LIST[$i]}"
    MINI="${MINI_LIST[$i]}"
    ATTEMPT_LOG="${RUN_LOG_DIR}/attempt_${i}_bs${BS}.log"
    echo ""
    echo "================================================"
    echo "=== Attempt $((i + 1)) / ${#BS_LIST[@]}: BS=$BS MINI=$MINI"
    echo "=== Log: $ATTEMPT_LOG"
    echo "================================================"
    : > "$ATTEMPT_LOG"

    python -m skyrl.train.entrypoints.main_base \
      data.train_data="['${DATA_DIR}/train.parquet']" \
      data.val_data="['${DATA_DIR}/validation.parquet']" \
      trainer.algorithm.advantage_estimator=grpo \
      trainer.algorithm.eps_clip_low=0.2 \
      trainer.algorithm.eps_clip_high=0.28 \
      trainer.algorithm.use_kl_loss=true \
      trainer.algorithm.kl_loss_coef=0.001 \
      trainer.policy.optimizer_config.lr=1.0e-6 \
      trainer.policy.optimizer_config.max_grad_norm=0.5 \
      trainer.policy.optimizer_config.num_warmup_steps=4 \
      trainer.policy.optimizer_config.scheduler=constant_with_warmup \
      trainer.policy.model.path="$MODEL_PATH" \
      trainer.placement.colocate_all=true \
      trainer.strategy=fsdp2 \
      trainer.policy.fsdp_config.cpu_offload=true \
      trainer.ref.fsdp_config.cpu_offload=true \
      trainer.gradient_checkpointing=true \
      trainer.use_liger_kernel=true \
      trainer.use_sample_packing=false \
      trainer.save_optimizer_state=false \
      trainer.placement.policy_num_gpus_per_node=${NUM_GPUS} \
      trainer.placement.ref_num_gpus_per_node=${NUM_GPUS} \
      generator.vision_language_generator=true \
      generator.inference_engine.num_engines=${NUM_GPUS} \
      generator.inference_engine.tensor_parallel_size=1 \
      generator.inference_engine.backend=vllm \
      generator.inference_engine.run_engines_locally=true \
      generator.inference_engine.weight_sync_backend=nccl \
      generator.inference_engine.gpu_memory_utilization=0.7 \
      generator.inference_engine.async_engine=true \
      generator.inference_engine.engine_init_kwargs.allowed_local_media_path=${SCRATCH_ROOT}/LLaMA-Factory-AutoReviewer \
      generator.inference_engine.engine_init_kwargs.disable_mm_preprocessor_cache=true \
      trainer.epochs=1 \
      trainer.update_epochs_per_batch=1 \
      trainer.train_batch_size=${BS} \
      trainer.policy_mini_batch_size=${MINI} \
      trainer.micro_forward_batch_size_per_gpu=1 \
      trainer.micro_train_batch_size_per_gpu=1 \
      trainer.max_prompt_length=${MAX_PROMPT_LEN} \
      generator.max_input_length=${MAX_INPUT_LEN} \
      generator.sampling_params.max_generate_length=4096 \
      generator.batched=false \
      generator.use_conversation_multi_turn=true \
      generator.append_eos_token_after_stop_str_in_multi_turn=true \
      generator.n_samples_per_prompt=8 \
      generator.max_turns=1 \
      generator.sampling_params.temperature=1.0 \
      generator.sampling_params.top_p=1.0 \
      generator.sampling_params.stop='["</answer>"]' \
      environment.env_class=search_arxiv \
      environment.skyrl_gym.max_env_workers=16 \
      environment.skyrl_gym.search_arxiv.max_turns=1 \
      environment.skyrl_gym.search_arxiv.search_enabled=false \
      trainer.logger=console \
      trainer.project_name=train-prediction \
      trainer.run_name="$RUN_NAME" \
      trainer.ckpt_interval=5 \
      trainer.hf_save_interval=20 \
      trainer.max_ckpts_to_keep=2 \
      trainer.resume_mode=latest \
      trainer.ckpt_path="${SCRATCH_ROOT}/ckpts/${RUN_NAME}" \
      trainer.log_path="${SCRATCH_ROOT}/skyrl-logs" \
      trainer.eval_batch_size=${BS} \
      trainer.eval_before_train=false \
      trainer.eval_interval=20 \
      generator.eval_sampling_params.temperature=1.0 \
      generator.eval_sampling_params.top_p=1.0 \
      generator.eval_sampling_params.stop='["</answer>"]' \
      generator.eval_sampling_params.max_generate_length=4096 \
      trainer.export_path="${SCRATCH_ROOT}/ckpts/${RUN_NAME}/exports" \
      "$@" 2>&1 | tee "$ATTEMPT_LOG"
    EXIT="${PIPESTATUS[0]}"

    if [ "$EXIT" -eq 0 ]; then
        echo "=== Training COMPLETED at BS=$BS"
        exit 0
    fi

    if grep -qE "OutOfMemoryError|CUDA out of memory|oom-kill|OUT_OF_MEMORY|DefaultCPUAllocator" "$ATTEMPT_LOG"; then
        echo "=== Memory failure detected at BS=$BS (exit=$EXIT) -- falling back"
        continue
    fi

    echo "=== Non-memory failure at BS=$BS (exit=$EXIT) -- not retrying"
    exit "$EXIT"
done

echo "=== All batch-size fallbacks exhausted"
exit 1
