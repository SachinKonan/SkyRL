#!/bin/bash
# SFT cold-start on Qwen2.5-7B-Instruct, run as a SLURM job array on gpu-test
# with %1 concurrency. Each task is a 55-min training slot that resumes from
# the previous task's FSDP checkpoint. The array runs serially until the
# trainer hits NUM_STEPS (the trainer's own loop will break early once
# global_step >= num_steps, which means later array tasks no-op quickly).
#
# Why an array on gpu-test instead of a single pli run:
#   - pli queue can be 18+ hours behind. gpu-test is usually <10 min.
#   - We pay 11-12 min/slot for tokenization+init overhead, but each cell
#     scheduled fast, and we can chain serially.
#
# Topology per cell:
#   1 node × 4× A100 80GB on gpu-test, FSDP2 sharded.
#
# Per-cell budget (60 min wall, 55 min training cap):
#   ~12 min tokenization + Ray init + model load
#   ~1 min final FSDP ckpt save
#   ~42 min actual training -> ~30 steps at 80-90s/step
#   412 total steps -> ~14 cells -> ~14h wall (still serial; %1 ensures one at a time)
#
# Resume protocol:
#   - First cell: resume_from=latest reads latest_ckpt_global_step.txt -> not found
#     -> trainer starts at step 0 cleanly.
#   - Subsequent cells: pointer exists -> trainer loads optimizer + scheduler +
#     model from global_step_N FSDP shards.
#   - Trainer's outer loop checks `global_step >= num_steps` so once we've
#     hit 412, remaining cells exit fast (no actual training).
#
# Checkpointing:
#   - save_fsdp_checkpoint=true: required for resume (model + optim + scheduler).
#   - ckpt_interval=10: every ~13 min of training, bounded loss on hard SIGKILL.
#   - hf_save_interval=206: HF safetensors at epoch boundaries for vLLM/GRPO init.
#   - max_ckpts_to_keep=3: keep only last 3 FSDP ckpts (~84 GB on disk).
#
# Manual resubmission (e.g., to extend past 15 cells):
#   sbatch --array=16-25%1 sbatch/train_prediction/sft_qwen2.5_7b_array_gputest.sbatch
#
#SBATCH --job-name=sft_7b_array
#SBATCH --partition=gpu-test
#SBATCH --array=1-15%1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=300G
#SBATCH --gres=gpu:4
#SBATCH --constraint="gpu80"
#SBATCH --time=00:55:00
#SBATCH --output=logs/sft/array_%A_%a.out
#SBATCH --error=logs/sft/array_%A_%a.err

set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$WORKDIR"
mkdir -p logs/sft

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
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$WANDB_DIR"

# ---------------------------------------------------------------------------
# Run identity: every cell of the array shares the same run name + ckpt dir
# (RUN_NAME based on the parent SLURM_ARRAY_JOB_ID, not the per-cell job id).
# This is what lets cell N+1 resume cell N's checkpoint.
# ---------------------------------------------------------------------------
ARRAY_JOB="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"
RUN_NAME="${RUN_NAME:-sft_7b_array_${ARRAY_JOB}}"
CKPT_DIR="${CKPT_DIR:-${SCRATCH_ROOT}/ckpts/${RUN_NAME}}"

DATA_DIR="${DATA_DIR_SFT:-${SCRATCH_ROOT}/SkyRLMain/data/iclr_arxiv_sft_yearbalanced}"
MODEL_PATH="${MODEL_PATH:-${SCRATCH_ROOT}/hf/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28}"

BATCH_SIZE="${BATCH_SIZE:-32}"
MICRO_BS="${MICRO_BS:-1}"
NUM_STEPS="${NUM_STEPS:-412}"
EPOCH_STEPS="${EPOCH_STEPS:-206}"
CKPT_INTERVAL="${CKPT_INTERVAL:-10}"
MAX_LEN="${MAX_LEN:-40960}"
LR="${LR:-1.0e-6}"
WARMUP="${WARMUP:-10}"

echo "=============================================="
echo "Array parent: $ARRAY_JOB   Task: $TASK_ID  (this cell's job: $SLURM_JOB_ID)"
echo "Host:         $(hostname)"
echo "Partition:    gpu-test, 4× A100 80GB, --time=00:55:00"
echo "Data:         $DATA_DIR"
echo "Model init:   $MODEL_PATH"
echo "Ckpt dir:     $CKPT_DIR  (shared across all cells)"
echo "Run name:     $RUN_NAME"
echo "Steps:        $NUM_STEPS  (batch=$BATCH_SIZE, micro=$MICRO_BS, max_len=$MAX_LEN)"
echo "Schedule:     cosine, $WARMUP warmup, lr=$LR"
echo "Resume:       latest (auto-detect, fresh start if no ckpt yet)"
echo "Save:         FSDP every $CKPT_INTERVAL steps, HF every $EPOCH_STEPS steps"
echo "=============================================="
nvidia-smi
if [ -f "${CKPT_DIR}/latest_ckpt_global_step.txt" ]; then
    echo "Found prior checkpoint pointer:"
    cat "${CKPT_DIR}/latest_ckpt_global_step.txt"
fi

source .venv/bin/activate
python -m skyrl.train.main_sft \
    strategy=fsdp2 \
    model.path="$MODEL_PATH" \
    dataset_name="${DATA_DIR}/train" \
    dataset_split="train" \
    messages_key=messages \
    multi_turn_loss=true \
    use_liger_kernel=true \
    use_sample_packing=false \
    max_length=${MAX_LEN} \
    num_steps=${NUM_STEPS} \
    batch_size=${BATCH_SIZE} \
    micro_train_batch_size_per_gpu=${MICRO_BS} \
    seed=42 \
    optimizer_config.lr=${LR} \
    optimizer_config.weight_decay=0.0 \
    optimizer_config.max_grad_norm=1.0 \
    optimizer_config.num_warmup_steps=${WARMUP} \
    optimizer_config.scheduler=cosine \
    placement.num_nodes=1 \
    placement.num_gpus_per_node=4 \
    fsdp_config.cpu_offload=false \
    fsdp_config.reshard_after_forward=true \
    logger=console \
    project_name=train-prediction-sft \
    run_name="$RUN_NAME" \
    ckpt_path="$CKPT_DIR" \
    ckpt_interval=${CKPT_INTERVAL} \
    hf_save_interval=${EPOCH_STEPS} \
    save_fsdp_checkpoint=true \
    max_ckpts_to_keep=3 \
    export_path="${CKPT_DIR}/exports" \
    resume_from=latest \
    "$@"
