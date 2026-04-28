#!/bin/bash
# SFT smoke for Qwen2.5-3B-Instruct on gpu-test (1 node × 2 A100 80GB).
# Uses the 5K-balanced subset at /scratch/gpfs/ZHUANGL/sk7524/SkyRLMain/data/iclr_arxiv_sft_5k_balanced.
# Prereq (one-time on login node):
#   .venv/bin/python examples/sft/build_sft_from_expert_traces.py build \
#       --out_dir /scratch/gpfs/ZHUANGL/sk7524/SkyRLMain/data/iclr_arxiv_sft_5k_balanced \
#       --balance_per_bucket 800
#
#SBATCH --job-name=sft_smoke_3b
#SBATCH --partition=gpu-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --gres=gpu:2
#SBATCH --constraint="gpu80"
#SBATCH --time=1:00:00
#SBATCH --output=logs/sft/smoke3b_%j.out
#SBATCH --error=logs/sft/smoke3b_%j.err

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

DATA_DIR="${DATA_DIR_SFT:-${SCRATCH_ROOT}/SkyRLMain/data/iclr_arxiv_sft_yearbalanced}"
MODEL_PATH="${MODEL_PATH:-${SCRATCH_ROOT}/hf/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1}"
RUN_NAME="${RUN_NAME:-sft_smoke_3b_${SLURM_JOB_ID}}"
CKPT_DIR="${CKPT_DIR:-${SCRATCH_ROOT}/ckpts/${RUN_NAME}}"

# ~6.6K full-trace examples with per-example tokens p50=29K, p99=40K, max=40.4K
# (measured on the yearbalanced data, 200-sample probe). 32K max_length would
# truncate 25% of examples (cutting the final <answer>\boxed{}</answer>), so
# we budget 40960. Logits at micro=1: 40960 * 152K * 2B = 12.4GB — still fits
# on 80GB A100 with sharded 3B + AdamW state. Sample packing left off in this
# smoke to isolate the config change; re-enable after verifying micro=1 works.
BATCH_SIZE="${BATCH_SIZE:-32}"
MICRO_BS="${MICRO_BS:-1}"
NUM_STEPS="${NUM_STEPS:-100}"
MAX_LEN="${MAX_LEN:-40960}"
LR="${LR:-2.0e-5}"

echo "=============================================="
echo "Job:        $SLURM_JOB_ID"
echo "Host:       $(hostname)"
echo "Data:       $DATA_DIR"
echo "Model:      $MODEL_PATH"
echo "Ckpt:       $CKPT_DIR"
echo "Run name:   $RUN_NAME"
echo "Steps:      $NUM_STEPS (batch=$BATCH_SIZE, micro=$MICRO_BS, max_len=$MAX_LEN)"
echo "=============================================="
nvidia-smi

source .venv/bin/activate
python -m skyrl.train.main_sft \
    strategy=fsdp2 \
    model.path="$MODEL_PATH" \
    dataset_name="${DATA_DIR}/train" \
    dataset_split="train" \
    messages_key=messages \
    multi_turn_loss=true \
    use_liger_kernel=true \
    max_length=${MAX_LEN} \
    num_steps=${NUM_STEPS} \
    batch_size=${BATCH_SIZE} \
    micro_train_batch_size_per_gpu=${MICRO_BS} \
    use_sample_packing=false \
    seed=42 \
    optimizer_config.lr=${LR} \
    optimizer_config.weight_decay=0.0 \
    optimizer_config.max_grad_norm=1.0 \
    optimizer_config.num_warmup_steps=10 \
    optimizer_config.scheduler=constant_with_warmup \
    placement.num_nodes=1 \
    placement.num_gpus_per_node=2 \
    fsdp_config.cpu_offload=false \
    fsdp_config.reshard_after_forward=true \
    logger=console \
    project_name=train-prediction-sft-smoke \
    run_name="$RUN_NAME" \
    ckpt_path="$CKPT_DIR" \
    ckpt_interval=${CKPT_INTERVAL:-10} \
    hf_save_interval=${CKPT_INTERVAL:-10} \
    export_path="${CKPT_DIR}/exports" \
    max_ckpts_to_keep=2 \
    resume_from="" \
    "$@"
