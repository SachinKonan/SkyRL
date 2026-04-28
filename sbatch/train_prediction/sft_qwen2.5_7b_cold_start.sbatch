#!/bin/bash
# SFT cold-start on Qwen2.5-7B-Instruct using Qwen 3.5-27B expert traces replayed
# through the search_arxiv env (see examples/sft/build_sft_from_expert_traces.py).
#
# Topology: 1 node × 4 × H100 on pli.  At micro_bs=1 + Liger fused CE, 7B fits
# in ~37 GB/GPU (FSDP-sharded): 3.5 GB weights + 3.5 GB grads + 14 GB AdamW
# + ~12 GB logits-free CE + ~4 GB activations with grad checkpointing.
#
# Batch plan: 6,595 train examples (iclr_arxiv_sft_yearbalanced, perfectly 50/50
# Accept/Reject within year, no truncation under max_len=40960).
#   effective bs=32, micro_bs=1 per GPU, 4 GPUs -> grad accum=8
#   ~6595 / 32 = 206 steps/epoch. Targeting 2 epochs = 412 steps.
#
# Scheduler: cosine decay with 10 warmup steps. LR=1e-6 (conservative for
# 7B-Instruct — preserves chat capability while absorbing the expert-trace
# distribution). Weight decay 0 (short run, no overfit concern).
#
# Checkpoints: save_fsdp_checkpoint=false (HF-only). HF export at epoch
# boundaries (step 206 and 412), landing under $CKPT_DIR/exports/. Those
# directories are directly loadable by vLLM / main_generate / GRPO init.
#
# Prereq (one-time on login node, ~13 min for 6,595 traces):
#   .venv/bin/python examples/sft/build_sft_from_expert_traces.py build \
#       --out_dir /scratch/gpfs/ZHUANGL/sk7524/SkyRLMain/data/iclr_arxiv_sft_yearbalanced \
#       --balance_within_year
#
# After SFT completes, set MODEL_PATH=$CKPT_DIR/exports/global_step_{206|412}/policy
# on run_text_qwen2.5_7b_4step_search_grpo.sbatch to seed GRPO.
#
#SBATCH --job-name=sft_7b_coldstart
#SBATCH --partition=pli
#SBATCH --account=llm_explore
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=300G
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
#SBATCH --output=logs/sft/coldstart_%j.out
#SBATCH --error=logs/sft/coldstart_%j.err

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
MODEL_PATH="${MODEL_PATH:-${SCRATCH_ROOT}/hf/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28}"
RUN_NAME="${RUN_NAME:-sft_7b_coldstart_${SLURM_JOB_ID}}"
CKPT_DIR="${CKPT_DIR:-${SCRATCH_ROOT}/ckpts/${RUN_NAME}}"

BATCH_SIZE="${BATCH_SIZE:-32}"
MICRO_BS="${MICRO_BS:-1}"
NUM_STEPS="${NUM_STEPS:-412}"
EPOCH_STEPS="${EPOCH_STEPS:-206}"
MAX_LEN="${MAX_LEN:-40960}"
LR="${LR:-1.0e-6}"
WARMUP="${WARMUP:-10}"

echo "=============================================="
echo "Job:         $SLURM_JOB_ID"
echo "Host:        $(hostname)"
echo "Partition:   pli (account=llm_explore), 4× H100"
echo "Data:        $DATA_DIR"
echo "Model:       $MODEL_PATH"
echo "Ckpt:        $CKPT_DIR  (HF-only, per-epoch)"
echo "Run name:    $RUN_NAME"
echo "Steps:       $NUM_STEPS  (batch=$BATCH_SIZE, micro=$MICRO_BS, max_len=$MAX_LEN)"
echo "Schedule:    cosine decay, $WARMUP warmup steps, lr=$LR"
echo "Ckpt cadence: every $EPOCH_STEPS steps (epoch boundary)"
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
    ckpt_interval=${EPOCH_STEPS} \
    hf_save_interval=${EPOCH_STEPS} \
    save_fsdp_checkpoint=false \
    export_path="${CKPT_DIR}/exports" \
    resume_from="" \
    "$@"
