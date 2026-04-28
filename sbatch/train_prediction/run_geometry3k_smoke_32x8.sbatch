#!/bin/bash
# Geometry-3K VL smoke. Trains Qwen2.5-VL-3B for a few GRPO steps on a small
# train slice from the geometry3k dataset to validate the new VL stack:
#   - vllm 0.19.1rc1.dev391+g80b18230e (PR #38405 multimodal /generate endpoint)
#   - torch 2.11
#   - main_base.py + generator.vision_language_generator=true dispatch
#
# Prereq -- build the parquet (one-time):
#   .venv/bin/python examples/train/geometry3k/geometry_3k_dataset.py \
#       --output_dir data/geometry_3k
#
#SBATCH --job-name=g3k_smoke_32x8
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --gres=gpu:2
#SBATCH --constraint="gpu80"
#SBATCH --partition=gpu-test
#SBATCH --output=logs/train_prediction/g3k_smoke_32x8_%j.out
#SBATCH --error=logs/train_prediction/g3k_smoke_32x8_%j.err

set -euo pipefail

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
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$WANDB_DIR"

OUT_ROOT="${OUT_ROOT:-/scratch/gpfs/ZHUANGL/sk7524/SkyRLVisionMain}"
DATA_DIR="${DATA_DIR:-${OUT_ROOT}/data/geometry_3k}"
MODEL_PATH="${MODEL_PATH:-/scratch/gpfs/ZHUANGL/sk7524/hf/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3}"
RUN_NAME="${RUN_NAME:-g3k_qwen2.5_vl_3b_smoke_32x8}"

echo "=============================================="
echo "Job:         $SLURM_JOB_ID"
echo "Host:        $(hostname)"
echo "Data:        $DATA_DIR"
echo "Model:       $MODEL_PATH"
echo "Run name:    $RUN_NAME"
echo "Out root:    $OUT_ROOT"
echo "=============================================="

source .venv/bin/activate
python examples/train/geometry3k/geometry3k_entrypoint.py \
  data.train_data="['${DATA_DIR}/train.parquet']" \
  data.val_data="['${DATA_DIR}/test.parquet']" \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.policy.fsdp_config.cpu_offload=false \
  trainer.ref.fsdp_config.cpu_offload=true \
  trainer.placement.policy_num_gpus_per_node=2 \
  trainer.placement.critic_num_gpus_per_node=2 \
  trainer.placement.ref_num_gpus_per_node=2 \
  generator.inference_engine.num_engines=2 \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.inference_engine.gpu_memory_utilization=0.7 \
  generator.batched=false \
  generator.vision_language_generator=true \
  generator.n_samples_per_prompt=8 \
  generator.max_turns=3 \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  generator.sampling_params.max_generate_length=2048 \
  trainer.epochs=1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=32 \
  trainer.policy_mini_batch_size=8 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.max_prompt_length=4096 \
  generator.max_input_length=8192 \
  trainer.use_sample_packing=false \
  trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.loss_reduction=token_mean_legacy \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.policy.optimizer_config.max_grad_norm=0.5 \
  trainer.policy.optimizer_config.num_warmup_steps=0 \
  environment.env_class=geometry3k \
  trainer.logger=console \
  trainer.project_name=geometry3k \
  trainer.run_name="$RUN_NAME" \
  trainer.ckpt_interval=999999 \
  trainer.hf_save_interval=999999 \
  trainer.max_ckpts_to_keep=1 \
  trainer.resume_mode=null \
  trainer.eval_batch_size=8 \
  trainer.eval_before_train=false \
  trainer.eval_interval=999999 \
  trainer.dump_eval_results=true \
  trainer.ckpt_path="${OUT_ROOT}/ckpts/${RUN_NAME}" \
  trainer.log_path="${OUT_ROOT}/skyrl-logs" \
  trainer.export_path="${OUT_ROOT}/ckpts/${RUN_NAME}/exports" \
  "$@"
