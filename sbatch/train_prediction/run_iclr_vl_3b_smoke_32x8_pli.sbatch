#!/bin/bash
# ICLR accept/reject 32x8 smoke -- pli variant. Mirror of g3k_smoke_32x8 but
# on the ICLR Vero VL data (21k papers from iclr_arxiv_vl_file/) so we can
# compare reward curves across tasks (geometry3k vs iclr accept/reject) on
# the same Qwen2.5-VL-3B base model + same hyperparams.
#
# Config: train_batch_size=32, policy_mini_batch_size=8 (4 grad accum steps),
# n_samples_per_prompt=8 = 256 trajectories/step.
#
#SBATCH --job-name=iclr_smoke_32x8
#SBATCH --account=llm_explore
#SBATCH --partition=pli
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --gres=gpu:2
#SBATCH --output=logs/train_prediction/iclr_smoke_32x8_pli_%j.out
#SBATCH --error=logs/train_prediction/iclr_smoke_32x8_pli_%j.err

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
export _SKYRL_USE_NEW_INFERENCE=1
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$WANDB_DIR"

DATA_DIR="${DATA_DIR_VL:-/scratch/gpfs/ZHUANGL/sk7524/SkyRLMain/data/iclr_arxiv_vl_file}"
MODEL_PATH="${MODEL_PATH:-/scratch/gpfs/ZHUANGL/sk7524/hf/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3}"
RUN_NAME="${RUN_NAME:-iclr_qwen2.5_vl_3b_smoke_32x8_pli}"

echo "=============================================="
echo "Job:        $SLURM_JOB_ID"
echo "Host:       $(hostname)"
echo "Data:       $DATA_DIR"
echo "Model:      $MODEL_PATH"
echo "Run name:   $RUN_NAME"
echo "=============================================="

MAX_PROMPT_LEN=16384
MAX_INPUT_LEN=$MAX_PROMPT_LEN
echo "Prompt length budget: max_prompt_length=$MAX_PROMPT_LEN, max_input_length=$MAX_INPUT_LEN"

source .venv/bin/activate
python -m skyrl.train.entrypoints.main_base \
  data.train_data="['${DATA_DIR}/train.parquet']" \
  data.val_data="['${DATA_DIR}/validation.parquet']" \
  trainer.use_sample_packing=false \
  generator.vision_language_generator=true \
  trainer.use_liger_kernel=true \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.policy.optimizer_config.max_grad_norm=0.5 \
  trainer.policy.optimizer_config.num_warmup_steps=0 \
  trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.loss_reduction=token_mean_legacy \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.policy.fsdp_config.cpu_offload=false \
  trainer.ref.fsdp_config.cpu_offload=true \
  trainer.gradient_checkpointing=true \
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
  generator.inference_engine.engine_init_kwargs.allowed_local_media_path=/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer \
  trainer.epochs=1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=32 \
  trainer.policy_mini_batch_size=8 \
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
  environment.skyrl_gym.max_env_workers=4 \
  environment.skyrl_gym.search_arxiv.max_turns=1 \
  environment.skyrl_gym.search_arxiv.search_enabled=false \
  trainer.logger=console \
  trainer.project_name=train-prediction \
  trainer.run_name="$RUN_NAME" \
  trainer.ckpt_interval=999999 \
  trainer.hf_save_interval=999999 \
  trainer.max_ckpts_to_keep=1 \
  trainer.resume_mode=null \
  trainer.ckpt_path="${SCRATCH_ROOT}/ckpts/${RUN_NAME}" \
  trainer.log_path="${SCRATCH_ROOT}/skyrl-logs" \
  trainer.eval_batch_size=8 \
  trainer.eval_before_train=false \
  trainer.eval_interval=999999 \
  generator.eval_sampling_params.temperature=1.0 \
  generator.eval_sampling_params.top_p=1.0 \
  generator.eval_sampling_params.stop='["</answer>"]' \
  generator.eval_sampling_params.max_generate_length=4096 \
  trainer.export_path="${SCRATCH_ROOT}/ckpts/${RUN_NAME}/exports" \
  "$@"
