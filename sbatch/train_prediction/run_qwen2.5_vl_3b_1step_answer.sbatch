#!/bin/bash
# Qwen2.5-VL-3B direct accept/reject — 1-step, no search. VL reads paper pages
# as images and emits <answer>Accept|Reject</answer>. Plumbing smoke on
# gpu-test (1 GPU-80, batch=2, 1-hour wall).
#
# Prereq — build the smoke parquets (8 rows, all 3 splits):
#   python examples/train/search_arxiv/searchr1_arxiv_vl_dataset.py \
#       --output_dir /scratch/gpfs/ZHUANGL/sk7524/SkyRLMain/data/iclr_arxiv_vl_smoke \
#       --max_rows 8 --splits train validation test
#
#SBATCH --job-name=tp_vl_smoke
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80"
#SBATCH --partition=gpu-test
#SBATCH --output=logs/train_prediction/vl_%j.out
#SBATCH --error=logs/train_prediction/vl_%j.err

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
export _SKYRL_USE_NEW_INFERENCE=1  # required by SkyRLVLMGymGenerator (RemoteInferenceClient / render_chat_completion)
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$WANDB_DIR"

DATA_DIR="${DATA_DIR_VL:-/scratch/gpfs/ZHUANGL/sk7524/SkyRLMain/data/iclr_arxiv_vl_smoke}"
MODEL_PATH="${MODEL_PATH:-/scratch/gpfs/ZHUANGL/sk7524/hf/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3}"
RUN_NAME="${RUN_NAME:-tp_qwen2.5_vl_3b_1step_smoke}"

echo "=============================================="
echo "Job:        $SLURM_JOB_ID"
echo "Host:       $(hostname)"
echo "Data:       $DATA_DIR"
echo "Model:      $MODEL_PATH"
echo "Run name:   $RUN_NAME"
echo "=============================================="

# main_vlm.py swaps the default SkyRLGymGenerator for SkyRLVLMGymGenerator so
# image observations flow through the multi-modal path.
source .venv/bin/activate
python examples/train/search_arxiv/main_vlm.py \
  data.train_data="['${DATA_DIR}/train.parquet']" \
  data.val_data="['${DATA_DIR}/validation.parquet']" \
  trainer.use_sample_packing=false \
  trainer.use_liger_kernel=true \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.policy.optimizer_config.max_grad_norm=0.5 \
  trainer.policy.optimizer_config.num_warmup_steps=0 \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.kl_loss_coef=0.001 \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.policy.fsdp_config.cpu_offload=true \
  trainer.ref.fsdp_config.cpu_offload=true \
  trainer.gradient_checkpointing=true \
  trainer.placement.policy_num_gpus_per_node=1 \
  trainer.placement.ref_num_gpus_per_node=1 \
  generator.inference_engine.num_engines=1 \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.gpu_memory_utilization=0.9 \
  generator.inference_engine.engine_init_kwargs.allowed_local_media_path=/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer \
  trainer.epochs=1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=2 \
  trainer.policy_mini_batch_size=2 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.max_prompt_length=49152 \
  generator.max_input_length=49152 \
  generator.sampling_params.max_generate_length=4096 \
  generator.inference_engine.async_engine=true \
  generator.batched=false \
  generator.use_conversation_multi_turn=true \
  generator.append_eos_token_after_stop_str_in_multi_turn=true \
  generator.n_samples_per_prompt=2 \
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
  trainer.resume_mode=latest \
  trainer.ckpt_path="${SCRATCH_ROOT}/ckpts/${RUN_NAME}" \
  trainer.log_path="${SCRATCH_ROOT}/skyrl-logs" \
  trainer.eval_batch_size=2 \
  trainer.eval_before_train=false \
  trainer.eval_interval=999999 \
  generator.eval_sampling_params.temperature=1.0 \
  generator.eval_sampling_params.top_p=1.0 \
  generator.eval_sampling_params.stop='["</answer>"]' \
  generator.eval_sampling_params.max_generate_length=4096 \
  trainer.export_path="${SCRATCH_ROOT}/ckpts/${RUN_NAME}/exports" \
  "$@"
