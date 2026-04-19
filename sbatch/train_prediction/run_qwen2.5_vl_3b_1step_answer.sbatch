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
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80"
#SBATCH --partition=gpu-test
#SBATCH --output=logs/train_prediction/vl_%j.out
#SBATCH --error=logs/train_prediction/vl_%j.err

set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$WORKDIR"
mkdir -p logs/train_prediction

export HF_HOME="${HF_HOME:-/scratch/gpfs/ZHUANGL/sk7524/hf}"
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export LOGURU_LEVEL=INFO

DATA_DIR="${DATA_DIR_VL:-/scratch/gpfs/ZHUANGL/sk7524/SkyRLMain/data/iclr_arxiv_vl_smoke}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}"
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
uv run --isolated --frozen --extra fsdp python examples/train/search_arxiv/main_vlm.py \
  data.train_data="['${DATA_DIR}/train.parquet']" \
  data.val_data="['${DATA_DIR}/validation.parquet']" \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.policy.optimizer_config.max_grad_norm=0.5 \
  trainer.policy.optimizer_config.num_warmup_steps=0 \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.kl_loss_coef=0.001 \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.policy.fsdp_config.cpu_offload=false \
  trainer.ref.fsdp_config.cpu_offload=true \
  trainer.placement.policy_num_gpus_per_node=1 \
  trainer.placement.ref_num_gpus_per_node=1 \
  generator.inference_engine.num_engines=1 \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.gpu_memory_utilization=0.9 \
  trainer.epochs=1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=2 \
  trainer.policy_mini_batch_size=2 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.max_prompt_length=24480 \
  generator.max_input_length=28672 \
  generator.sampling_params.max_generate_length=4096 \
  generator.inference_engine.async_engine=true \
  generator.batched=false \
  generator.use_conversation_multi_turn=false \
  generator.n_samples_per_prompt=2 \
  generator.max_turns=1 \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  generator.sampling_params.stop='["</answer>"]' \
  environment.env_class=search_arxiv \
  environment.skyrl_gym.max_env_workers=4 \
  environment.skyrl_gym.search_arxiv.max_turns=1 \
  environment.skyrl_gym.search_arxiv.search_enabled=false \
  trainer.logger=wandb \
  trainer.project_name=train-prediction \
  trainer.run_name="$RUN_NAME" \
  trainer.ckpt_interval=999999 \
  trainer.hf_save_interval=999999 \
  trainer.max_ckpts_to_keep=1 \
  trainer.resume_mode=latest \
  trainer.ckpt_path="$HOME/ckpts/${RUN_NAME}" \
  trainer.eval_batch_size=2 \
  trainer.eval_before_train=false \
  trainer.eval_interval=999999 \
  generator.eval_sampling_params.temperature=1.0 \
  generator.eval_sampling_params.top_p=1.0 \
  generator.eval_sampling_params.stop='["</answer>"]' \
  generator.eval_sampling_params.max_generate_length=4096 \
  trainer.export_path="$HOME/ckpts/${RUN_NAME}/exports" \
  "$@"
