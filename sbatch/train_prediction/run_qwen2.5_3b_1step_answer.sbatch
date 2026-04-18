#!/bin/bash
# Qwen2.5-3B direct accept/reject — one step, no search. Smoke test / baseline.
#
#SBATCH --job-name=tp_1step
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --gres=gpu:2
#SBATCH --constraint="gpu80"
#SBATCH --partition=pli
#SBATCH --output=logs/train_prediction/1step_%j.out
#SBATCH --error=logs/train_prediction/1step_%j.err

set -euo pipefail

WORKDIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &> /dev/null && pwd)"
cd "$WORKDIR"
mkdir -p logs/train_prediction

# Deliberately do NOT source _common.sh — we don't need the retriever.
export HF_HOME="${HF_HOME:-/scratch/gpfs/ZHUANGL/sk7524/hf}"
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export LOGURU_LEVEL=INFO

DATA_DIR="${DATA_DIR_TEXT:-/scratch/gpfs/ZHUANGL/sk7524/SkyRLMain/data/iclr_arxiv_text}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
RUN_NAME="${RUN_NAME:-tp_qwen2.5_3b_1step_answer}"

echo "=============================================="
echo "Job:        $SLURM_JOB_ID"
echo "Host:       $(hostname)"
echo "Data:       $DATA_DIR"
echo "Model:      $MODEL_PATH"
echo "Run name:   $RUN_NAME"
echo "=============================================="

uv run --isolated --frozen --extra fsdp -m skyrl.train.entrypoints.main_base \
  data.train_data="['${DATA_DIR}/train.parquet']" \
  data.val_data="['${DATA_DIR}/validation.parquet']" \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.policy.optimizer_config.max_grad_norm=0.5 \
  trainer.policy.optimizer_config.num_warmup_steps=40 \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.kl_loss_coef=0.001 \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.policy.fsdp_config.cpu_offload=false \
  trainer.ref.fsdp_config.cpu_offload=true \
  trainer.placement.policy_num_gpus_per_node=2 \
  trainer.placement.ref_num_gpus_per_node=2 \
  generator.inference_engine.num_engines=2 \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.gpu_memory_utilization=0.5 \
  trainer.epochs=1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=64 \
  trainer.policy_mini_batch_size=32 \
  trainer.micro_forward_batch_size_per_gpu=2 \
  trainer.micro_train_batch_size_per_gpu=2 \
  trainer.max_prompt_length=8192 \
  generator.max_input_length=10000 \
  generator.sampling_params.max_generate_length=64 \
  generator.inference_engine.async_engine=true \
  generator.batched=false \
  generator.use_conversation_multi_turn=false \
  generator.n_samples_per_prompt=4 \
  generator.max_turns=1 \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  generator.sampling_params.stop='["</answer>"]' \
  environment.env_class=search_arxiv \
  environment.skyrl_gym.max_env_workers=16 \
  environment.skyrl_gym.search_arxiv.max_turns=1 \
  environment.skyrl_gym.search_arxiv.search_enabled=false \
  trainer.logger=wandb \
  trainer.project_name=train-prediction \
  trainer.run_name="$RUN_NAME" \
  trainer.ckpt_interval=20 \
  trainer.hf_save_interval=100 \
  trainer.max_ckpts_to_keep=3 \
  trainer.resume_mode=latest \
  trainer.ckpt_path="$HOME/ckpts/${RUN_NAME}" \
  trainer.eval_batch_size=128 \
  trainer.eval_before_train=false \
  generator.eval_sampling_params.temperature=0 \
  generator.eval_sampling_params.stop='["</answer>"]' \
  generator.eval_sampling_params.max_generate_length=64 \
  trainer.export_path="$HOME/ckpts/${RUN_NAME}/exports" \
  trainer.eval_interval=50 \
  "$@"
