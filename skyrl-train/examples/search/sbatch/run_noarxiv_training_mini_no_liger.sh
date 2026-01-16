#!/bin/bash
# Mini version for testing on interactive node (4x A100 80GB) - NoArxiv - NO LIGER
# Usage: ./run_noarxiv_training_mini_no_liger.sh

# srun --nodes=1 --ntasks=1 --cpus-per-task=20 --mem=200G --gres=gpu:4 --constraint="gpu80" -t 1:00:00 -p gpu-test --pty bash

set -e

# NoArxiv Training - No search tool access (baseline)
# Model must predict paper acceptance using only paper content

echo "Job started on $(date)"
echo "Running on node: $(hostname)"

JOB_ID="${SLURM_JOB_ID:-$$}"

cd /scratch/gpfs/ZHUANGL/sk7524/SkyRLSearchEnvs/skyrl-train
echo "Working directory: $(pwd)"

export TRANSFORMERS_OFFLINE=1
export HF_HOME=/scratch/gpfs/ZHUANGL/sk7524/hf
export WANDB_API_KEY="a9c6eb313dc5aab38bec7680526824ccdbb7f5f0"
export LOGURU_LEVEL=INFO

mkdir -p logs/noarxiv
nvidia-smi

echo "==========================================="
echo "NoArxiv Training - MINI (No Liger)"
echo "Model: Qwen/Qwen3-4B-Thinking-2507"
echo "GPUs: 4, Max Turns: 1"
echo "==========================================="

MODEL_PATH="/scratch/gpfs/ZHUANGL/sk7524/hf/hub/models--Qwen--Qwen3-4B-Thinking-2507/snapshots/768f209d9ea81521153ed38c47d515654e938aea"
DATA_DIR="/scratch/gpfs/ZHUANGL/sk7524/SkyRL/skyrl-train/data/noarxiv_iclr"
EXPORT_PATH="/scratch/gpfs/ZHUANGL/sk7524/SkyRLSearchEnvs/skyrl-train/exports/noarxiv_mini_no_liger/${JOB_ID}"

echo "Data directory: $DATA_DIR"
echo "Export path: $EXPORT_PATH"

if [ ! -f "${DATA_DIR}/train.parquet" ]; then
    echo "Error: Training data not found at ${DATA_DIR}/train.parquet"
    exit 1
fi

source .venv/bin/activate
module load cudatoolkit/12.8

echo "==========================================="
echo "  train_batch_size: 64"
echo "  policy_mini_batch_size: 32"
echo "  n_samples_per_prompt: 5"
echo "  liger_kernel: DISABLED"
echo "==========================================="

python -m skyrl_train.entrypoints.main_base \
  data.train_data="['${DATA_DIR}/train.parquet']" \
  data.val_data="['${DATA_DIR}/validation.parquet']" \
  trainer.logger="console" \
  trainer.project_name="noarxiv_mini_no_liger" \
  trainer.run_name="noarxiv_mini_no_liger_${JOB_ID}" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.export_path="$EXPORT_PATH/exports" \
  trainer.dump_eval_results=false \
  trainer.dump_eval_every_batch=false \
  trainer.dump_data_batch=false \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.policy.fsdp_config.cpu_offload=false \
  trainer.policy.optimizer_config.lr=1.2e-6 \
  trainer.policy.optimizer_config.max_grad_norm=0.5 \
  trainer.policy.optimizer_config.num_warmup_steps=3 \
  trainer.placement.policy_num_gpus_per_node=4 \
  trainer.placement.ref_num_gpus_per_node=4 \
  trainer.ref.fsdp_config.cpu_offload=true \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.kl_loss_coef=0.001 \
  trainer.use_liger_kernel=false \
  trainer.epochs=1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=64 \
  trainer.policy_mini_batch_size=32 \
  trainer.micro_forward_batch_size_per_gpu=2 \
  trainer.micro_train_batch_size_per_gpu=2 \
  trainer.max_prompt_length=20480 \
  trainer.eval_batch_size=128 \
  trainer.eval_before_train=false \
  trainer.eval_interval=10 \
  trainer.ckpt_interval=50 \
  trainer.hf_save_interval=100 \
  trainer.max_ckpts_to_keep=2 \
  trainer.resume_mode=latest \
  trainer.ckpt_path="$EXPORT_PATH/ckpts" \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.async_engine=true \
  generator.batched=false \
  generator.num_inference_engines=4 \
  generator.inference_engine_tensor_parallel_size=1 \
  generator.gpu_memory_utilization=0.85 \
  generator.sampling_params.max_generate_length=2048 \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  generator.sampling_params.stop='["</answer>"]' \
  generator.eval_sampling_params.max_generate_length=2048 \
  generator.eval_sampling_params.temperature=0.0 \
  generator.eval_sampling_params.stop='["</answer>"]' \
  generator.max_input_length=20480 \
  generator.use_conversation_multi_turn=true \
  generator.append_eos_token_after_stop_str_in_multi_turn=true \
  generator.max_turns=1 \
  generator.n_samples_per_prompt=5 \
  generator.eval_n_samples_per_prompt=5 \
  environment.env_class=search_arxiv \
  environment.skyrl_gym.max_env_workers=32 \
  "$@"

TRAINING_EXIT_CODE=$?
echo "Training exit code: $TRAINING_EXIT_CODE"
echo "End time: $(date)"
exit $TRAINING_EXIT_CODE
