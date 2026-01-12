#!/bin/bash
#SBATCH --job-name=noarxiv_train_qwen3_4b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=250G
#SBATCH --gres=gpu:4
#SBATCH --partition=ailab
#SBATCH --time=24:00:00
#SBATCH --output=logs/noarxiv/%j.out
#SBATCH --error=logs/noarxiv/%j.err

# NoArxiv Training - No search tool access (baseline)
# Model must predict paper acceptance using only paper content

# Print job information
echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Change to the correct directory
cd /scratch/gpfs/ZHUANGL/sk7524/SkyRLSearchEnvs/skyrl-train
echo "Working directory: $(pwd)"

# Set offline mode for transformers (models are cached locally)
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/scratch/gpfs/ZHUANGL/sk7524/hf
export WANDB_API_KEY="a9c6eb313dc5aab38bec7680526824ccdbb7f5f0"
export WANDB_MODE=offline

# Suppress debug logs
export LOGURU_LEVEL=INFO

# Create logs directory
mkdir -p logs/noarxiv

# Check GPU availability
nvidia-smi

echo "==========================================="
echo "NoArxiv Training - BASELINE (No Search)"
echo "Config: lowkl, bz256, gs8, cosdecay10"
echo "Model: Qwen/Qwen3-4B-Thinking-2507"
echo "Backend: vLLM"
echo "GPUs: 4"
echo "Max Turns: 1 (no search possible)"
echo "==========================================="

# ============================================
# Paths Configuration
# ============================================
MODEL_PATH="/scratch/gpfs/ZHUANGL/sk7524/hf/hub/models--Qwen--Qwen3-4B-Thinking-2507/snapshots/768f209d9ea81521153ed38c47d515654e938aea"
DATA_DIR="/scratch/gpfs/ZHUANGL/sk7524/SkyRL/skyrl-train/data/noarxiv_iclr"
EXPORT_PATH="/scratch/gpfs/ZHUANGL/sk7524/SkyRLSearchEnvs/skyrl-train/exports/noarxiv_full/${SLURM_JOB_ID}"

echo "Data directory: $DATA_DIR"
echo "Export path: $EXPORT_PATH"

# Verify data exists
if [ ! -f "${DATA_DIR}/train.parquet" ]; then
    echo "Error: Training data not found at ${DATA_DIR}/train.parquet"
    echo "Please run: python examples/search/searchr1_noarxiv_dataset.py --input_json <path_to_json> --output_dir $DATA_DIR"
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate
echo "Activated virtual environment ready to train"

# ============================================
# Start Training
# ============================================
echo "==========================================="
echo "Starting Training"
echo "==========================================="

module load cudatoolkit/12.8

# Run training with search_arxiv environment but max_turns=1 (no search possible)
# Stop token: </answer> only (no search tags needed)
python -m skyrl_train.entrypoints.main_base \
  data.train_data="['${DATA_DIR}/train.parquet']" \
  data.val_data="['${DATA_DIR}/validation.parquet']" \
  trainer.logger="wandb" \
  trainer.project_name="noarxiv_acceptance" \
  trainer.run_name="noarxiv_lowkl_bz256_gs8_cosdecay_${SLURM_JOB_ID}" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.export_path="$EXPORT_PATH/exports" \
  trainer.dump_eval_results=true \
  trainer.dump_eval_every_batch=true \
  trainer.dump_data_batch=false \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.policy.fsdp_config.cpu_offload=false \
  trainer.policy.optimizer_config.lr=1.2e-6 \
  trainer.policy.optimizer_config.max_grad_norm=0.5 \
  trainer.policy.optimizer_config.scheduler="cosine" \
  trainer.policy.optimizer_config.num_warmup_steps=3 \
  trainer.placement.policy_num_gpus_per_node=4 \
  trainer.placement.ref_num_gpus_per_node=4 \
  trainer.ref.fsdp_config.cpu_offload=true \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.kl_loss_coef=0.0001 \
  trainer.epochs=1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=256 \
  trainer.policy_mini_batch_size=128 \
  trainer.micro_forward_batch_size_per_gpu=4 \
  trainer.micro_train_batch_size_per_gpu=4 \
  trainer.max_prompt_length=20480 \
  trainer.eval_batch_size=256 \
  trainer.eval_before_train=false \
  trainer.eval_interval=20 \
  trainer.ckpt_interval=-1 \
  trainer.hf_save_interval=20 \
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
  generator.n_samples_per_prompt=8 \
  generator.eval_n_samples_per_prompt=8 \
  generator.weight_sync_backend=nccl \
  environment.env_class=search_arxiv \
  environment.skyrl_gym.max_env_workers=32 \
  "$@"

TRAINING_EXIT_CODE=$?

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully"
else
    echo "Training failed with exit code: $TRAINING_EXIT_CODE"
fi

echo "End time: $(date)"
exit $TRAINING_EXIT_CODE
