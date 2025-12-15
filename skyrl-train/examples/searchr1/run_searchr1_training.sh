#!/bin/bash
#SBATCH --job-name=searchr1_gen_qwen3_4b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=300G
#SBATCH --gres=gpu:4
#SBATCH --partition=ailab
#SBATCH --time=10:00:00
#SBATCH --output=logs/searchr1/%j.out
#SBATCH --error=logs/searchr1/%j.err

module load proxy/default

# Print job information
echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Change to the correct directory
cd /scratch/gpfs/ZHUANGL/sk7524/SkyRL/skyrl-train
echo "Working directory: $(pwd)"

# Activate virtual environment
source .venv/bin/activate
echo "Activated virtual environment"

# Set offline mode for transformers (models are cached locally)
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/scratch/gpfs/ZHUANGL/sk7524/hf
export WANDB_API_KEY="a9c6eb313dc5aab38bec7680526824ccdbb7f5f0"

# Gemini API key for embeddings (REST API)
export GEMINI_API_KEY="AIzaSyCIjfLlDmrPpVUUCSz4BXC3tDfmllWYF9M"

# Suppress debug logs
export LOGURU_LEVEL=INFO

# Create logs directory
mkdir -p logs/searchr1

# Check GPU availability
nvidia-smi

echo "==========================================="
echo "SearchR1 Generation-Only Evaluation"
echo "Model: Qwen/Qwen3-4B-Instruct-2507"
echo "Backend: vLLM"
echo "GPUs: 4"
echo "==========================================="

# Paths
MODEL_PATH="/scratch/gpfs/ZHUANGL/sk7524/hf/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"
DATA_DIR="/scratch/gpfs/ZHUANGL/sk7524/SkyRL/skyrl-train/data/searchr1"
EXPORT_PATH="/scratch/gpfs/ZHUANGL/sk7524/SkyRL/skyrl-train/exports/searchr1/${SLURM_JOB_ID}"
CHROMA_PATH="/scratch/gpfs/ZHUANGL/sk7524/SkyRL/skyrl-train/data/searchr1/chroma_db/tmp/tianyi/tool_caches/local_chroma/chroma_db"

echo "==========================================="
echo "Starting Generation (no server needed!)"
echo "==========================================="

# Run generation-only evaluation
# ChromaDB and Gemini clients are embedded in the environment
# NOTE: Config uses environment.skyrl_gym.<env_class>.<param> pattern
python -m skyrl_train.entrypoints.main_base \
  data.train_data="['${DATA_DIR}/parquets/train.parquet']" \
  data.val_data="['${DATA_DIR}/parquets/validation.parquet']" \
  trainer.logger="wandb" \
  trainer.project_name="searchr1" \
  trainer.run_name="searchr1_train_qwen3_4b_4turns_maxgenlen_500_${SLURM_JOB_ID}" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.export_path="$EXPORT_PATH/exports" \
  trainer.dump_eval_results=true \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.policy.fsdp_config.cpu_offload=false \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.policy.optimizer_config.max_grad_norm=0.5 \
  trainer.policy.optimizer_config.num_warmup_steps=94 \
  trainer.placement.policy_num_gpus_per_node=4 \
  trainer.placement.ref_num_gpus_per_node=4 \
  trainer.ref.fsdp_config.cpu_offload=true \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.kl_loss_coef=0.001 \
  trainer.epochs=1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=128 \
  trainer.policy_mini_batch_size=64 \
  trainer.micro_forward_batch_size_per_gpu=4 \
  trainer.micro_train_batch_size_per_gpu=4 \
  trainer.max_prompt_length=2048 \
  trainer.eval_batch_size=256 \
  trainer.eval_before_train=false \
  trainer.eval_interval=50 \
  trainer.ckpt_interval=20 \
  trainer.hf_save_interval=100 \
  trainer.max_ckpts_to_keep=5 \
  trainer.resume_mode=latest \
  trainer.ckpt_path="$EXPORT_PATH/ckpts" \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.async_engine=true \
  generator.batched=false \
  generator.num_inference_engines=2 \
  generator.inference_engine_tensor_parallel_size=2 \
  generator.gpu_memory_utilization=0.7 \
  generator.sampling_params.max_generate_length=500 \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  generator.sampling_params.stop='["</search>", "</answer>"]' \
  generator.eval_sampling_params.max_generate_length=500 \
  generator.eval_sampling_params.temperature=0.0 \
  generator.eval_sampling_params.stop='["</search>", "</answer>"]' \
  generator.max_input_length=4096 \
  generator.use_conversation_multi_turn=true \
  generator.append_eos_token_after_stop_str_in_multi_turn=true \
  generator.max_turns=4 \
  generator.n_samples_per_prompt=10 \
  generator.eval_n_samples_per_prompt=10 \
  generator.weight_sync_backend=nccl \
  environment.env_class=searchr1embeddings \
  environment.skyrl_gym.max_env_workers=32 \
  environment.skyrl_gym.searchr1embeddings.chroma_path="$CHROMA_PATH" \
  environment.skyrl_gym.searchr1embeddings.collection_name="wiki_embeddings" \
  environment.skyrl_gym.searchr1embeddings.topk=3 \
  environment.skyrl_gym.searchr1embeddings.log_requests=true \
  "$@"

if [ $? -eq 0 ]; then
    echo "Generation completed successfully"
else
    echo "Generation failed"
fi

echo "End time: $(date)"