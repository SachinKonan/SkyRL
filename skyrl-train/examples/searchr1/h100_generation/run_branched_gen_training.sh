#!/bin/bash
#SBATCH --job-name=searchr1_branched_gen_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=400G
#SBATCH --gres=gpu:2
#SBATCH --partition=pli
#SBATCH --account=llm_explore
#SBATCH --time=12:00:00
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

# Memory optimization for PyTorch (reduce fragmentation)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Gemini API key for embeddings (REST API)
export GEMINI_API_KEY="AIzaSyCIjfLlDmrPpVUUCSz4BXC3tDfmllWYF9M"

# Suppress debug logs
export LOGURU_LEVEL=INFO

# Create logs directory
mkdir -p logs/searchr1

# Check GPU availability
nvidia-smi

echo "==========================================="
echo "SearchR1 BRANCHED Generation on H100"
echo "Model: Qwen/Qwen3-4B-Instruct-2507"
echo "Backend: vLLM"
echo "GPUs: 2 H100 (tensor_parallel=2)"
echo "Data: training.parquet"
echo "Branching: src_trajectories=6, group_size=10"
echo "==========================================="

# Paths
MODEL_PATH="/scratch/gpfs/ZHUANGL/sk7524/hf/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"
DATA_DIR="/scratch/gpfs/ZHUANGL/sk7524/SkyRL/skyrl-train/data/searchr1"
EXPORT_PATH="/scratch/gpfs/ZHUANGL/sk7524/SkyRL/skyrl-train/exports/searchr1_branched_h100/${SLURM_JOB_ID}"
CHROMA_PATH="/scratch/gpfs/ZHUANGL/sk7524/SkyRL/skyrl-train/data/searchr1/chroma_db/tmp/tianyi/tool_caches/local_chroma/chroma_db"

echo "==========================================="
echo "Starting Branched Generation on training.parquet"
echo "==========================================="

# Run branched generation on training data
# Using data.val_data to pass training.parquet since this is a generation-only run
# Trajectory metadata will be saved including:
#   - trajectory_id: unique identifier
#   - is_branched: whether this is a branched trajectory
#   - source_repetition_id: which trajectory we branched from
#   - branch_turn: which turn we branched from
#   - branch_token_idx: token index where we branched
python -m skyrl_train.entrypoints.main_generate \
  data.val_data="['$DATA_DIR/parquets/train.parquet']" \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.logger="wandb" \
  trainer.project_name="searchr1" \
  trainer.run_name="searchr1_branched_gen_h100_train_${SLURM_JOB_ID}" \
  trainer.placement.colocate_all=false \
  trainer.export_path="$EXPORT_PATH" \
  trainer.dump_eval_results=true \
  trainer.dump_data_batch=true \
  trainer.dump_every_batch=true \
  trainer.eval_batch_size=512 \
  generator.backend=vllm \
  generator.num_inference_engines=1 \
  generator.inference_engine_tensor_parallel_size=2 \
  generator.gpu_memory_utilization=0.9 \
  generator.eval_sampling_params.max_generate_length=500 \
  generator.eval_sampling_params.temperature=0.7 \
  generator.eval_sampling_params.stop='["</search>", "</answer>"]' \
  generator.max_input_length=4096 \
  generator.use_conversation_multi_turn=true \
  generator.append_eos_token_after_stop_str_in_multi_turn=true \
  generator.max_turns=4 \
  generator.eval_n_samples_per_prompt=10 \
  generator.branching.enabled=true \
  generator.branching.src_trajectories=6 \
  generator.branching.num_branches=2 \
  environment.env_class=searchr1embeddings \
  environment.skyrl_gym.max_env_workers=32 \
  environment.skyrl_gym.searchr1embeddings.chroma_path="$CHROMA_PATH" \
  environment.skyrl_gym.searchr1embeddings.collection_name="wiki_embeddings" \
  environment.skyrl_gym.searchr1embeddings.topk=3 \
  environment.skyrl_gym.searchr1embeddings.log_requests=true \
  "$@"

if [ $? -eq 0 ]; then
    echo "Branched generation completed successfully"
    echo "Results saved to: $EXPORT_PATH"
    echo ""
    echo "Output files include trajectory metadata:"
    echo "  - trajectory_id: unique identifier (e.g., uid_0_from_1_t0_tok15)"
    echo "  - is_branched: true if this is a branched trajectory"
    echo "  - source_repetition_id: parent trajectory's repetition_id"
    echo "  - branch_turn: which turn we branched from (0-indexed)"
    echo "  - branch_token_idx: token index within response where branch occurred"
else
    echo "Branched generation failed"
fi

echo "End time: $(date)"
