#!/bin/bash
#SBATCH --job-name=arxiv_train_qwen3_4b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=250G
#SBATCH --gres=gpu:4
#SBATCH --partition=ailab
#SBATCH --time=24:00:00
#SBATCH --output=logs/arxiv/%j.out
#SBATCH --error=logs/arxiv/%j.err

# No proxy - using WANDB offline mode

# Print job information
echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Change to the correct directory (original SkyRL repo for training)
cd /scratch/gpfs/ZHUANGL/sk7524/SkyRLSearchEnvs/skyrl-train
echo "Working directory: $(pwd)"

# Set offline mode for transformers (models are cached locally)
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/scratch/gpfs/ZHUANGL/sk7524/hf
export WANDB_API_KEY="a9c6eb313dc5aab38bec7680526824ccdbb7f5f0"
export WANDB_MODE=offline
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Suppress debug logs
export LOGURU_LEVEL=INFO

# Create logs directory
mkdir -p logs/arxiv

# Check GPU availability
nvidia-smi

echo "==========================================="
echo "ArXiv Training - 3 turns, 800 generate"
echo "Model: Qwen/Qwen3-4B-Thinking-2507"
echo "Backend: vLLM"
echo "GPUs: 4"
echo "Environment: search_arxiv"
echo "==========================================="

# ============================================
# Paths Configuration
# ============================================
MODEL_PATH="/scratch/gpfs/ZHUANGL/sk7524/hf/hub/models--Qwen--Qwen3-4B-Thinking-2507/snapshots/768f209d9ea81521153ed38c47d515654e938aea"
DATA_DIR="/scratch/gpfs/ZHUANGL/sk7524/SkyRL/skyrl-train/data/arxiv_iclr"
EXPORT_PATH="/scratch/gpfs/ZHUANGL/sk7524/SkyRLSearchEnvs/skyrl-train/exports/arxiv_full/${SLURM_JOB_ID}"

# ArXiv index and corpus paths
INDEX_PATH="${INDEX_PATH:-/scratch/gpfs/ZHUANGL/sk7524/SkyRL/skyrl-train/data/searchr1_original/arxiv/qwen3_06_embed/qwen3_Flat.index}"
CORPUS_PATH="${CORPUS_PATH:-/scratch/gpfs/ZHUANGL/sk7524/SkyRL/skyrl-train/data/searchr1_original/arxiv/arxiv_wikiformat.jsonl}"
ARXIV_METADATA_PATH="${ARXIV_METADATA_PATH:-/scratch/gpfs/ZHUANGL/sk7524/SkyRL/skyrl-train/data/searchr1_original/arxiv/arxiv-metadata-oai-snapshot.jsonl}"

# Retrieval server settings
RETRIEVAL_PORT=8000
SEARCH_URL="http://127.0.0.1:${RETRIEVAL_PORT}/retrieve"

# Task description for Qwen3 query instruction
TASK_DESCRIPTION="Given a query about academic research, retrieve relevant arXiv papers that address the topic"

echo "Data directory: $DATA_DIR"
echo "Export path: $EXPORT_PATH"
echo "Index path: $INDEX_PATH"
echo "Corpus path: $CORPUS_PATH"
echo "ArXiv metadata: $ARXIV_METADATA_PATH"
echo "Search URL: $SEARCH_URL"

# Verify data exists
if [ ! -f "${DATA_DIR}/train.parquet" ]; then
    echo "Error: Training data not found at ${DATA_DIR}/train.parquet"
    echo "Please run generate_dataset.sh first"
    exit 1
fi

if [ ! -f "$INDEX_PATH" ]; then
    echo "Error: Index file not found at $INDEX_PATH"
    exit 1
fi

if [ ! -f "$CORPUS_PATH" ]; then
    echo "Error: Corpus file not found at $CORPUS_PATH"
    exit 1
fi

# ============================================
# Start Retrieval Server via retrieval_launch.sh
# ============================================
echo "==========================================="
echo "Starting Retrieval Server"
echo "==========================================="

RETRIEVAL_LAUNCH_SCRIPT="/scratch/gpfs/ZHUANGL/sk7524/SkyRLSearchEnvs/skyrl-train/examples/search/retriever/retrieval_launch.sh"

echo "Launching retrieval server..."
echo "  Index: $INDEX_PATH"
echo "  Corpus: $CORPUS_PATH"
echo "  Port: $RETRIEVAL_PORT"
echo "  Task: $TASK_DESCRIPTION"
if [ -f "$ARXIV_METADATA_PATH" ]; then
    echo "  Arxiv metadata: $ARXIV_METADATA_PATH"
fi

# Call retrieval_launch.sh (it handles conda env internally)
bash "$RETRIEVAL_LAUNCH_SCRIPT" \
    qwen3 \
    "$INDEX_PATH" \
    "$CORPUS_PATH" \
    "$RETRIEVAL_PORT" \
    gpu \
    "$TASK_DESCRIPTION" \
    "$ARXIV_METADATA_PATH"

# Find the PID of the retrieval server (launched by retrieval_launch.sh with nohup)
sleep 2  # Give it a moment to start
RETRIEVAL_PID=$(pgrep -f "retrieval_server.py.*--port $RETRIEVAL_PORT" | head -1)
if [ -z "$RETRIEVAL_PID" ]; then
    echo "Warning: Could not find retrieval server PID"
else
    echo "Retrieval server PID: $RETRIEVAL_PID"
fi

# Wait for server to be ready
echo "Waiting for retrieval server to be ready..."
MAX_WAIT=300  # 5 minutes max wait
WAIT_INTERVAL=5
ELAPSED=0

while [ $ELAPSED -lt $MAX_WAIT ]; do
    if curl -s -X POST "$SEARCH_URL" \
        -H "Content-Type: application/json" \
        -d '{"query": "test", "topk": 1}' > /dev/null 2>&1; then
        echo "Retrieval server is ready!"
        break
    fi

    # Check if server process is still running (if we have the PID)
    if [ -n "$RETRIEVAL_PID" ] && ! kill -0 $RETRIEVAL_PID 2>/dev/null; then
        echo "Error: Retrieval server process died."
        exit 1
    fi

    echo "  Waiting... ($ELAPSED seconds elapsed)"
    sleep $WAIT_INTERVAL
    ELAPSED=$((ELAPSED + WAIT_INTERVAL))
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "Error: Retrieval server failed to start within $MAX_WAIT seconds"
    [ -n "$RETRIEVAL_PID" ] && kill $RETRIEVAL_PID 2>/dev/null
    exit 1
fi

# Cleanup function
cleanup() {
    echo "Cleaning up..."
    # Try to find and kill the retrieval server
    RETRIEVAL_PID=$(pgrep -f "retrieval_server.py.*--port $RETRIEVAL_PORT" | head -1)
    if [ -n "$RETRIEVAL_PID" ]; then
        echo "Stopping retrieval server (PID: $RETRIEVAL_PID)"
        kill $RETRIEVAL_PID 2>/dev/null
        wait $RETRIEVAL_PID 2>/dev/null
    fi
}
#trap cleanup EXIT

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

# Run training with search_arxiv environment
# Stop tokens: </ssearch>, </asearch>, </answer>
python -m skyrl_train.entrypoints.main_base \
  data.train_data="['${DATA_DIR}/train.parquet']" \
  data.val_data="['${DATA_DIR}/validation.parquet']" \
  trainer.logger="wandb" \
  trainer.project_name="arxiv_acceptance" \
  trainer.run_name="arxiv_train_qwen3_4b_3turns_800gen_${SLURM_JOB_ID}" \
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
  trainer.policy.optimizer_config.num_warmup_steps=3 \
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
  trainer.micro_forward_batch_size_per_gpu=2 \
  trainer.micro_train_batch_size_per_gpu=2 \
  trainer.max_prompt_length=32768 \
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
  generator.sampling_params.max_generate_length=800 \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  generator.sampling_params.stop='["</ssearch>", "</asearch>", "</answer>"]' \
  generator.eval_sampling_params.max_generate_length=800 \
  generator.eval_sampling_params.temperature=0.0 \
  generator.eval_sampling_params.stop='["</ssearch>", "</asearch>", "</answer>"]' \
  generator.max_input_length=32768 \
  generator.use_conversation_multi_turn=true \
  generator.append_eos_token_after_stop_str_in_multi_turn=true \
  generator.max_turns=3 \
  generator.n_samples_per_prompt=10 \
  generator.eval_n_samples_per_prompt=10 \
  generator.weight_sync_backend=nccl \
  environment.env_class=search_arxiv \
  environment.skyrl_gym.max_env_workers=32 \
  +environment.skyrl_gym.search_arxiv.search_url="$SEARCH_URL" \
  +environment.skyrl_gym.search_arxiv.topk=5 \
  +environment.skyrl_gym.search_arxiv.timeout=30 \
  +environment.skyrl_gym.search_arxiv.log_requests=false \
  "$@"

TRAINING_EXIT_CODE=$?

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully"
else
    echo "Training failed with exit code: $TRAINING_EXIT_CODE"
fi

echo "End time: $(date)"
exit $TRAINING_EXIT_CODE
