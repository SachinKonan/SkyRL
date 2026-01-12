#!/bin/bash
#SBATCH --job-name=eval_arxiv
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=150G
#SBATCH --gres=gpu:2
#SBATCH --partition=ailab
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval_arxiv/%j.out
#SBATCH --error=logs/eval_arxiv/%j.err

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
export WANDB_MODE=offline

# Suppress debug logs
export LOGURU_LEVEL=INFO

# Create logs directory
mkdir -p logs/eval_arxiv

# Check GPU availability
nvidia-smi

echo "==========================================="
echo "ArXiv Evaluation - 3 turns, 800 generate"
echo "Config: lowkl, bz256, gs8, cosdecay"
echo "GPUs: 2 (tensor_parallel_size=2)"
echo "==========================================="

# ============================================
# Required Arguments (pass when submitting)
# ============================================
# MODEL_PATH: Path to the model checkpoint to evaluate
# Example: sbatch --export=MODEL_PATH=/path/to/checkpoint eval_arxiv.sbatch

if [ -z "$MODEL_PATH" ]; then
    echo "Error: MODEL_PATH not set. Pass it via --export=MODEL_PATH=/path/to/checkpoint"
    exit 1
fi

echo "Model path: $MODEL_PATH"

# ============================================
# Paths Configuration
# ============================================
INPUT_FILE="/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer/data/iclr_2020_2025_80_20_split5_balanced_deepreview_clean_binary_no_reviews_v3_test/data.json"
OUTPUT_DIR="/scratch/gpfs/ZHUANGL/sk7524/SkyRLSearchEnvs/skyrl-train/exports/eval_results/${SLURM_JOB_ID}"
OUTPUT_FILE="${OUTPUT_DIR}/eval_results.jsonl"

# ArXiv index and corpus paths
INDEX_PATH="${INDEX_PATH:-/scratch/gpfs/ZHUANGL/sk7524/SkyRL/skyrl-train/data/searchr1_original/arxiv/qwen3_06_embed/qwen3_Flat.index}"
CORPUS_PATH="${CORPUS_PATH:-/scratch/gpfs/ZHUANGL/sk7524/SkyRL/skyrl-train/data/searchr1_original/arxiv/arxiv_wikiformat.jsonl}"
ARXIV_METADATA_PATH="${ARXIV_METADATA_PATH:-/scratch/gpfs/ZHUANGL/sk7524/SkyRL/skyrl-train/data/searchr1_original/arxiv/arxiv-metadata-oai-snapshot.jsonl}"

# Retrieval server settings
RETRIEVAL_PORT=8000
SEARCH_URL="http://127.0.0.1:${RETRIEVAL_PORT}/retrieve"

# Task description for Qwen3 query instruction
TASK_DESCRIPTION="Given a query about academic research, retrieve relevant arXiv papers that address the topic"

echo "Input file: $INPUT_FILE"
echo "Output file: $OUTPUT_FILE"
echo "Index path: $INDEX_PATH"
echo "Corpus path: $CORPUS_PATH"
echo "ArXiv metadata: $ARXIV_METADATA_PATH"
echo "Search URL: $SEARCH_URL"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Verify data exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found at $INPUT_FILE"
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
trap cleanup EXIT

# Activate virtual environment
source .venv/bin/activate
echo "Activated virtual environment"

# ============================================
# Run Evaluation with vllm_infer.py
# ============================================
echo "==========================================="
echo "Starting Evaluation"
echo "==========================================="

module load cudatoolkit/12.8

# Evaluation parameters (matching training sbatch eval config)
# - max_generate_length=800
# - max_turns=3
# - eval_n_samples_per_prompt=8 (group_size)
# - topk=5
# - temperature=1.0
# - tensor_parallel_size=2 (2 GPUs)
# - gpu_memory_utilization=0.85

python scripts/vllm_infer.py \
    trainer.logger="console" \
    trainer.max_prompt_length=32768 \
    trainer.policy.model.path="$MODEL_PATH" \
    generator.backend=vllm \
    generator.run_engines_locally=true \
    generator.num_inference_engines=2 \
    generator.inference_engine_tensor_parallel_size=1 \
    generator.gpu_memory_utilization=0.85 \
    generator.max_turns=3 \
    generator.eval_n_samples_per_prompt=8 \
    generator.eval_sampling_params.max_generate_length=800 \
    generator.eval_sampling_params.temperature=1.0 \
    environment.env_class=search_arxiv \
    +environment.skyrl_gym.search_arxiv.search_url="$SEARCH_URL" \
    +environment.skyrl_gym.search_arxiv.topk=5 \
    +infer_input_file="$INPUT_FILE" \
    +infer_output_file="$OUTPUT_FILE" \
    +infer_use_arxiv=true \
    "$@"

EVAL_EXIT_CODE=$?

if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo "Evaluation completed successfully"
    echo "Results saved to: $OUTPUT_FILE"
else
    echo "Evaluation failed with exit code: $EVAL_EXIT_CODE"
fi

echo "End time: $(date)"
exit $EVAL_EXIT_CODE
