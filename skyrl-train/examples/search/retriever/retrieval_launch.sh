#!/bin/bash
# Usage: ./retrieval_launch.sh <model> <index_path> <corpus_path> [port] [faiss_gpu] [task_description] [arxiv_metadata_path]
# Example: ./retrieval_launch.sh qwen3 /path/to/qwen3_Flat.index /path/to/corpus.jsonl 8000

set -e

# Load modules and activate environment
module load anaconda3/2024.6
conda activate retriever

# Environment setup
export HF_HOME=/scratch/gpfs/ZHUANGL/sk7524/hf
export TRANSFORMERS_OFFLINE=1
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Local model paths
E5_MODEL_PATH=/scratch/gpfs/ZHUANGL/sk7524/hf/hub/models--intfloat--e5-base-v2/snapshots/f52bf8ec8c7124536f0efb74aca902b2995e5bcd
QWEN3_MODEL_PATH=/scratch/gpfs/ZHUANGL/sk7524/hf/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/c54f2e6e80b2d7b7de06f51cec4959f6b3e03418

if [ $# -lt 3 ]; then
    echo "Usage: $0 <model> <index_path> <corpus_path> [port] [faiss_gpu] [task_description] [arxiv_metadata_path]"
    echo "  model:               e5 or qwen3"
    echo "  index_path:          path to .index file"
    echo "  corpus_path:         path to .jsonl corpus file"
    echo "  port:                server port (default: 8000)"
    echo "  faiss_gpu:           'gpu' to use GPU for FAISS index (default: cpu)"
    echo "  task_description:    custom Qwen3 query instruction (optional)"
    echo "  arxiv_metadata_path: path to arxiv jsonl for author search (optional)"
    echo ""
    echo "Example: $0 qwen3 /path/to/qwen3_Flat.index /path/to/wiki-18.jsonl 8000"
    exit 1
fi

MODEL=$1
INDEX_PATH=$2
CORPUS_PATH=$3
PORT=${4:-8000}
FAISS_GPU=${5:-cpu}
TASK_DESCRIPTION="$6"
ARXIV_METADATA_PATH="$7"

# Validate files exist
if [ ! -f "$INDEX_PATH" ]; then
    echo "Error: Index file not found: $INDEX_PATH"
    exit 1
fi
if [ ! -f "$CORPUS_PATH" ]; then
    echo "Error: Corpus file not found: $CORPUS_PATH"
    exit 1
fi

# Model-specific configuration
case $MODEL in
    e5)
        retriever_name=e5
        retriever_path=$E5_MODEL_PATH
        ;;
    qwen3)
        retriever_name=qwen3
        retriever_path=$QWEN3_MODEL_PATH
        ;;
    *)
        echo "Error: Unknown model '$MODEL'. Supported: e5, qwen3"
        exit 1
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Create log directory
LOG_DIR="$REPO_ROOT/logs/retrieval_server"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(date '+%Y%m%d_%H%M%S').log"

echo "Starting retrieval server..."
echo "  Model:  $retriever_name ($retriever_path)"
echo "  Index:  $INDEX_PATH"
echo "  Corpus: $CORPUS_PATH"
echo "  Port:   $PORT"
echo "  FAISS:  $FAISS_GPU"
if [ -n "$TASK_DESCRIPTION" ]; then
    echo "  Task:   $TASK_DESCRIPTION"
fi
if [ -n "$ARXIV_METADATA_PATH" ]; then
    echo "  Arxiv:  $ARXIV_METADATA_PATH"
fi
echo "  Log:    $LOG_FILE"

# Build FAISS GPU flag
FAISS_GPU_FLAG=""
if [ "$FAISS_GPU" = "gpu" ]; then
    FAISS_GPU_FLAG="--faiss_gpu"
fi

# Build task description flag
TASK_FLAG=""
if [ -n "$TASK_DESCRIPTION" ]; then
    TASK_FLAG="--task_description"
fi

# Build arxiv metadata flag
ARXIV_FLAG=""
if [ -n "$ARXIV_METADATA_PATH" ]; then
    ARXIV_FLAG="--arxiv_metadata_path"
fi

nohup python "$SCRIPT_DIR/retrieval_server.py" \
  --index_path $INDEX_PATH \
  --corpus_path $CORPUS_PATH \
  --topk 3 \
  --retriever_name $retriever_name \
  --retriever_model $retriever_path \
  --port $PORT \
  $FAISS_GPU_FLAG \
  $TASK_FLAG "$TASK_DESCRIPTION" \
  $ARXIV_FLAG "$ARXIV_METADATA_PATH" > "$LOG_FILE" 2>&1 &

echo "Server started in background (PID: $!)"
echo ""
echo "Test with:"
echo "  curl -X POST http://localhost:$PORT/retrieve -H 'Content-Type: application/json' -d '{\"query\": \"What is Python?\", \"topk\": 3}'"
if [ -n "$ARXIV_METADATA_PATH" ]; then
    echo ""
    echo "Author search:"
    echo "  curl -X POST http://localhost:$PORT/retrieve -H 'Content-Type: application/json' -d '{\"last_name_author_search\": \"smith, jones\", \"topk\": 5}'"
fi
echo ""
echo "View logs:"
echo "  tail -f $LOG_FILE"
