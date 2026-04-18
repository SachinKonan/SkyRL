#!/bin/bash
# Smoke test for the arxiv_retriever FastAPI server. Launches the server with
# either the 0.6B or 4B Qwen3 embedding config (EMBED_SIZE=06b|4b), polls
# /retrieve for readiness, fires a handful of queries, then tears down.
#
# Usage:
#   sbatch sbatch/test_retrieval/retrieval_test.sbatch              # default EMBED_SIZE=06b
#   EMBED_SIZE=4b sbatch sbatch/test_retrieval/retrieval_test.sbatch
#
#SBATCH --job-name=retrieval_test
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80"
#SBATCH --partition=gpu-test
#SBATCH --output=logs/retrieval_test_%j.out
#SBATCH --error=logs/retrieval_test_%j.err

set -euo pipefail

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
EMBED_SIZE="${EMBED_SIZE:-06b}"                     # 06b | 4b
RETRIEVAL_PORT="${RETRIEVAL_PORT:-8000}"
ARXIV_RETRIEVER_DIR="${ARXIV_RETRIEVER_DIR:-/scratch/gpfs/ZHUANGL/sk7524/arxiv_retriever}"
RETRIEVER_CONDA_ENV="${RETRIEVER_CONDA_ENV:-retriever}"
WAIT_TIMEOUT_SEC="${WAIT_TIMEOUT_SEC:-1200}"        # 20 min — 4B model load is slow

WORKDIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &> /dev/null && pwd)"
cd "$WORKDIR"
mkdir -p logs

case "$EMBED_SIZE" in
    06b|4b) ;;
    *) echo "ERROR: EMBED_SIZE must be 06b or 4b (got '$EMBED_SIZE')"; exit 2 ;;
esac

RETRIEVAL_CONFIG="${ARXIV_RETRIEVER_DIR}/configs/retrieval/qwen3_${EMBED_SIZE}.yaml"
RETRIEVAL_URL="http://127.0.0.1:${RETRIEVAL_PORT}/retrieve"

if [ ! -f "$RETRIEVAL_CONFIG" ]; then
    echo "ERROR: retrieval config not found: $RETRIEVAL_CONFIG"
    exit 2
fi

export HF_HOME="${HF_HOME:-/scratch/gpfs/ZHUANGL/sk7524/hf}"
export TRANSFORMERS_OFFLINE=1

echo "=============================================="
echo "Job:               $SLURM_JOB_ID"
echo "Host:              $(hostname)"
echo "EMBED_SIZE:        $EMBED_SIZE"
echo "Retrieval config:  $RETRIEVAL_CONFIG"
echo "Retrieval URL:     $RETRIEVAL_URL"
echo "=============================================="

# ----------------------------------------------------------------------------
# Cleanup on exit
# ----------------------------------------------------------------------------
cleanup() {
    echo "--- cleanup ---"
    if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# ----------------------------------------------------------------------------
# Launch server (conda env so faiss-gpu resolves)
# ----------------------------------------------------------------------------
echo "=== launching retrieval server (embed=$EMBED_SIZE) ==="
(
    set -e
    module load anaconda3/2024.6 2>/dev/null || true
    conda activate "$RETRIEVER_CONDA_ENV"
    cd "$ARXIV_RETRIEVER_DIR"
    python src/arxiv_retriever/server/retrieval_server.py \
        --config "$RETRIEVAL_CONFIG" \
        "server.port=$RETRIEVAL_PORT"
) > "logs/retrieval_test_server_${SLURM_JOB_ID}.log" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# ----------------------------------------------------------------------------
# Poll /retrieve until ready (or timeout)
# ----------------------------------------------------------------------------
echo "=== waiting for server (timeout ${WAIT_TIMEOUT_SEC}s) ==="
deadline=$(( $(date +%s) + WAIT_TIMEOUT_SEC ))
while true; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: server process died. Last 40 log lines:"
        tail -n 40 "logs/retrieval_test_server_${SLURM_JOB_ID}.log" || true
        exit 1
    fi
    if curl -sS -f -X POST "$RETRIEVAL_URL" \
            -H 'Content-Type: application/json' \
            -d '{"query":"test","topk":1}' > /dev/null 2>&1; then
        echo "  server ready"
        break
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "ERROR: timeout waiting for server"
        tail -n 40 "logs/retrieval_test_server_${SLURM_JOB_ID}.log" || true
        exit 1
    fi
    sleep 10
done

# ----------------------------------------------------------------------------
# Canned queries
# ----------------------------------------------------------------------------
run_query() {
    local label="$1"; shift
    local payload="$1"; shift
    echo ""
    echo "----- $label -----"
    echo "payload: $payload"
    curl -sS -X POST "$RETRIEVAL_URL" \
        -H 'Content-Type: application/json' \
        -d "$payload" \
        | python -c "import json,sys; d=json.loads(sys.stdin.read()); r=d.get('result',[]); n=sum(len(b) for b in r) if r else 0; print(f'result batches={len(r)} total_docs={n}'); \
                     print('first doc title:', (r[0][0].get('document',{}).get('title') or r[0][0].get('title')) if (r and r[0]) else 'none')"
}

run_query "semantic search (basic)" \
  '{"query":"transformer language model scaling laws","topk":3}'

run_query "semantic search (date-bounded, exclude title)" \
  '{"query":"robust training with noisy labels","topk":3,"upper_bound_datetime":"2019-09-25","exclude_title":"ROBUST TRAINING WITH ENSEMBLE CONSENSUS"}'

run_query "author search" \
  '{"last_name_author_search":"vaswani","topk":3}'

run_query "semantic search (scores on)" \
  '{"query":"deep learning on noisy datasets","topk":3,"return_scores":true}'

echo ""
echo "=============================================="
echo "Retrieval test complete for EMBED_SIZE=$EMBED_SIZE"
echo "=============================================="
