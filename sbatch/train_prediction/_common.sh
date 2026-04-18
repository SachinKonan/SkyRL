# Shared helpers for sbatch/train_prediction/*.sbatch.
# Source with: source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
#
# Provides:
#   - env vars (HF_HOME, TRANSFORMERS_OFFLINE, WANDB_MODE)
#   - start_retriever / stop_retriever helpers (uses EMBED_SIZE, RETRIEVAL_PORT)

set -euo pipefail

export HF_HOME="${HF_HOME:-/scratch/gpfs/ZHUANGL/sk7524/hf}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export LOGURU_LEVEL="${LOGURU_LEVEL:-INFO}"

ARXIV_RETRIEVER_DIR="${ARXIV_RETRIEVER_DIR:-/scratch/gpfs/ZHUANGL/sk7524/arxiv_retriever}"
RETRIEVER_CONDA_ENV="${RETRIEVER_CONDA_ENV:-retriever}"
EMBED_SIZE="${EMBED_SIZE:-06b}"
RETRIEVAL_PORT="${RETRIEVAL_PORT:-8000}"
RETRIEVAL_URL="http://127.0.0.1:${RETRIEVAL_PORT}/retrieve"
RETRIEVAL_WAIT_SEC="${RETRIEVAL_WAIT_SEC:-1200}"

RETRIEVAL_PID=""

start_retriever() {
    case "$EMBED_SIZE" in
        06b|4b) ;;
        *) echo "ERROR: EMBED_SIZE must be 06b or 4b (got '$EMBED_SIZE')"; return 2 ;;
    esac
    local cfg="${ARXIV_RETRIEVER_DIR}/configs/retrieval/qwen3_${EMBED_SIZE}.yaml"
    if [ ! -f "$cfg" ]; then
        echo "ERROR: retrieval config missing: $cfg"; return 2
    fi

    mkdir -p logs
    echo "=== starting arxiv_retriever (embed=$EMBED_SIZE, port=$RETRIEVAL_PORT) ==="
    (
        set -e
        module load anaconda3/2024.6 2>/dev/null || true
        conda activate "$RETRIEVER_CONDA_ENV"
        cd "$ARXIV_RETRIEVER_DIR"
        python src/arxiv_retriever/server/retrieval_server.py \
            --config "$cfg" \
            "server.port=$RETRIEVAL_PORT"
    ) > "logs/retriever_${SLURM_JOB_ID:-local}.log" 2>&1 &
    RETRIEVAL_PID=$!
    echo "retriever PID: $RETRIEVAL_PID"

    local deadline=$(( $(date +%s) + RETRIEVAL_WAIT_SEC ))
    while true; do
        if ! kill -0 "$RETRIEVAL_PID" 2>/dev/null; then
            echo "ERROR: retriever died. Tail:"
            tail -n 40 "logs/retriever_${SLURM_JOB_ID:-local}.log" || true
            return 1
        fi
        if curl -sS -f -X POST "$RETRIEVAL_URL" \
                -H 'Content-Type: application/json' \
                -d '{"query":"test","topk":1}' > /dev/null 2>&1; then
            echo "retriever ready at $RETRIEVAL_URL"
            return 0
        fi
        if [ "$(date +%s)" -ge "$deadline" ]; then
            echo "ERROR: retriever start timeout (${RETRIEVAL_WAIT_SEC}s)"
            tail -n 40 "logs/retriever_${SLURM_JOB_ID:-local}.log" || true
            return 1
        fi
        sleep 10
    done
}

stop_retriever() {
    if [ -n "$RETRIEVAL_PID" ] && kill -0 "$RETRIEVAL_PID" 2>/dev/null; then
        echo "=== stopping retriever (PID $RETRIEVAL_PID) ==="
        kill "$RETRIEVAL_PID" 2>/dev/null || true
        wait "$RETRIEVAL_PID" 2>/dev/null || true
    fi
}
