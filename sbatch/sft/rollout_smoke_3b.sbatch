#!/bin/bash
# Zero-shot rollout smoke: Qwen2.5-3B-Instruct on search_arxiv env.
# Single gpu-test node, 2 A100s: GPU0 retrieval (06b embed), GPU1 vLLM + driver.
# Purpose: eyeball what 3B produces zero-shot vs our SFT replay dataset.
#
#SBATCH --job-name=rollout_smoke_3b
#SBATCH --partition=gpu-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --gres=gpu:2
#SBATCH --constraint="gpu80"
#SBATCH --time=1:00:00
#SBATCH --output=logs/sft/rollout_smoke_%j.out
#SBATCH --error=logs/sft/rollout_smoke_%j.err

set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$WORKDIR"
mkdir -p logs/sft

SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/gpfs/ZHUANGL/sk7524}"
export HF_HOME="${HF_HOME:-${SCRATCH_ROOT}/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${SCRATCH_ROOT}/hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${SCRATCH_ROOT}/hf/datasets}"
export TORCH_HOME="${TORCH_HOME:-${SCRATCH_ROOT}/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SCRATCH_ROOT}/.cache}"
export TRANSFORMERS_OFFLINE=1
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME"

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-${SCRATCH_ROOT}/hf/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1}"
N_ROLLOUTS="${N_ROLLOUTS:-10}"
MAX_TURNS="${MAX_TURNS:-6}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-0.7}"

RETRIEVAL_PORT="${RETRIEVAL_PORT:-8000}"
VLLM_PORT="${VLLM_PORT:-6001}"
EMBED_SIZE="${EMBED_SIZE:-06b}"
ARXIV_RETRIEVER_DIR="${ARXIV_RETRIEVER_DIR:-${SCRATCH_ROOT}/arxiv_retriever}"
RETRIEVER_CONDA_ENV="${RETRIEVER_CONDA_ENV:-retriever}"

OUT_DIR="${OUT_DIR:-${SCRATCH_ROOT}/SkyRLMain/data/rollout_smoke}"
OUT_JSONL="${OUT_DIR}/rollouts_${SLURM_JOB_ID}.jsonl"

echo "=============================================="
echo "Job:              $SLURM_JOB_ID"
echo "Host:             $(hostname)"
echo "Model:            $MODEL_PATH"
echo "N rollouts:       $N_ROLLOUTS"
echo "Retrieval port:   $RETRIEVAL_PORT (GPU0, $EMBED_SIZE embed)"
echo "vLLM port:        $VLLM_PORT (GPU1)"
echo "Output JSONL:     $OUT_JSONL"
echo "=============================================="
nvidia-smi

# ----------------------------------------------------------------------------
# Cleanup on exit
# ----------------------------------------------------------------------------
SERVER_PIDS=()
cleanup() {
    echo "--- cleanup ---"
    for pid in "${SERVER_PIDS[@]}"; do
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    # Give them a second, then SIGKILL
    sleep 2
    for pid in "${SERVER_PIDS[@]}"; do
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
}
trap cleanup EXIT

# ----------------------------------------------------------------------------
# GPU0: retrieval server (conda env "retriever")
# ----------------------------------------------------------------------------
echo "=== launching retrieval server on GPU 0 (embed=$EMBED_SIZE) ==="
RETRIEVAL_CONFIG="${ARXIV_RETRIEVER_DIR}/configs/retrieval/qwen3_${EMBED_SIZE}.yaml"
(
    set -e
    set +u
    module load anaconda3/2024.6
    conda activate "$RETRIEVER_CONDA_ENV"
    set -u
    export CUDA_VISIBLE_DEVICES=0
    cd "$ARXIV_RETRIEVER_DIR"
    python src/arxiv_retriever/server/retrieval_server.py \
        --config "$RETRIEVAL_CONFIG" \
        "server.port=$RETRIEVAL_PORT"
) > "logs/sft/rollout_smoke_retrieval_${SLURM_JOB_ID}.log" 2>&1 &
RET_PID=$!
SERVER_PIDS+=("$RET_PID")
echo "retrieval PID: $RET_PID"

# ----------------------------------------------------------------------------
# GPU1: vLLM OpenAI server for Qwen2.5-3B
# ----------------------------------------------------------------------------
echo "=== launching vLLM for $MODEL_PATH on GPU 1 ==="
(
    export CUDA_VISIBLE_DEVICES=1
    source .venv/bin/activate
    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --served-model-name qwen2.5-3b \
        --host 127.0.0.1 --port "$VLLM_PORT" \
        --dtype bfloat16 \
        --max-model-len 32768 \
        --gpu-memory-utilization 0.85 \
        --enforce-eager
) > "logs/sft/rollout_smoke_vllm_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!
SERVER_PIDS+=("$VLLM_PID")
echo "vLLM PID: $VLLM_PID"

# ----------------------------------------------------------------------------
# Wait for both to be ready
# ----------------------------------------------------------------------------
wait_http() {
    local url="$1"; local payload="$2"; local label="$3"; local deadline=$(( $(date +%s) + 1800 ))
    echo "--- waiting for $label ($url) ---"
    while true; do
        if curl -sS -f -X POST "$url" -H 'Content-Type: application/json' -d "$payload" > /dev/null 2>&1; then
            echo "  $label ready"; return 0
        fi
        if [ "$(date +%s)" -ge "$deadline" ]; then
            echo "ERROR: timeout waiting for $label"; return 1
        fi
        sleep 10
    done
}
wait_http_get() {
    local url="$1"; local label="$2"; local deadline=$(( $(date +%s) + 1800 ))
    echo "--- waiting for $label ($url) ---"
    while true; do
        if curl -sS -f "$url" > /dev/null 2>&1; then
            echo "  $label ready"; return 0
        fi
        if [ "$(date +%s)" -ge "$deadline" ]; then
            echo "ERROR: timeout waiting for $label"; return 1
        fi
        sleep 10
    done
}
wait_http "http://127.0.0.1:${RETRIEVAL_PORT}/retrieve" '{"query":"test","topk":1}' "retrieval"
wait_http_get "http://127.0.0.1:${VLLM_PORT}/v1/models" "vllm"

# ----------------------------------------------------------------------------
# Driver: 10 rollouts via env + vLLM + retrieval
# ----------------------------------------------------------------------------
echo "=== running ${N_ROLLOUTS} rollouts ==="
mkdir -p "$OUT_DIR"
source .venv/bin/activate
python examples/sft/rollout_smoke.py \
    --vllm_url "http://127.0.0.1:${VLLM_PORT}/v1" \
    --model qwen2.5-3b \
    --search_url "http://127.0.0.1:${RETRIEVAL_PORT}/retrieve" \
    --n_rollouts "$N_ROLLOUTS" \
    --max_turns "$MAX_TURNS" \
    --max_tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --out "$OUT_JSONL" \
    --replay_dir "${SCRATCH_ROOT}/SkyRLMain/data/iclr_arxiv_sft/val"

echo ""
echo "=============================================="
echo "Rollout smoke complete. Output: $OUT_JSONL"
echo "=============================================="
