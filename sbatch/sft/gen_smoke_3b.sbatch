#!/bin/bash
# Zero-shot eval of Qwen2.5-3B-Instruct on the search_arxiv env via SkyRL's
# built-in main_generate entrypoint. Runs the real generator + env + reward
# stack on iclr_arxiv_text_search/validation.parquet and dumps per-trajectory
# results (same format as training-time eval).
#
# Topology: 1 gpu-test node × 2 A100 80GB.
#   GPU 0: arxiv_retriever FastAPI (06b embed)
#   GPU 1: vLLM + Ray workers for main_generate
#
# Prereqs (one-time on login node, ~3s):
#   .venv/bin/python examples/train/search_arxiv/searchr1_arxiv_dataset.py \
#       --output_dir /scratch/gpfs/ZHUANGL/sk7524/SkyRLMain/data/iclr_arxiv_text_search \
#       --splits validation test --prompt_mode search
#
#SBATCH --job-name=gen_smoke_3b
#SBATCH --partition=gpu-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --gres=gpu:2
#SBATCH --constraint="gpu80"
#SBATCH --time=1:00:00
#SBATCH --output=logs/sft/gen_smoke_%j.out
#SBATCH --error=logs/sft/gen_smoke_%j.err

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
export WANDB_DIR="${WANDB_DIR:-${SCRATCH_ROOT}/wandb}"
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export LOGURU_LEVEL=INFO
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$WANDB_DIR"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR="${DATA_DIR:-${SCRATCH_ROOT}/SkyRLMain/data/iclr_arxiv_text_search}"
MODEL_PATH="${MODEL_PATH:-${SCRATCH_ROOT}/hf/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1}"
RUN_NAME="${RUN_NAME:-gen_smoke_3b_${SLURM_JOB_ID}}"
EXPORT_PATH="${EXPORT_PATH:-${SCRATCH_ROOT}/ckpts/${RUN_NAME}/exports}"

RETRIEVAL_PORT="${RETRIEVAL_PORT:-8000}"
EMBED_SIZE="${EMBED_SIZE:-06b}"
ARXIV_RETRIEVER_DIR="${ARXIV_RETRIEVER_DIR:-${SCRATCH_ROOT}/arxiv_retriever}"
RETRIEVER_CONDA_ENV="${RETRIEVER_CONDA_ENV:-retriever}"

MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-32768}"
MAX_GEN_LEN="${MAX_GEN_LEN:-2048}"
MAX_TURNS="${MAX_TURNS:-6}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
N_SAMPLES="${N_SAMPLES:-1}"
TEMPERATURE="${TEMPERATURE:-0.7}"

echo "=============================================="
echo "Job:        $SLURM_JOB_ID"
echo "Host:       $(hostname)"
echo "Data:       $DATA_DIR/validation.parquet"
echo "Model:      $MODEL_PATH"
echo "Run:        $RUN_NAME"
echo "Export:     $EXPORT_PATH"
echo "Retrieval:  GPU0 (embed=$EMBED_SIZE, port=$RETRIEVAL_PORT)"
echo "vLLM+Ray:   GPU1 (max_turns=$MAX_TURNS, batch=$EVAL_BATCH_SIZE)"
echo "=============================================="
nvidia-smi

# ---------------------------------------------------------------------------
# Cleanup on exit
# ---------------------------------------------------------------------------
cleanup() {
    echo "--- cleanup ---"
    if [ -n "${RET_PID:-}" ] && kill -0 "$RET_PID" 2>/dev/null; then
        kill "$RET_PID" 2>/dev/null || true
        sleep 2
        kill -9 "$RET_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# GPU0: arxiv_retriever
# ---------------------------------------------------------------------------
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
) > "logs/sft/gen_smoke_retrieval_${SLURM_JOB_ID}.log" 2>&1 &
RET_PID=$!
echo "retrieval PID: $RET_PID"

# Poll until ready
echo "=== waiting for retrieval server ==="
deadline=$(( $(date +%s) + 1200 ))
while true; do
    if ! kill -0 "$RET_PID" 2>/dev/null; then
        echo "ERROR: retrieval server died. Last 40 lines:"
        tail -n 40 "logs/sft/gen_smoke_retrieval_${SLURM_JOB_ID}.log" || true
        exit 1
    fi
    if curl -sS -f -X POST "http://127.0.0.1:${RETRIEVAL_PORT}/retrieve" \
            -H 'Content-Type: application/json' \
            -d '{"query":"test","topk":1}' > /dev/null 2>&1; then
        echo "  retrieval ready"
        break
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "ERROR: timeout waiting for retrieval"
        tail -n 40 "logs/sft/gen_smoke_retrieval_${SLURM_JOB_ID}.log" || true
        exit 1
    fi
    sleep 10
done

# ---------------------------------------------------------------------------
# GPU1: main_generate (Ray + vLLM + env workers)
# ---------------------------------------------------------------------------
echo "=== launching main_generate on GPU 1 ==="
export CUDA_VISIBLE_DEVICES=1

source .venv/bin/activate
python -m skyrl.train.entrypoints.main_generate \
  data.train_data="['${DATA_DIR}/validation.parquet']" \
  data.val_data="['${DATA_DIR}/validation.parquet']" \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.policy.fsdp_config.cpu_offload=false \
  trainer.ref.fsdp_config.cpu_offload=true \
  trainer.gradient_checkpointing=true \
  trainer.placement.policy_num_gpus_per_node=1 \
  trainer.placement.ref_num_gpus_per_node=1 \
  generator.inference_engine.num_engines=1 \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.gpu_memory_utilization=0.85 \
  generator.inference_engine.async_engine=true \
  trainer.train_batch_size=$EVAL_BATCH_SIZE \
  trainer.policy_mini_batch_size=$EVAL_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=2 \
  trainer.micro_train_batch_size_per_gpu=2 \
  trainer.max_prompt_length=$MAX_PROMPT_LEN \
  generator.max_input_length=$MAX_PROMPT_LEN \
  generator.sampling_params.max_generate_length=$MAX_GEN_LEN \
  generator.batched=false \
  generator.use_conversation_multi_turn=true \
  generator.n_samples_per_prompt=1 \
  generator.eval_n_samples_per_prompt=$N_SAMPLES \
  generator.max_turns=$MAX_TURNS \
  generator.sampling_params.temperature=$TEMPERATURE \
  generator.sampling_params.top_p=1.0 \
  generator.sampling_params.stop='["</ssearch>","</python>","</answer>"]' \
  environment.env_class=search_arxiv \
  environment.skyrl_gym.max_env_workers=8 \
  environment.skyrl_gym.search_arxiv.max_turns=$MAX_TURNS \
  environment.skyrl_gym.search_arxiv.search_enabled=true \
  environment.skyrl_gym.search_arxiv.python_enabled=true \
  environment.skyrl_gym.search_arxiv.search_url="http://127.0.0.1:${RETRIEVAL_PORT}/retrieve" \
  trainer.logger=console \
  trainer.project_name=train-prediction-gen-smoke \
  trainer.run_name="$RUN_NAME" \
  trainer.eval_batch_size=$EVAL_BATCH_SIZE \
  trainer.eval_before_train=true \
  trainer.dump_eval_results=true \
  generator.eval_sampling_params.temperature=$TEMPERATURE \
  generator.eval_sampling_params.top_p=1.0 \
  generator.eval_sampling_params.stop='["</ssearch>","</python>","</answer>"]' \
  generator.eval_sampling_params.max_generate_length=$MAX_GEN_LEN \
  trainer.export_path="$EXPORT_PATH" \
  "$@"

echo ""
echo "=============================================="
echo "Generation smoke complete."
echo "Export path:   $EXPORT_PATH"
echo "Dumped evals:  $EXPORT_PATH/dumped_evals/eval_only/"
echo "=============================================="
