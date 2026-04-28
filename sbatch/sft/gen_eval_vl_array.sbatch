#!/bin/bash
# Vero VL search_arxiv eval, parameterised as a SLURM array — one task per
# chunk_<k>/validation.parquet under DATA_DIR_BASE.
#
# Topology per cell:
#   GPU 0: arxiv_retriever FastAPI (06b embed)
#   GPU 1: vLLM (vision_language_generator) + Ray workers for main_vlm_generate
#
# Required env (passed via --export=ALL,...):
#   MODEL_PATH        — Vero-Qwen25-7B HF directory
#   DATA_DIR_BASE     — root containing chunk_<k>/validation.parquet
#   RUN_NAME_PREFIX   — per-chunk run names: ${RUN_NAME_PREFIX}_chunk<k>
#
#SBATCH --job-name=gen_eval_vl
#SBATCH --partition=gpu-test
#SBATCH --array=0-6%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=320G
#SBATCH --gres=gpu:2
#SBATCH --constraint="gpu80"
#SBATCH --time=1:00:00
#SBATCH --output=logs/sft/gen_eval_vl_%A_%a.out
#SBATCH --error=logs/sft/gen_eval_vl_%A_%a.err

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
export _SKYRL_USE_NEW_INFERENCE=1   # required by SkyRLVLMGymGenerator
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1   # allow max_model_len > native (rope_theta extends safely)
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$WANDB_DIR"

CHUNK="${SLURM_ARRAY_TASK_ID:-0}"
DATA_DIR="${DATA_DIR_BASE}/chunk_${CHUNK}"
RUN_NAME="${RUN_NAME_PREFIX}_chunk${CHUNK}"
EXPORT_PATH="${SCRATCH_ROOT}/ckpts/${RUN_NAME}/exports"

RETRIEVAL_PORT="${RETRIEVAL_PORT:-$((20000 + SLURM_JOB_ID % 30000))}"
EMBED_SIZE="${EMBED_SIZE:-06b}"
ARXIV_RETRIEVER_DIR="${ARXIV_RETRIEVER_DIR:-${SCRATCH_ROOT}/arxiv_retriever}"
RETRIEVER_CONDA_ENV="${RETRIEVER_CONDA_ENV:-retriever}"

MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-36864}"
MAX_INPUT_LEN="${MAX_INPUT_LEN:-$MAX_PROMPT_LEN}"
MAX_GEN_LEN="${MAX_GEN_LEN:-4096}"
MAX_TURNS="${MAX_TURNS:-10}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"   # VL is heavier than text — smaller batch
N_SAMPLES="${N_SAMPLES:-1}"
TEMPERATURE="${TEMPERATURE:-0.7}"

echo "=============================================="
echo "Array job:  ${SLURM_ARRAY_JOB_ID:-?}  task: $CHUNK  cell job: $SLURM_JOB_ID"
echo "Host:       $(hostname)"
echo "Data:       $DATA_DIR/validation.parquet"
echo "Model:      $MODEL_PATH"
echo "Run:        $RUN_NAME"
echo "Export:     $EXPORT_PATH"
echo "vLLM cfg:   max_turns=$MAX_TURNS batch=$EVAL_BATCH_SIZE prompt=$MAX_PROMPT_LEN gen=$MAX_GEN_LEN"
echo "=============================================="
nvidia-smi
test -f "$DATA_DIR/validation.parquet" || { echo "ERROR: $DATA_DIR/validation.parquet missing"; exit 1; }
test -e "$MODEL_PATH" || { echo "ERROR: MODEL_PATH not found: $MODEL_PATH"; exit 1; }

cleanup() {
    echo "--- cleanup ---"
    if [ -n "${RET_PID:-}" ] && kill -0 "$RET_PID" 2>/dev/null; then
        kill "$RET_PID" 2>/dev/null || true
        sleep 2
        kill -9 "$RET_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

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
) > "logs/sft/gen_eval_vl_${SLURM_ARRAY_JOB_ID}_${CHUNK}_retrieval.log" 2>&1 &
RET_PID=$!
echo "retrieval PID: $RET_PID"

echo "=== waiting for retrieval server ==="
deadline=$(( $(date +%s) + 1200 ))
while true; do
    if ! kill -0 "$RET_PID" 2>/dev/null; then
        echo "ERROR: retrieval server died. Last 40 lines:"
        tail -n 40 "logs/sft/gen_eval_vl_${SLURM_ARRAY_JOB_ID}_${CHUNK}_retrieval.log" || true
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
        exit 1
    fi
    sleep 10
done

echo "=== launching main_vlm_generate on GPU 1 ==="
export CUDA_VISIBLE_DEVICES=1

source .venv/bin/activate
python -m skyrl.train.entrypoints.main_vlm_generate \
  data.train_data="['${DATA_DIR}/validation.parquet']" \
  data.val_data="['${DATA_DIR}/validation.parquet']" \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.policy.fsdp_config.cpu_offload=true \
  trainer.ref.fsdp_config.cpu_offload=true \
  trainer.gradient_checkpointing=true \
  trainer.use_liger_kernel=true \
  trainer.use_sample_packing=false \
  trainer.placement.policy_num_gpus_per_node=1 \
  trainer.placement.ref_num_gpus_per_node=1 \
  generator.inference_engine.num_engines=1 \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.gpu_memory_utilization=0.5 \
  generator.inference_engine.async_engine=true \
  generator.inference_engine.engine_init_kwargs.disable_mm_preprocessor_cache=true \
  generator.inference_engine.engine_init_kwargs.max_model_len=40960 \
  trainer.train_batch_size=$EVAL_BATCH_SIZE \
  trainer.policy_mini_batch_size=$EVAL_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.max_prompt_length=$MAX_PROMPT_LEN \
  generator.max_input_length=$MAX_INPUT_LEN \
  generator.sampling_params.max_generate_length=$MAX_GEN_LEN \
  generator.batched=false \
  generator.use_conversation_multi_turn=true \
  generator.append_eos_token_after_stop_str_in_multi_turn=true \
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
  trainer.project_name=train-prediction-gen-eval \
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
echo "VL eval chunk $CHUNK complete."
echo "Dumped evals:  $EXPORT_PATH/dumped_evals/eval_only/"
echo "=============================================="
