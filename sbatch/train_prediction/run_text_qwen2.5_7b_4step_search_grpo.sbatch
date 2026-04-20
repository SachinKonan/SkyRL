#!/bin/bash
# Production: Qwen2.5-7B-Instruct GRPO, 3-turn arXiv search.
# 4 x H200 on ailab, 24h, weights-only checkpoints.
#
# Launches the arxiv_retriever FastAPI server on the same node in the
# retriever conda env; training talks to it over http://127.0.0.1:$PORT.
#
#SBATCH --job-name=tp_text_7b_4step_search_grpo
#SBATCH --partition=ailab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=300G
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_prediction/text_4step_%j.out
#SBATCH --error=logs/train_prediction/text_4step_%j.err

set -uo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$WORKDIR"
mkdir -p logs/train_prediction

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

DATA_DIR="${DATA_DIR_TEXT:-${SCRATCH_ROOT}/SkyRLMain/data/iclr_arxiv_text}"
MODEL_PATH="${MODEL_PATH:-${SCRATCH_ROOT}/hf/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28}"
RUN_NAME="${RUN_NAME:-tp_text_7b_4step_search_grpo_${SLURM_JOB_ID}}"

ARXIV_RETRIEVER_DIR="${ARXIV_RETRIEVER_DIR:-${SCRATCH_ROOT}/arxiv_retriever}"
EMBED_SIZE="${EMBED_SIZE:-06b}"
RETRIEVAL_PORT="${RETRIEVAL_PORT:-8000}"
RETRIEVAL_URL="http://127.0.0.1:${RETRIEVAL_PORT}/retrieve"
RETRIEVAL_CONFIG="${ARXIV_RETRIEVER_DIR}/configs/retrieval/qwen3_${EMBED_SIZE}.yaml"
RETRIEVER_CONDA_ENV="${RETRIEVER_CONDA_ENV:-retriever}"
RETRIEVAL_WAIT_SEC="${RETRIEVAL_WAIT_SEC:-1200}"

echo "=============================================="
echo "Job:         $SLURM_JOB_ID"
echo "Host:        $(hostname)"
echo "Data:        $DATA_DIR"
echo "Model:       $MODEL_PATH"
echo "Run name:    $RUN_NAME"
echo "Retrieval:   $RETRIEVAL_URL (embed=$EMBED_SIZE)"
echo "=============================================="
nvidia-smi

# --- retriever launch (conda, faiss-gpu) ---
RETRIEVAL_PID=""
cleanup() {
    if [ -n "$RETRIEVAL_PID" ] && kill -0 "$RETRIEVAL_PID" 2>/dev/null; then
        echo "--- killing retriever (PID $RETRIEVAL_PID) ---"
        kill "$RETRIEVAL_PID" 2>/dev/null || true
        wait "$RETRIEVAL_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "=== launching arxiv_retriever (embed=$EMBED_SIZE, port=$RETRIEVAL_PORT) ==="
(
    set -e
    set +u
    module load anaconda3/2024.6
    conda activate "$RETRIEVER_CONDA_ENV"
    set -u
    cd "$ARXIV_RETRIEVER_DIR"
    python src/arxiv_retriever/server/retrieval_server.py \
        --config "$RETRIEVAL_CONFIG" \
        "server.port=$RETRIEVAL_PORT"
) > "${SCRATCH_ROOT}/skyrl-logs/retriever_${SLURM_JOB_ID}.log" 2>&1 &
RETRIEVAL_PID=$!
echo "Retriever PID: $RETRIEVAL_PID"

deadline=$(( $(date +%s) + RETRIEVAL_WAIT_SEC ))
while true; do
    if ! kill -0 "$RETRIEVAL_PID" 2>/dev/null; then
        echo "ERROR: retriever died before becoming ready"
        tail -n 40 "${SCRATCH_ROOT}/skyrl-logs/retriever_${SLURM_JOB_ID}.log" || true
        exit 1
    fi
    if curl -sS -f -X POST "$RETRIEVAL_URL" \
            -H 'Content-Type: application/json' \
            -d '{"query":"test","topk":1}' > /dev/null 2>&1; then
        echo "--- retriever ready ---"
        break
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "ERROR: retriever start timeout (${RETRIEVAL_WAIT_SEC}s)"
        tail -n 40 "${SCRATCH_ROOT}/skyrl-logs/retriever_${SLURM_JOB_ID}.log" || true
        exit 1
    fi
    sleep 10
done

# --- training ---
source .venv/bin/activate
python -m skyrl.train.entrypoints.main_base \
  data.train_data="['${DATA_DIR}/train.parquet']" \
  data.val_data="['${DATA_DIR}/validation.parquet']" \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.eps_clip_low=0.2 \
  trainer.algorithm.eps_clip_high=0.28 \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.kl_loss_coef=0.001 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.policy.optimizer_config.max_grad_norm=0.5 \
  trainer.policy.optimizer_config.num_warmup_steps=4 \
  trainer.policy.optimizer_config.scheduler=constant_with_warmup \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.policy.fsdp_config.cpu_offload=false \
  trainer.ref.fsdp_config.cpu_offload=true \
  trainer.gradient_checkpointing=true \
  trainer.use_liger_kernel=true \
  trainer.save_optimizer_state=false \
  trainer.placement.policy_num_gpus_per_node=4 \
  trainer.placement.ref_num_gpus_per_node=4 \
  generator.inference_engine.num_engines=4 \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.gpu_memory_utilization=0.85 \
  generator.inference_engine.async_engine=true \
  trainer.epochs=1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=256 \
  trainer.policy_mini_batch_size=128 \
  trainer.micro_forward_batch_size_per_gpu=2 \
  trainer.micro_train_batch_size_per_gpu=2 \
  trainer.max_prompt_length=32768 \
  generator.max_input_length=32768 \
  generator.sampling_params.max_generate_length=800 \
  generator.batched=false \
  generator.use_conversation_multi_turn=true \
  generator.append_eos_token_after_stop_str_in_multi_turn=true \
  generator.n_samples_per_prompt=8 \
  generator.max_turns=3 \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  generator.sampling_params.stop='["</ssearch>", "</asearch>", "</answer>"]' \
  environment.env_class=search_arxiv \
  environment.skyrl_gym.max_env_workers=32 \
  environment.skyrl_gym.search_arxiv.max_turns=3 \
  environment.skyrl_gym.search_arxiv.search_enabled=true \
  environment.skyrl_gym.search_arxiv.topk=5 \
  environment.skyrl_gym.search_arxiv.timeout=30 \
  environment.skyrl_gym.search_arxiv.search_url="$RETRIEVAL_URL" \
  trainer.logger=console \
  trainer.project_name=train-prediction \
  trainer.run_name="$RUN_NAME" \
  trainer.ckpt_interval=10 \
  trainer.hf_save_interval=40 \
  trainer.max_ckpts_to_keep=2 \
  trainer.resume_mode=latest \
  trainer.ckpt_path="${SCRATCH_ROOT}/ckpts/${RUN_NAME}" \
  trainer.log_path="${SCRATCH_ROOT}/skyrl-logs" \
  trainer.eval_batch_size=256 \
  trainer.eval_before_train=false \
  trainer.eval_interval=20 \
  generator.eval_sampling_params.temperature=1.0 \
  generator.eval_sampling_params.top_p=1.0 \
  generator.eval_sampling_params.stop='["</ssearch>", "</asearch>", "</answer>"]' \
  generator.eval_sampling_params.max_generate_length=800 \
  trainer.export_path="${SCRATCH_ROOT}/ckpts/${RUN_NAME}/exports" \
  "$@"
