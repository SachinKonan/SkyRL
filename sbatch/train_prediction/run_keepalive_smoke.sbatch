#!/bin/bash
# Verification smoke for the gpu_keepalive integration in main_base.py.
#
# Strategy:
#   1. Submit a normal SkyRL VL training command BUT point at a nonexistent
#      train parquet so PromptDataset.__init__ raises FileNotFoundError early.
#   2. The ray.get(skyrl_entrypoint.remote(cfg)) re-raises in main(); our
#      wrapper catches it, writes ${ckpt_path}/.error.txt, and calls
#      keepalive(ckpt_path), which holds the slurm allocation.
#   3. A background watcher in this sbatch self-touches
#      ${ckpt_path}/.keepalive.release after 90 seconds, which causes the
#      keepalive to exit cleanly and the python process to return rc=0.
#
# Pass criteria:
#   - .error.txt exists with the FileNotFoundError traceback
#   - .keepalive.log shows at least one heartbeat line
#   - python exits rc=0 (clean release-triggered exit)
#   - slurm reports COMPLETED, not FAILED/TIMEOUT/OUT_OF_MEMORY
#
#SBATCH --job-name=keepalive_smoke
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80"
#SBATCH --partition=gpu-test
#SBATCH --output=logs/train_prediction/keepalive_smoke_%j.out
#SBATCH --error=logs/train_prediction/keepalive_smoke_%j.err

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
export _SKYRL_USE_NEW_INFERENCE=1
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$WANDB_DIR"

RUN_NAME="keepalive_smoke_${SLURM_JOB_ID}"
CKPT_PATH="${SCRATCH_ROOT}/keepalive_smoke/${RUN_NAME}"
mkdir -p "$CKPT_PATH"

MODEL_PATH="${MODEL_PATH:-/scratch/gpfs/ZHUANGL/sk7524/hf/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3}"

echo "=============================================="
echo "Keepalive smoke (SkyRL main_base.py)"
echo "  Job ID:    $SLURM_JOB_ID"
echo "  Node:      $(hostname)"
echo "  CKPT path: $CKPT_PATH"
echo "  Strategy:  point train_data at /nonexistent.parquet -> training raises ->"
echo "             main() should call keepalive(ckpt_path) instead of dying"
echo "=============================================="

# Background watcher: after 90s, touch the release file to trigger clean exit
( sleep 90 && \
  echo "[smoke] self-touching $CKPT_PATH/.keepalive.release" && \
  touch "$CKPT_PATH/.keepalive.release" ) &
WATCHER_PID=$!

cleanup() { kill "$WATCHER_PID" 2>/dev/null || true; }
trap cleanup EXIT

source .venv/bin/activate
python -m skyrl.train.entrypoints.main_base \
  data.train_data="['/nonexistent.parquet']" \
  data.val_data="['/nonexistent.parquet']" \
  trainer.use_sample_packing=false \
  generator.vision_language_generator=true \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=1 \
  trainer.placement.ref_num_gpus_per_node=1 \
  generator.inference_engine.num_engines=1 \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.inference_engine.gpu_memory_utilization=0.9 \
  trainer.epochs=1 \
  trainer.train_batch_size=2 \
  trainer.policy_mini_batch_size=2 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.max_prompt_length=1024 \
  generator.max_input_length=1024 \
  generator.sampling_params.max_generate_length=128 \
  generator.batched=false \
  generator.n_samples_per_prompt=2 \
  generator.max_turns=1 \
  environment.env_class=search_arxiv \
  environment.skyrl_gym.search_arxiv.search_enabled=false \
  trainer.logger=console \
  trainer.project_name=keepalive_smoke \
  trainer.run_name="$RUN_NAME" \
  trainer.ckpt_interval=999999 \
  trainer.hf_save_interval=999999 \
  trainer.eval_before_train=false \
  trainer.eval_interval=999999 \
  trainer.ckpt_path="$CKPT_PATH" \
  trainer.log_path="${SCRATCH_ROOT}/skyrl-logs" \
  trainer.gpu_keepalive_on_failure=true

RC=$?

echo "=============================================="
echo "main_base.py exited rc=$RC"
echo "  .error.txt:        $(test -f "$CKPT_PATH/.error.txt" && echo PRESENT || echo MISSING)"
echo "  .keepalive.log:    $(test -f "$CKPT_PATH/.keepalive.log" && echo PRESENT || echo MISSING)"
echo "  .keepalive.release: $(test -f "$CKPT_PATH/.keepalive.release" && echo PRESENT || echo MISSING)"
echo "=============================================="

if [ -f "$CKPT_PATH/.error.txt" ]; then
    echo "--- .error.txt (head) ---"
    head -10 "$CKPT_PATH/.error.txt"
fi

if [ -f "$CKPT_PATH/.keepalive.log" ]; then
    echo "--- .keepalive.log (tail) ---"
    tail -20 "$CKPT_PATH/.keepalive.log"
fi

exit $RC
