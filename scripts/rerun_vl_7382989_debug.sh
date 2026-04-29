#!/usr/bin/env bash
# Interactive debug rerun of the VL-7B 1-step GRPO command, using the slurm
# allocation already held by job 7382989's keepalive on della-i19g3.
#
# Usage (after `ssh della-i19g3`):
#   bash /scratch/gpfs/ZHUANGL/sk7524/SkyRLMain/scripts/rerun_vl_7382989_debug.sh
#   bash /scratch/gpfs/ZHUANGL/sk7524/SkyRLMain/scripts/rerun_vl_7382989_debug.sh 128   # different concurrency
#   bash /scratch/gpfs/ZHUANGL/sk7524/SkyRLMain/scripts/rerun_vl_7382989_debug.sh 256
#
# What it does:
#   1. touches .keepalive.pause on the held job's model_dir to free GPUs
#   2. waits ~5 s for vLLM to release them, prints nvidia-smi
#   3. installs an EXIT/INT/TERM trap that auto-rms .keepalive.pause when this
#      script ends, so the OUTER keepalive immediately resumes warming the GPUs
#      (vLLM subprocess re-starts within ~2 s of our exit -- no manual rm needed)
#   4. exports SKYRL_GENERATE_CONCURRENCY_PER_ENGINE=$1 (default 64)
#   5. runs python -m skyrl.train.entrypoints.main_base with the same flags as the
#      original sbatch but with gpu_keepalive_on_failure=false (this rerun
#      fails fast if it crashes; the OUTER keepalive keeps the slot alive).
#
# To exit the held allocation entirely (release slurm node):
#   touch /scratch/gpfs/ZHUANGL/sk7524/ckpts/tp_vl_7b_1step_grpo_small_7382989/.keepalive.release

set -uo pipefail

# args: $1 = generator.gather_mini_batch_size  -- "" or 0 = no chunking, else chunked tqdm.gather (default 64)
#       $2 = train_batch_size                 (default 256)
#       $3 = n_samples_per_prompt             (default 8)
#       $4 = SKYRL_GENERATE_CONCURRENCY_PER_ENGINE   (default 0 = disabled, all tasks fire immediately)
# total rollouts per step = $2 * $3 ; chunked into ceil(total/$1) sequential awaits.
GATHER_MB="${1:-64}"
BS="${2:-256}"
SAMPLES="${3:-8}"
CONCURRENCY="${4:-0}"

ORIG_RUN_NAME=tp_vl_7b_1step_grpo_small_7382989
ORIG_CKPT=/scratch/gpfs/ZHUANGL/sk7524/ckpts/${ORIG_RUN_NAME}

echo "=============================================="
echo "VL-7B debug rerun"
echo "  gather_mini_batch  : ${GATHER_MB}  (chunked tqdm.gather across all rollouts)"
echo "  train_batch_size   : ${BS}"
echo "  n_samples_per_prompt: ${SAMPLES}"
echo "  total rollouts/step: $((BS * SAMPLES))   chunks=$(( (BS * SAMPLES + GATHER_MB - 1) / GATHER_MB ))"
echo "  in-flight cap/engine: ${CONCURRENCY}  (0=off, fires all chunk tasks at once)"
echo "  outer keepalive    : ${ORIG_CKPT}"
echo "=============================================="

# 1) pause the outer keepalive's vLLM so its 4 H200s are free
echo "[step 1] pausing outer keepalive..."
touch "${ORIG_CKPT}/.keepalive.pause"
sleep 5
echo "[step 1] GPU state after pause:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>&1 | head -8

# Always resume the outer keepalive on exit (success, failure, or Ctrl+C)
# so the slurm allocation stays warm even after this debug run ends.
_resume_keepalive() {
    if [ -f "${ORIG_CKPT}/.keepalive.pause" ]; then
        echo ""
        echo "[exit-trap] resuming outer keepalive (rm ${ORIG_CKPT}/.keepalive.pause)"
        rm -f "${ORIG_CKPT}/.keepalive.pause"
    fi
}
trap _resume_keepalive EXIT INT TERM

# 2) env
cd /scratch/gpfs/ZHUANGL/sk7524/SkyRLMain
source .venv/bin/activate

export SCRATCH_ROOT=/scratch/gpfs/ZHUANGL/sk7524
export HF_HOME=$SCRATCH_ROOT/hf
export TRANSFORMERS_CACHE=$SCRATCH_ROOT/hf
export HF_DATASETS_CACHE=$SCRATCH_ROOT/hf/datasets
export TORCH_HOME=$SCRATCH_ROOT/torch
export XDG_CACHE_HOME=$SCRATCH_ROOT/.cache
export WANDB_DIR=$SCRATCH_ROOT/wandb
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export LOGURU_LEVEL=INFO
export _SKYRL_USE_NEW_INFERENCE=1
export SKYRL_GENERATE_CONCURRENCY_PER_ENGINE=${CONCURRENCY}

NUM_GPUS=4
DATA_DIR=$SCRATCH_ROOT/SkyRLMain/data/iclr_arxiv_vl_file
MODEL_PATH=$SCRATCH_ROOT/hf/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5
RUN_NAME="${ORIG_RUN_NAME}_rerun_mb${GATHER_MB}_b${BS}_s${SAMPLES}_c${CONCURRENCY}_$(date +%Y%m%d_%H%M%S)"
MINI=${BS}
MAX_PROMPT_LEN=36864

echo "[step 2] launching:"
echo "  RUN_NAME           : ${RUN_NAME}"
echo "  ckpt_path          : ${SCRATCH_ROOT}/ckpts/${RUN_NAME}"
echo "  bs / mini / samples: ${BS} / ${MINI} / 8"
echo ""

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
  trainer.policy.fsdp_config.cpu_offload=true \
  trainer.ref.fsdp_config.cpu_offload=true \
  trainer.gradient_checkpointing=true \
  trainer.use_liger_kernel=true \
  trainer.use_sample_packing=false \
  trainer.save_optimizer_state=false \
  trainer.placement.policy_num_gpus_per_node=${NUM_GPUS} \
  trainer.placement.ref_num_gpus_per_node=${NUM_GPUS} \
  generator.vision_language_generator=true \
  generator.inference_engine.num_engines=${NUM_GPUS} \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.gpu_memory_utilization=0.7 \
  generator.inference_engine.async_engine=true \
  generator.inference_engine.engine_init_kwargs.allowed_local_media_path=${SCRATCH_ROOT}/LLaMA-Factory-AutoReviewer \
  generator.inference_engine.engine_init_kwargs.disable_mm_preprocessor_cache=true \
  trainer.epochs=1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=${BS} \
  trainer.policy_mini_batch_size=${MINI} \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.max_prompt_length=${MAX_PROMPT_LEN} \
  generator.max_input_length=${MAX_PROMPT_LEN} \
  generator.sampling_params.max_generate_length=4096 \
  generator.batched=false \
  generator.use_conversation_multi_turn=true \
  generator.append_eos_token_after_stop_str_in_multi_turn=true \
  generator.n_samples_per_prompt=${SAMPLES} \
  generator.gather_mini_batch_size=${GATHER_MB} \
  generator.max_turns=1 \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  generator.sampling_params.stop='["</answer>"]' \
  environment.env_class=search_arxiv \
  environment.skyrl_gym.max_env_workers=16 \
  environment.skyrl_gym.search_arxiv.max_turns=1 \
  environment.skyrl_gym.search_arxiv.search_enabled=false \
  trainer.logger=console \
  trainer.project_name=train-prediction \
  trainer.run_name="$RUN_NAME" \
  trainer.ckpt_interval=999999 \
  trainer.hf_save_interval=999999 \
  trainer.max_ckpts_to_keep=1 \
  trainer.resume_mode=null \
  trainer.eval_before_train=false \
  trainer.eval_interval=999999 \
  trainer.gpu_keepalive_on_failure=false \
  trainer.ckpt_path="${SCRATCH_ROOT}/ckpts/${RUN_NAME}" \
  trainer.log_path="${SCRATCH_ROOT}/skyrl-logs" \
  "$@"

RC=$?

echo ""
echo "=============================================="
echo "rerun exited rc=${RC}"
echo "  outer keepalive will auto-resume via EXIT trap (rm of .keepalive.pause)"
echo "  to release the slurm allocation entirely:"
echo "    touch ${ORIG_CKPT}/.keepalive.release"
echo "=============================================="

exit $RC
