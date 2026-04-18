#!/bin/bash
# Qwen2.5-VL-3B with 4-turn arXiv search. Uses the VLM entrypoint so image
# observations flow through the multi-modal generator.
#
#SBATCH --job-name=tp_vl
#SBATCH --time=30:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=400G
#SBATCH --gres=gpu:4
#SBATCH --constraint="gpu80"
#SBATCH --partition=pli
#SBATCH --output=logs/train_prediction/vl_%j.out
#SBATCH --error=logs/train_prediction/vl_%j.err

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$WORKDIR"
mkdir -p logs/train_prediction

source "$WORKDIR/sbatch/train_prediction/_common.sh"

DATA_DIR="${DATA_DIR_VL:-/scratch/gpfs/ZHUANGL/sk7524/SkyRLMain/data/iclr_arxiv_vl}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}"
RUN_NAME="${RUN_NAME:-tp_qwen2.5_vl_3b_4step_search}"

echo "=============================================="
echo "Job:          $SLURM_JOB_ID"
echo "Host:         $(hostname)"
echo "Data:         $DATA_DIR"
echo "Model:        $MODEL_PATH"
echo "EMBED_SIZE:   $EMBED_SIZE"
echo "Retrieval:    $RETRIEVAL_URL"
echo "Run name:     $RUN_NAME"
echo "=============================================="

trap stop_retriever EXIT
start_retriever

# VL path uses its own entrypoint that swaps in SkyRLVLMGymGenerator.
uv run --isolated --frozen --extra fsdp python examples/train/search_arxiv/main_vlm.py \
  data.train_data="['${DATA_DIR}/train.parquet']" \
  data.val_data="['${DATA_DIR}/validation.parquet']" \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.policy.optimizer_config.max_grad_norm=0.5 \
  trainer.policy.optimizer_config.num_warmup_steps=40 \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.kl_loss_coef=0.001 \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.policy.fsdp_config.cpu_offload=false \
  trainer.ref.fsdp_config.cpu_offload=true \
  trainer.placement.policy_num_gpus_per_node=4 \
  trainer.placement.ref_num_gpus_per_node=4 \
  generator.inference_engine.num_engines=2 \
  generator.inference_engine.tensor_parallel_size=2 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.gpu_memory_utilization=0.5 \
  trainer.epochs=1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=64 \
  trainer.policy_mini_batch_size=32 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.max_prompt_length=24480 \
  generator.max_input_length=25600 \
  generator.sampling_params.max_generate_length=500 \
  generator.inference_engine.async_engine=true \
  generator.batched=false \
  generator.use_conversation_multi_turn=true \
  generator.append_eos_token_after_stop_str_in_multi_turn=true \
  generator.n_samples_per_prompt=4 \
  generator.max_turns=4 \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  generator.sampling_params.stop='["</ssearch>", "</asearch>", "</answer>"]' \
  environment.env_class=search_arxiv \
  environment.skyrl_gym.max_env_workers=8 \
  environment.skyrl_gym.search_arxiv.log_requests=false \
  environment.skyrl_gym.search_arxiv.search_url="$RETRIEVAL_URL" \
  environment.skyrl_gym.search_arxiv.topk=5 \
  environment.skyrl_gym.search_arxiv.max_turns=4 \
  environment.skyrl_gym.search_arxiv.search_enabled=true \
  trainer.logger=wandb \
  trainer.project_name=train-prediction \
  trainer.run_name="$RUN_NAME" \
  trainer.ckpt_interval=20 \
  trainer.hf_save_interval=100 \
  trainer.max_ckpts_to_keep=3 \
  trainer.resume_mode=latest \
  trainer.ckpt_path="$HOME/ckpts/${RUN_NAME}" \
  trainer.eval_batch_size=64 \
  trainer.eval_before_train=false \
  generator.eval_sampling_params.temperature=0 \
  generator.eval_sampling_params.stop='["</ssearch>", "</asearch>", "</answer>"]' \
  generator.eval_sampling_params.max_generate_length=500 \
  trainer.export_path="$HOME/ckpts/${RUN_NAME}/exports" \
  trainer.eval_interval=50 \
  "$@"
