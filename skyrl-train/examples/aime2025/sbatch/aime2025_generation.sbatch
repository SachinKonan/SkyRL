#!/bin/bash
#SBATCH --job-name=aime2025_gen_qwen3_4b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --time=00:20:00
#SBATCH --output=logs/aime2025/%j.out
#SBATCH --error=logs/aime2025/%j.err

module load proxy/default

# Print job information
echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Change to the correct directory
cd /scratch/gpfs/ZHUANGL/sk7524/SkyRL/skyrl-train
echo "Working directory: $(pwd)"

# Activate virtual environment
source .venv/bin/activate
echo "Activated virtual environment"

# Set offline mode for transformers (models are cached locally)
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/scratch/gpfs/ZHUANGL/sk7524/hf
export WANDB_API_KEY="a9c6eb313dc5aab38bec7680526824ccdbb7f5f0"

# Memory optimization for PyTorch (reduce fragmentation)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Check GPU availability
nvidia-smi

echo "=========================================="
echo "AIME2025 Generation-Only Evaluation"
echo "Model: Qwen/Qwen3-4B-Instruct-2507"
echo "Data: AIME2025 (30 problems)"
echo "Backend: vLLM"
echo "GPUs: 1"
echo "=========================================="

# Paths
MODEL_PATH="/scratch/gpfs/ZHUANGL/sk7524/hf/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"
DATA_DIR="/scratch/gpfs/ZHUANGL/sk7524/SkyRL/skyrl-train/data/aime2025"
EXPORT_PATH="/scratch/gpfs/ZHUANGL/sk7524/SkyRL/skyrl-train/exports/aime2025/${SLURM_JOB_ID}"

# Run generation-only evaluation (using pre-installed venv to avoid network issues)
python -m skyrl_train.entrypoints.main_generate \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.logger="wandb" \
  trainer.project_name="aime2025" \
  trainer.run_name="aime2025_gen_qwen3_4b_${SLURM_JOB_ID}" \
  trainer.placement.colocate_all=false \
  trainer.export_path="$EXPORT_PATH" \
  trainer.dump_eval_results=true \
  generator.backend=vllm \
  generator.num_inference_engines=1 \
  generator.inference_engine_tensor_parallel_size=1 \
  generator.gpu_memory_utilization=0.9 \
  generator.eval_sampling_params.max_generate_length=8192 \
  generator.eval_sampling_params.temperature=0.7 \
  environment.env_class=aime \
  "$@"

if [ $? -eq 0 ]; then
    echo "Generation completed successfully"
else
    echo "Generation failed"
fi

echo "End time: $(date)"
