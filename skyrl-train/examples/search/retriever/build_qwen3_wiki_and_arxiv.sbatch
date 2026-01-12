#!/bin/bash
#SBATCH --job-name=qwen3_index
#SBATCH --partition=pli
#SBATCH --account=llm_explore
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/qwen3_index_%A_%a.out
#SBATCH --error=logs/qwen3_index_%A_%a.err

# Load modules and activate environment
module load anaconda3/2024.6
conda activate retriever

# CD to repo root
cd /scratch/gpfs/ZHUANGL/sk7524/SkyRLSearchEnvs/skyrl-train

# Environment setup
export HF_HOME=/scratch/gpfs/ZHUANGL/sk7524/hf
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Model path
QWEN3_MODEL_PATH=/scratch/gpfs/ZHUANGL/sk7524/hf/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/c54f2e6e80b2d7b7de06f51cec4959f6b3e03418

# Data paths
WIKI_CORPUS=/scratch/gpfs/ZHUANGL/sk7524/SkyRL/skyrl-train/data/searchr1_original/wiki/wiki-18.jsonl
WIKI_OUTPUT=/scratch/gpfs/ZHUANGL/sk7524/SkyRL/skyrl-train/data/searchr1_original/wiki/qwen3_06_embed

ARXIV_CORPUS=/scratch/gpfs/ZHUANGL/sk7524/SkyRL/skyrl-train/data/searchr1_original/arxiv/arxiv_wikiformat.jsonl
ARXIV_OUTPUT=/scratch/gpfs/ZHUANGL/sk7524/SkyRL/skyrl-train/data/searchr1_original/arxiv/qwen3_06_embed

# Select dataset based on array task ID
if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    echo "Building Qwen3 index for WIKI dataset..."
    CORPUS_PATH=$WIKI_CORPUS
    OUTPUT_DIR=$WIKI_OUTPUT
elif [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    echo "Building Qwen3 index for ARXIV dataset..."
    CORPUS_PATH=$ARXIV_CORPUS
    OUTPUT_DIR=$ARXIV_OUTPUT
else
    echo "Unknown array task ID: $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "  Model:  qwen3 ($QWEN3_MODEL_PATH)"
echo "  Corpus: $CORPUS_PATH"
echo "  Output: $OUTPUT_DIR"
echo "  Max Length: 256"
echo "  Batch Size: 384"

mkdir -p $OUTPUT_DIR

python examples/search/retriever/index_builder.py \
    --retrieval_method qwen3 \
    --model_path $QWEN3_MODEL_PATH \
    --corpus_path $CORPUS_PATH \
    --save_dir $OUTPUT_DIR \
    --use_fp16 \
    --max_length 256 \
    --batch_size 384 \
    --pooling_method last_token \
    --faiss_type Flat \
    --save_embedding \
    --faiss_gpu

echo "Done! Index saved to: $OUTPUT_DIR/qwen3_Flat.index"
