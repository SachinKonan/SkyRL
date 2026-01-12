#!/bin/bash
# Rebuild arxiv FAISS index from existing embeddings

set -e

# Load modules and activate environment
module load anaconda3/2024.6
conda activate retriever

# CD to repo root
cd /scratch/gpfs/ZHUANGL/sk7524/SkyRLSearchEnvs/skyrl-train

# Environment setup
export HF_HOME=/scratch/gpfs/ZHUANGL/sk7524/hf
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Paths
QWEN3_MODEL_PATH=/scratch/gpfs/ZHUANGL/sk7524/hf/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/c54f2e6e80b2d7b7de06f51cec4959f6b3e03418
ARXIV_CORPUS=/scratch/gpfs/ZHUANGL/sk7524/SkyRL/skyrl-train/data/searchr1_original/arxiv/arxiv_wikiformat.jsonl
ARXIV_OUTPUT=/scratch/gpfs/ZHUANGL/sk7524/SkyRL/skyrl-train/data/searchr1_original/arxiv/qwen3_06_embed
EMBEDDING_PATH=/scratch/gpfs/ZHUANGL/sk7524/SkyRL/skyrl-train/data/searchr1_original/arxiv/qwen3_06_embed/emb_qwen3.memmap

echo "Rebuilding FAISS index from existing embeddings..."
echo "  Corpus: $ARXIV_CORPUS"
echo "  Embeddings: $EMBEDDING_PATH"
echo "  Output: $ARXIV_OUTPUT"

python examples/search/retriever/index_builder.py \
    --retrieval_method qwen3 \
    --model_path $QWEN3_MODEL_PATH \
    --corpus_path $ARXIV_CORPUS \
    --save_dir $ARXIV_OUTPUT \
    --embedding_path $EMBEDDING_PATH \
    --pooling_method last_token \
    --faiss_type Flat \
    --faiss_gpu

echo "Done! Index saved to: $ARXIV_OUTPUT/qwen3_Flat.index"
