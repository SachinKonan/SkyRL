# search_arxiv — ICLR accept/reject prediction with arXiv retrieval

Environment `search_arxiv` (registered in `skyrl-gym`) that lets the model query an
arXiv FAISS retrieval server via `<ssearch>` and `<asearch>` tags before emitting
`<answer>Accept|Reject</answer>`. Reward is exact-match of the answer against the
ICLR decision in the dataset.

## Prepare the dataset

Converts the LLaMA-Factory ICLR text JSONs into SkyRL parquet format:

```bash
python examples/train/search_arxiv/searchr1_arxiv_dataset.py \
    --input_base /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer/data/iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered \
    --output_dir /scratch/gpfs/ZHUANGL/sk7524/SkyRLMain/data/iclr_arxiv_text \
    --splits train validation test
```

Outputs `{train,validation,test}.parquet`.

## Retrieval server

The env hits a FastAPI server shipped by the separate `arxiv_retriever` repo at
`/scratch/gpfs/ZHUANGL/sk7524/arxiv_retriever/`. Two embedding models are
available via yaml config:

- `configs/retrieval/qwen3_06b.yaml` — Qwen3-Embedding-0.6B (1 GPU)
- `configs/retrieval/qwen3_4b.yaml`  — Qwen3-Embedding-4B   (2 GPUs, slower, stronger)

FAISS-GPU requires a conda env; see `../arxiv_retriever/conda-env.yml`. Launch
pattern (also used in `sbatch/test_retrieval/retrieval_test.sbatch`):

```bash
cd /scratch/gpfs/ZHUANGL/sk7524/arxiv_retriever
conda activate <retriever_env>
python src/arxiv_retriever/server/retrieval_server.py \
    --config configs/retrieval/qwen3_06b.yaml \
    "server.port=8000"
```

## Point the env at the server

Add Hydra overrides when launching training:

```
environment.env_class=search_arxiv
+environment.skyrl_gym.search_arxiv.search_url="http://127.0.0.1:8000/retrieve"
+environment.skyrl_gym.search_arxiv.topk=5
+environment.skyrl_gym.search_arxiv.max_turns=4
```

For single-step (no search) training, set `max_turns=1` and `search_enabled=false`.

## Quick retrieval smoke test

```
sbatch sbatch/test_retrieval/retrieval_test.sbatch                 # default: EMBED_SIZE=06b
EMBED_SIZE=4b sbatch sbatch/test_retrieval/retrieval_test.sbatch
```
