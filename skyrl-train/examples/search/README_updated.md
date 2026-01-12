# SearchR1 Multi-Model Retrieval Setup

This document describes the updated retrieval system that supports multiple embedding models (E5 and Qwen3) with configurable scripts for different datasets.

## Overview

The retrieval system consists of:
1. **Index Builder** - Creates FAISS indexes from `.jsonl` corpus files
2. **Retrieval Server** - FastAPI server that handles search queries

### Supported Models

| Model | HuggingFace Path | Pooling Method | Query Prefix |
|-------|------------------|----------------|--------------|
| e5 | `intfloat/e5-base-v2` | mean | `query: {q}` |
| qwen3 | `Qwen/Qwen3-Embedding-0.6B` | last_token | `Instruct: {task}\nQuery:{q}` |

**Key Finding:** Qwen3-Embedding uses **last_token pooling** (not mean pooling like E5).

---

## Data Paths

```
Base directory: /scratch/gpfs/ZHUANGL/sk7524/SkyRL/skyrl-train/data/searchr1_original/wiki/

Files:
├── wiki-18.jsonl                    # Corpus (21M documents)
├── e5-index/
│   └── e5_Flat.index               # Pre-built E5 index
└── qwen3_06_embed/                  # Qwen3 index directory (to be created)
    └── qwen3_Flat.index            # Qwen3 index (after building)
```

---

## Scripts

### 1. Build Index (`build_index.sh`)

Creates a FAISS index from a corpus file.

**Usage:**
```bash
./examples/search/retriever/build_index.sh <model> <corpus_path> <output_dir>
```

**Arguments:**
- `model`: `e5` or `qwen3`
- `corpus_path`: Path to `.jsonl` corpus file
- `output_dir`: Directory to save the index

**Examples:**

```bash
# Build Qwen3 index for wiki-18
bash examples/search/retriever/build_index.sh qwen3 \
  /scratch/gpfs/ZHUANGL/sk7524/SkyRL/skyrl-train/data/searchr1_original/wiki/wiki-18.jsonl \
  /scratch/gpfs/ZHUANGL/sk7524/SkyRL/skyrl-train/data/searchr1_original/wiki/qwen3_06_embed

# Build Qwen3 index for a different dataset (e.g., arxiv)
bash examples/search/retriever/build_index.sh qwen3 \
  /path/to/arxiv.jsonl \
  /path/to/arxiv_qwen3_index

# Build E5 index (if needed)
bash examples/search/retriever/build_index.sh e5 \
  /path/to/corpus.jsonl \
  /path/to/e5_index
```

**Output:** Creates `{model}_Flat.index` in the output directory.

---

### 2. Launch Retrieval Server (`retrieval_launch.sh`)

Starts the FastAPI retrieval server.

**Usage:**
```bash
./examples/search/retriever/retrieval_launch.sh <model> <index_path> <corpus_path> [port]
```

**Arguments:**
- `model`: `e5` or `qwen3`
- `index_path`: Path to `.index` file
- `corpus_path`: Path to `.jsonl` corpus file
- `port`: Server port (default: 8000)

**Environment:** The script automatically sets up:
- `HF_HOME=/scratch/gpfs/ZHUANGL/sk7524/hf`
- `TRANSFORMERS_OFFLINE=1`
- Activates conda env at `/scratch/gpfs/ZHUANGL/sk7524/.conda/retriever`

**Examples:**

```bash
# Launch E5 server for wiki-18
bash examples/search/retriever/retrieval_launch.sh e5 \
  /scratch/gpfs/ZHUANGL/sk7524/SkyRL/skyrl-train/data/searchr1_original/wiki/e5-index/e5_Flat.index \
  /scratch/gpfs/ZHUANGL/sk7524/SkyRL/skyrl-train/data/searchr1_original/wiki/wiki-18.jsonl \
  8000

# Launch Qwen3 server for wiki-18
bash examples/search/retriever/retrieval_launch.sh qwen3 \
  /scratch/gpfs/ZHUANGL/sk7524/SkyRL/skyrl-train/data/searchr1_original/wiki/qwen3_06_embed/qwen3_Flat.index \
  /scratch/gpfs/ZHUANGL/sk7524/SkyRL/skyrl-train/data/searchr1_original/wiki/wiki-18.jsonl \
  8000

# Launch on a different port (e.g., for multiple servers)
bash examples/search/retriever/retrieval_launch.sh qwen3 \
  /path/to/arxiv_index/qwen3_Flat.index \
  /path/to/arxiv.jsonl \
  8001
```

---

## API Usage

Once the server is running, you can query it:

```bash
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of France?", "topk": 3, "return_scores": true}'
```

**Request format:**
```json
{
  "query": "Your search query",
  "topk": 3,
  "return_scores": true
}
```

**Response format:**
```json
{
  "result": [[
    {"document": {"id": "...", "contents": "..."}, "score": 0.85},
    {"document": {"id": "...", "contents": "..."}, "score": 0.82},
    {"document": {"id": "...", "contents": "..."}, "score": 0.79}
  ]]
}
```

---

## Corpus Format

The corpus must be a `.jsonl` file with each line containing:
```json
{"id": "0", "contents": "\"Title\"\nDocument text content..."}
```

---

## File Changes Summary

| File | Status | Description |
|------|--------|-------------|
| `retriever/index_builder.py` | **NEW** | Index building with Qwen3 support (last_token pooling) |
| `retriever/build_index.sh` | **NEW** | Configurable index building script |
| `retriever/retrieval_server.py` | **MODIFIED** | Added Qwen3 support, `--port` arg, auto pooling detection |
| `retriever/retrieval_launch.sh` | **MODIFIED** | Configurable launch script with env setup |

---

## Adding New Models

To add support for a new embedding model:

1. **Update `MODEL2POOLING`** in both `index_builder.py` and `retrieval_server.py`:
   ```python
   MODEL2POOLING = {
       "e5": "mean",
       "bge": "cls",
       "qwen": "last_token",
       "new_model": "mean",  # Add your model
   }
   ```

2. **Add query preprocessing** (if needed) in `retrieval_server.py` `Encoder.encode()`:
   ```python
   if "new_model" in self.model_name.lower():
       if is_query:
           query_list = [f"Your prefix: {query}" for query in query_list]
   ```

3. **Update shell scripts** to include the new model in the `case` statement.

---

## Troubleshooting

### Common Issues

1. **"Index file not found"**: Ensure the index has been built first using `build_index.sh`

2. **"Corpus file not found"**: Check the path to your `.jsonl` file

3. **CUDA out of memory**: Reduce `--batch_size` in `build_index.sh` or use fewer GPUs

4. **Model download issues**: Ensure `HF_HOME` is set and model is cached locally

### Checking Server Status

```bash
# Check if server is running
curl http://localhost:8000/docs  # OpenAPI docs

# Test a simple query
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "topk": 1}'
```
