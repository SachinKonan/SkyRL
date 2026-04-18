# Retriever

This directory is intentionally thin. The arXiv retrieval server lives in the
separate `arxiv_retriever` repo:

    /scratch/gpfs/ZHUANGL/sk7524/arxiv_retriever/

To launch it, see the top-level `examples/train/search_arxiv/README.md` and the
reusable launch snippet in `sbatch/test_retrieval/retrieval_test.sbatch` at the
repo root.

The SkyRL env only hits the `/retrieve` HTTP endpoint — it does not import
`arxiv_retriever` directly. If you want to import the Python client (e.g.
`arxiv_retriever.clients.arxiv_client.ArxivClient`) from a notebook, install the
optional extra:

    uv sync --extra search-arxiv
