#!/bin/bash
# Generate parquet dataset from ShareGPT JSON for arXiv paper acceptance prediction
#
# Usage: ./generate_dataset.sh [input_json] [output_dir] [train_ratio]
#
# Example:
#   ./generate_dataset.sh /path/to/data.json /output/dir 0.9

set -e

# Default paths (using SkyRL original repo paths for output)
INPUT_JSON="${1:-/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer/data/iclr_2020_2025_80_20_split5_balanced_deepreview_clean_binary_no_reviews_v3_train/data.json}"
OUTPUT_DIR="${2:-/scratch/gpfs/ZHUANGL/sk7524/SkyRL/skyrl-train/data/arxiv_iclr}"
TRAIN_RATIO="${3:-0.9}"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==========================================="
echo "ArXiv Dataset Generator"
echo "==========================================="
echo "Input JSON:  $INPUT_JSON"
echo "Output Dir:  $OUTPUT_DIR"
echo "Train Ratio: $TRAIN_RATIO"
echo "==========================================="

# Validate input file exists
if [ ! -f "$INPUT_JSON" ]; then
    echo "Error: Input file not found: $INPUT_JSON"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the dataset generator
python "$SCRIPT_DIR/searchr1_arxiv_dataset.py" \
    --input_json "$INPUT_JSON" \
    --output_dir "$OUTPUT_DIR" \
    --train_ratio "$TRAIN_RATIO" \
    --shuffle \
    --seed 42

echo ""
echo "Dataset generation complete!"
echo "Output files:"
ls -la "$OUTPUT_DIR"/*.parquet 2>/dev/null || echo "No parquet files found"
