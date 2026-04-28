#!/bin/bash
# Submit per-model SLURM arrays for the 4-way eval comparison, chained
# sequentially via --dependency=afterany so they run one after the other
# (avoids 28-way contention on gpu-test, isolates failure modes).
#
# Usage: bash sbatch/sft/eval_7b_all.sh [base|step206|step412|vero|all]
#
# Each array is 7 tasks (one per chunk). 4 arrays × 7 tasks = 28 evals.
# Per-cell: 1h gpu-test, 2× A100 80GB.

set -euo pipefail

SCRATCH=${SCRATCH_ROOT:-/scratch/gpfs/ZHUANGL/sk7524}
TEXT_CHUNKS=${SCRATCH}/SkyRLMain/data/iclr_arxiv_text_search_2526_test_chunks
VL_CHUNKS=${SCRATCH}/SkyRLMain/data/iclr_arxiv_vl_2526_test_chunks
SFT_ROOT=${SCRATCH}/ckpts/sft_7b_array_7343633/exports
BASE=${SCRATCH}/hf/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
VERO=${SCRATCH}/hf/hub/models--zlab-princeton--Vero-Qwen25-7B/snapshots/180e84be5acb2aa887cf51015b84b6a6e453ee90

DEP=""
DEP_PREV=""

chain() {  # tag, sbatch, data_base, model
    local tag=$1 sb=$2 base=$3 model=$4
    if [ ! -e "$model" ]; then
        echo "skip $tag — model not found: $model"
        return
    fi
    if [ ! -d "$base/chunk_0" ]; then
        echo "skip $tag — chunk root not found: $base"
        return
    fi
    local args=(--parsable
                --export=ALL,MODEL_PATH=$model,DATA_DIR_BASE=$base,RUN_NAME_PREFIX=gen_eval_$tag)
    if [ -n "$DEP" ]; then
        args+=(--dependency=afterany:$DEP)
    fi
    DEP=$(sbatch "${args[@]}" "$sb")
    echo "submitted $tag -> array $DEP (depends on ${DEP_PREV:-none})"
    DEP_PREV=$DEP
}

case ${1:-all} in
    base)
        chain base sbatch/sft/gen_eval_text_array.sbatch "$TEXT_CHUNKS" "$BASE"
        ;;
    step206)
        chain step206 sbatch/sft/gen_eval_text_array.sbatch "$TEXT_CHUNKS" "$SFT_ROOT/global_step_206/policy"
        ;;
    step412)
        chain step412 sbatch/sft/gen_eval_text_array.sbatch "$TEXT_CHUNKS" "$SFT_ROOT/global_step_412/policy"
        ;;
    vero)
        chain vero sbatch/sft/gen_eval_vl_array.sbatch "$VL_CHUNKS" "$VERO"
        ;;
    all)
        chain base    sbatch/sft/gen_eval_text_array.sbatch "$TEXT_CHUNKS" "$BASE"
        chain step206 sbatch/sft/gen_eval_text_array.sbatch "$TEXT_CHUNKS" "$SFT_ROOT/global_step_206/policy"
        chain step412 sbatch/sft/gen_eval_text_array.sbatch "$TEXT_CHUNKS" "$SFT_ROOT/global_step_412/policy"
        chain vero    sbatch/sft/gen_eval_vl_array.sbatch   "$VL_CHUNKS"   "$VERO"
        ;;
    *)
        echo "usage: $0 [base|step206|step412|vero|all]"
        exit 2
        ;;
esac
