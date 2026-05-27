#!/usr/bin/env bash
# Experiment driver for GLM-4.7-Flash MLA sparse attention (AIME26, mean@16).
#
# Matrix:
#   1. full_attention  — dense baseline, run ONCE (topk is ignored). Uses the
#      dense flashinfer MLA path because the vortex sparse backends require
#      enable_vortex_sparsity=True (so full attention can't use cuda_mla).
#   2. rope_aware_block_sparse_mla  } cuda_mla decode + Triton tensor-core indexer,
#   3. lserve_centroid_mla          } NO layer skip (all layers sparse), topk sweep.
#
# Each sparse run uses:
#   --attention-backend cuda_mla     hand-CUDA block-table decode (geometry-agnostic)
#   --vortex-impl-backend triton     tensor-core is a Triton-indexer feature ...
#   --vortex-use-tensor-core         ... and requires impl-backend=triton
#   --vortex-layers-skip             (no values) => skip none, all layers sparse
#
# Results land in $SUMMARY_DIR/*.json; collect_algo2_results.py renders them to
# examples/algo2_results.md (with the raw summaries embedded).
#
# NOTE on GPUs: 0 and 2 are broken on this host; 6/7 are often taken by other
# users. Override CUDA_VISIBLE_DEVICES to a free, working GPU before running.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export HF_HOME=/raid/catalyst/models/

MODEL="zai-org/GLM-4.7-Flash"
DATA="examples/aime26_glm.jsonl"
SUMMARY_DIR="summary-glm4.7-flash"
TRIALS=16
TOPK_VAL=(61 93 125 157 253)
SPARSE_MODULES=(rope_aware_block_sparse_mla lserve_centroid_mla)

COMMON=(
  --trials "$TRIALS"
  --page-size 16 --block-size 16 --workload-chunk-size 64 --topk-ratio 0.00
  --model-name "$MODEL" --data-path "$DATA" --mem 0.9
  --generation-max-new-tokens 32768 --max-input-length 4096 --tp-size 1
  --summary-dir "$SUMMARY_DIR" --skip-already-finished-check
)

run() {  # one config; do not abort the whole sweep if a single run fails
  echo ">>> $*"
  python examples/verify_algo.py "${COMMON[@]}" "$@" || echo "!!! FAILED: $*"
}

# --- 1. full-attention dense baseline (run once; topk ignored) ----------------
# Use sglang's plain Triton backend for dense: the flashinfer dense-MLA path has
# an illegal-memory-access / page-index bug on this pool (see marks/mla/progress.md).
# block/page=32 (dense isn't compatible with block_size=16).
run --vortex-module-name full_attention --attention-backend triton \
    --topk-val 253 --block-size 32 --page-size 32

# --- 2/3. sparse modules: cuda_mla + tensor-core indexer, all layers sparse ---
for algo in "${SPARSE_MODULES[@]}"; do
  for k in "${TOPK_VAL[@]}"; do
    run --vortex-module-name "$algo" --topk-val "$k" \
        --attention-backend cuda_mla \
        --vortex-impl-backend triton \
        --vortex-use-tensor-core \
        --vortex-layers-skip            # MUST be last: no values => skip none
  done
done

# --- render results markdown --------------------------------------------------
python examples/collect_algo2_results.py --summary-dir "$SUMMARY_DIR" --out examples/algo2_results.md
echo "=== done; see examples/algo2_results.md ==="
