#!/usr/bin/env bash
set -e

sparse_algos=(
full_attention
)

for algo in "${sparse_algos[@]}"; do
  echo ">>> Running verify_algo.py with --vortex-module-name ${algo}"
  python examples/verify_algo.py \
    --trials 8 \
    --topk-val 29 \
    --page-size 128 \
    --workload-chunk-size 16 \
    --block-size 16 \
    --vortex-module-name "${algo}" \
    --model-name Qwen/Qwen3-1.7B \
    --mem 0.9 \
    --data-path examples/aime24.jsonl \
    --generation-max-new-tokens 16384 \
    --max-input-length 4096
done

rm -rf ~/.vortex_compilation_cache