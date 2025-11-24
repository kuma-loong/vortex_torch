#!/usr/bin/env bash
set -e

sparse_algos=(
  "gqa_block_sparse_attention"
  "gqa_quest_sparse_attention"
  "block_sparse_attention"
)

for algo in "${sparse_algos[@]}"; do
  echo ">>> Running verify_algo.py with --vortex-module-name ${algo}"
  python examples/verify_algo.py \
    --trials 8 \
    --topk-val 30 \
    --vortex-module-name "${algo}" \
    --model-name Qwen/Qwen3-1.7B
done

