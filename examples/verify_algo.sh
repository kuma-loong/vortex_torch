#!/usr/bin/env bash
set -e

sparse_algos=(

)

for algo in "${sparse_algos[@]}"; do
  echo ">>> Running verify_algo.py with --vortex-module-name ${algo}"
  python examples/verify_algo.py \
    --trials 8 \
    --topk-val 30 \
    --vortex-module-name "${algo}" \
    --model-name Qwen/Qwen3-1.7B \
    --mem 0.7
done

