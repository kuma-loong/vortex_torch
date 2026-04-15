#!/usr/bin/env bash
set -e
sparse_algos=(
full_attention
block_sparse_attention
gqa_quest_sparse_attention
)

models=(
Qwen/Qwen3-0.6B
Qwen/Qwen3-1.7B
Qwen/Qwen3-4B
)

for algo in "${sparse_algos[@]}"; do
  for model in "${models[@]}"; do
  echo ">>> Running verify_algo.py with --vortex-module-name ${algo} and --model-name ${model}"
    python examples/verify_algo.py \
        --trials 8 \
        --topk-val 29 \
        --page-size 64 \
        --workload-chunk-size 32 \
        --block-size 16 \
        --vortex-module-name "${algo}" \
        --model-name  "${model}" \
        --mem 0.85 \
        --data-path examples/aime24.jsonl \
        --generation-max-new-tokens 16384 \
        --max-input-length 4096
    done
done


  


rm -rf ~/.vortex_compilation_cache