#!/usr/bin/env bash
set -e
sparse_algos=(
block_sparse_attention
)

models=(
Qwen/Qwen3-4B
)
trials=(
8
)
topk_val=(
61
)

for algo in "${sparse_algos[@]}"; do
  for model in "${models[@]}"; do
    for trial in "${trials[@]}"; do
      for k_val in "${topk_val[@]}"; do
        echo ">>> Running verify_algo.py with --vortex-module-name ${algo} and --model-name ${model} for ${trial} trials"
        python examples/verify_algo.py \
            --trials ${trial} \
            --topk-val ${k_val} \
            --page-size 16 \
            --workload-chunk-size 64 \
            --block-size 16 \
            --vortex-module-name "${algo}" \
            --model-name  "${model}" \
            --mem 0.9 \
            --data-path examples/aime24.jsonl \
            --generation-max-new-tokens 16384 \
            --max-input-length 4096
      done
    done
  done
done


  


rm -rf ~/.vortex_compilation_cache