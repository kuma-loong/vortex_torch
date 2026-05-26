#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
set -e
sparse_algos=(
block_sparse_attention
)

models=(
Qwen/Qwen3-8B
)
trials=(
32
)
topk_val=(
125
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
            --topk-ratio 0.00 \
            --vortex-module-name "${algo}" \
            --model-name  "${model}" \
            --mem 0.9 \
            --data-path examples/aime26.jsonl \
            --generation-max-new-tokens 28672 \
            --max-input-length 4096 \
            --tp-size 1 \
            --vortex-attention-backend trtllm \
            --vortex-impl-backend triton \
            --vortex-use-tensor-core \
            --vortex-layers-skip \
            --summary-dir summary-Qwen3-8B \
            --skip-already-finished-check
      done
    done
  done
done
