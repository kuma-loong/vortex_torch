#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=7
export HF_HOME=/raid/catalyst/models/
set -e
sparse_algos=(
full_attention
)

models=(
Qwen/Qwen3-30B-A3B-FP8
)
trials=(
32
)
topk_val=(
253
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
            --data-path examples/aime24.jsonl \
            --generation-max-new-tokens 32768 \
            --max-input-length 4096 \
            --tp-size 1 \
            --vortex-attention-backend trtllm \
            --vortex-impl-backend triton \
            --vortex-use-tensor-core \
            --summary-dir summary-Qwen3-4B-sglang-trtllm \
            --skip-already-finished-check
      done
    done
  done
done