#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=4,7
export HF_HOME=/raid/catalyst/models/
set -e
sparse_algos=(
block_sparse_attention
)

models=(
MiniMaxAI/MiniMax-M2.7
)
trials=(
16
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
            --page-size 32 \
            --workload-chunk-size 64 \
            --block-size 32 \
            --topk-ratio 0.00 \
            --vortex-module-name "${algo}" \
            --model-name  "${model}" \
            --mem 0.9 \
            --data-path examples/aime26_minimax.jsonl \
            --generation-max-new-tokens 32768 \
            --max-input-length 4096 \
            --tp-size 2 \
            --vortex-attention-backend trtllm \
            --vortex-impl-backend triton \
            --vortex-use-tensor-core \
            --vortex-layers-skip \
            --summary-dir summary-MiniMax-M2.7-sglang-trtllm \
            --skip-already-finished-check
      done
    done
  done
done