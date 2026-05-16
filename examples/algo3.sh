#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3
set -e
sparse_algos=(
block_sparse_attention
)

models=(
Qwen/Qwen3-4B
)
trials=(
32
)
topk_val=(
61
125
253
)
workload_chunk_sizes=(
32
64
128
)
for algo in "${sparse_algos[@]}"; do
  for model in "${models[@]}"; do
    for trial in "${trials[@]}"; do
      for k_val in "${topk_val[@]}"; do
        for wcs in "${workload_chunk_sizes[@]}"; do
          echo ">>> Running verify_algo.py with --vortex-module-name ${algo} and --model-name ${model} for ${trial} trials, workload-chunk-size=${wcs}"
          python examples/verify_algo.py \
              --trials ${trial} \
              --topk-val ${k_val} \
              --page-size 16 \
              --workload-chunk-size ${wcs} \
              --block-size 16 \
              --topk-ratio 0.00 \
              --vortex-module-name "${algo}" \
              --model-name  "${model}" \
              --mem 0.9 \
              --data-path examples/aime24.jsonl \
              --generation-max-new-tokens 16384 \
              --max-input-length 4096 \
              --tp-size 1 \
              --summary-dir summary-Qwen3-4B-sglang-flashinfer \
              --skip-already-finished-check
        done
      done
    done
  done
done