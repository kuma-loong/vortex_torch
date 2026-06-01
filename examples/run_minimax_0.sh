#!/usr/bin/env bash
# Chunk 0/5 of examples/run_minimax.sh — jobs [0..3] of 19:
#   [0] dense baseline (full_attention, trtllm_mha, block=page=32)
#   [1] block_sparse_attention  topk=61   block=16
#   [2] block_sparse_attention  topk=93   block=16
#   [3] block_sparse_attention  topk=125  block=16
#
# Same usage as run_minimax.sh:
#   ./run_minimax_0.sh                     # default model
#   ./run_minimax_0.sh <hf-model-id>       # override model
#   GPUS="0 1 2 3 4 5 6 7" ./run_minimax_0.sh
source "$(dirname "${BASH_SOURCE[0]}")/run_minimax_common.sh"
mm_run_chunk 0 4
