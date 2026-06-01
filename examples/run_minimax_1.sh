#!/usr/bin/env bash
# Chunk 1/5 of examples/run_minimax.sh — jobs [4..7] of 19:
#   [4] block_sparse_attention       topk=157  block=16
#   [5] block_sparse_attention       topk=253  block=16
#   [6] gqa_quest_sparse_attention   topk=61   block=16
#   [7] gqa_quest_sparse_attention   topk=93   block=16
source "$(dirname "${BASH_SOURCE[0]}")/run_minimax_common.sh"
mm_run_chunk 4 4
