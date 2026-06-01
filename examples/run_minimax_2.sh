#!/usr/bin/env bash
# Chunk 2/5 of examples/run_minimax.sh — jobs [8..11] of 19:
#   [8]  gqa_quest_sparse_attention   topk=125  block=16
#   [9]  gqa_quest_sparse_attention   topk=157  block=16
#   [10] gqa_quest_sparse_attention   topk=253  block=16
#   [11] block_sparse_attention       topk=61   block=32  (budget 2048)
source "$(dirname "${BASH_SOURCE[0]}")/run_minimax_common.sh"
mm_run_chunk 8 4
