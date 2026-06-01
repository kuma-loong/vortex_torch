#!/usr/bin/env bash
# Chunk 3/5 of examples/run_minimax.sh — jobs [12..15] of 19:
#   [12] block_sparse_attention       topk=125  block=32  (budget 4096)
#   [13] block_sparse_attention       topk=29   block=64  (budget 2048)
#   [14] block_sparse_attention       topk=61   block=64  (budget 4096)
#   [15] gqa_quest_sparse_attention   topk=61   block=32  (budget 2048)
source "$(dirname "${BASH_SOURCE[0]}")/run_minimax_common.sh"
mm_run_chunk 12 4
