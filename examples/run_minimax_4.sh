#!/usr/bin/env bash
# Chunk 4/5 of examples/run_minimax.sh — jobs [16..18] of 19 (last chunk, 3 jobs):
#   [16] gqa_quest_sparse_attention   topk=125  block=32  (budget 4096)
#   [17] gqa_quest_sparse_attention   topk=29   block=64  (budget 2048)
#   [18] gqa_quest_sparse_attention   topk=61   block=64  (budget 4096)
source "$(dirname "${BASH_SOURCE[0]}")/run_minimax_common.sh"
mm_run_chunk 16 3
