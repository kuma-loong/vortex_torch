"""Pure-torch reference for ``topk_output`` — slow but obviously correct.

Per (req, kv_head) row, pick the top-``k`` block indices by score from
the middle ``[bos, num_blocks - eos)`` slice and write them into the
sparse output buffer. ``k`` is derived from the sparse indptr (CSR) or
sparse seqlens (trtllm) — these are pre-set by the planner so each row
gets exactly ``k + bos + eos`` slots. BOS / EOS positions are
pre-filled by the planner upstream; this op only writes the middle.
"""
from __future__ import annotations

import torch


def _pick_topk(scores: torch.Tensor, indices: torch.Tensor, k: int) -> torch.Tensor:
    """Return the ``min(k, len(scores))`` highest-scoring indices."""
    n = scores.shape[0]
    if n == 0 or k <= 0:
        return indices[:0]
    if k >= n:
        return indices
    _, top_pos = torch.topk(scores.float(), k=k)
    return indices[top_pos]


def reference_flashinfer(
    x,
    dense_kv_indptr,
    sparse_kv_indptr,
    dense_kv_indices,
    sparse_kv_indices,
    eff_batch_size,
    reserved_bos,
    reserved_eos,
    max_num_pages,
):
    """CSR top-k. ``sparse_kv_indices`` is written in-place over the middle slots."""
    for pid in range(eff_batch_size):
        d_lo = int(dense_kv_indptr[pid].item())
        d_hi = int(dense_kv_indptr[pid + 1].item())
        s_lo = int(sparse_kv_indptr[pid].item())
        s_hi = int(sparse_kv_indptr[pid + 1].item())
        sparse_count = s_hi - s_lo
        middle_k = sparse_count - reserved_bos - reserved_eos
        if middle_k <= 0:
            continue
        mid_lo = d_lo + reserved_bos
        mid_hi = d_hi - reserved_eos
        if mid_hi <= mid_lo:
            continue
        scores = x[mid_lo:mid_hi, 0, 0]
        indices = dense_kv_indices[mid_lo:mid_hi]
        picked = _pick_topk(scores, indices, middle_k)
        sparse_kv_indices[s_lo + reserved_bos : s_lo + reserved_bos + picked.shape[0]] = picked


def reference_trtllm(
    x,
    dense_seqlens,
    sparse_seqlens,
    dense_block_tables,
    sparse_block_tables,
    eff_batch_size,
    reserved_bos,
    reserved_eos,
    max_blocks_per_seq,
    block_size,
):
    """Block-table top-k. ``sparse_block_tables`` is written in-place over
    the middle slots [bos, bos + middle_k)."""
    for pid in range(eff_batch_size):
        d_tokens = int(dense_seqlens[pid].item())
        s_tokens = int(sparse_seqlens[pid].item())
        d_num_blocks = (d_tokens + block_size - 1) // block_size
        s_num_blocks = (s_tokens + block_size - 1) // block_size
        middle_k = s_num_blocks - reserved_bos - reserved_eos
        if middle_k <= 0:
            continue
        mid_lo = reserved_bos
        mid_hi = d_num_blocks - reserved_eos
        if mid_hi <= mid_lo:
            continue
        row_start = pid * max_blocks_per_seq
        scores = x[row_start + mid_lo : row_start + mid_hi, 0, 0]
        ids = dense_block_tables[pid, mid_lo:mid_hi]
        picked = _pick_topk(scores, ids, middle_k)
        sparse_block_tables[pid, reserved_bos : reserved_bos + picked.shape[0]] = picked
