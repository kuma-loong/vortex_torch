"""Pure-torch reference for ``TopK(k)`` — slow but obviously correct.

Trtllm-only. Per row, pick the top-``topk_val`` blocks by score from
the middle, then **restore** the ``bos`` head blocks at slot 0 and the
``eos`` tail blocks at the very end, and write the per-row sparse
``seqlens`` in tokens (respecting the dense tail's partial fill).

Unlike :mod:`topk_output._reference`, this op writes both
``sparse_block_tables`` **and** ``sparse_seqlens``. Row-too-small
fallback (``num_blocks <= bos + eos + topk_val``): copy the entire
dense row through.
"""
from __future__ import annotations

import torch


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
    topk_val,
):
    """Write ``sparse_block_tables[pid, :]`` and ``sparse_seqlens[pid]``
    in place."""
    for pid in range(eff_batch_size):
        d_tokens = int(dense_seqlens[pid].item())
        if d_tokens <= 0:
            sparse_seqlens[pid] = 0
            continue
        d_num_blocks = (d_tokens + block_size - 1) // block_size
        last_mod = d_tokens % block_size
        last_block_tokens = block_size if last_mod == 0 else last_mod
        dense_row = dense_block_tables[pid, :d_num_blocks]

        # Row-too-small: copy entire dense row through.
        if d_num_blocks <= reserved_bos + reserved_eos + topk_val:
            sparse_block_tables[pid, :d_num_blocks] = dense_row
            sparse_seqlens[pid] = d_tokens
            continue

        # Pick top-topk_val by score from the middle range.
        row_start = pid * max_blocks_per_seq
        mid_lo = reserved_bos
        mid_hi = d_num_blocks - reserved_eos
        scores = x[row_start + mid_lo : row_start + mid_hi, 0, 0]
        mid_ids = dense_row[mid_lo:mid_hi]
        _, top_pos = torch.topk(scores.float(), k=topk_val)
        # Stable: sort top positions ascending (matches kernel's index-sorted output).
        top_pos_sorted, _ = top_pos.sort()
        picked_mid = mid_ids[top_pos_sorted]

        # Layout: [bos head blocks] [picked middle] [eos tail blocks]
        out_blocks = (
            list(dense_row[:reserved_bos].cpu().tolist())
            + list(picked_mid.cpu().tolist())
            + list(dense_row[d_num_blocks - reserved_eos : d_num_blocks].cpu().tolist())
        )
        n_out = len(out_blocks)
        sparse_block_tables[pid, :n_out] = torch.tensor(
            out_blocks, dtype=sparse_block_tables.dtype, device=sparse_block_tables.device
        )
        # Seqlens: every block but the tail one is full; the tail keeps the
        # dense partial-fill count.
        sparse_seqlens[pid] = (n_out - 1) * block_size + last_block_tokens
