"""Pure-torch reference for ``Union()`` — slow but obviously correct.

Trtllm-only. Mirrors the per-row contract documented on
``vortex_torch.indexer.output_func.Union`` and the union CUDA kernels:

  * For each row, dedupe
    ``block_tables_0[bx, :blocks_0] ∪ block_tables_1[bx, :blocks_1]``
    (excluding the dense row's last block id), then append
    ``last_block_id`` at the tail.
  * Write ``sparse_seqlens[bx] = u * block_size + last_block_len`` where
    ``u`` is the dedup count and ``last_block_len`` is the partial token
    count of the dense path's trailing block.

Returns the same ``(sparse_block_tables, sparse_seqlens)`` tensors as
the CUDA kernels. Slow O(eff_bs * max_blocks_per_seq^2) Python loop —
do not call on the hot path.

Ported from ``vortex_torch/kernels/topk/benchmark_union.py``.
"""
from __future__ import annotations

import torch


def reference_trtllm(
    dense_seqlens,
    sparse_seqlens,
    dense_block_tables,
    block_tables_0,
    seqlens_0,
    block_tables_1,
    seqlens_1,
    sparse_block_tables,
    eff_batch_size,
    max_blocks_per_seq,
    block_size,
):
    """Write ``sparse_block_tables`` and ``sparse_seqlens`` in place."""
    device = dense_seqlens.device

    ds = dense_seqlens.cpu().tolist()
    dt = dense_block_tables.cpu().tolist()
    bt0 = block_tables_0.cpu().tolist()
    sl0 = seqlens_0.cpu().tolist()
    bt1 = block_tables_1.cpu().tolist()
    sl1 = seqlens_1.cpu().tolist()

    bt_out = [[0] * max_blocks_per_seq for _ in range(eff_batch_size)]
    sl_out = [0] * eff_batch_size

    for bx in range(eff_batch_size):
        tokens = ds[bx]
        if tokens <= 0:
            sl_out[bx] = 0
            continue
        dense_block_len = (tokens + block_size - 1) // block_size
        last_mod = tokens % block_size
        last_block_len = block_size if last_mod == 0 else last_mod
        last_block_id = dt[bx][dense_block_len - 1]

        blocks0 = (sl0[bx] + block_size - 1) // block_size
        blocks1 = (sl1[bx] + block_size - 1) // block_size

        seen = []
        seen_set = set()
        for src in (bt0[bx][:blocks0], bt1[bx][:blocks1]):
            for v in src:
                if v == last_block_id or v in seen_set:
                    continue
                seen_set.add(v)
                seen.append(v)
        for i, v in enumerate(seen):
            bt_out[bx][i] = v
        u = len(seen)
        bt_out[bx][u] = last_block_id
        sl_out[bx] = u * block_size + last_block_len

    sparse_block_tables.copy_(
        torch.tensor(bt_out, dtype=sparse_block_tables.dtype, device=device)
    )
    sparse_seqlens.copy_(
        torch.tensor(sl_out, dtype=sparse_seqlens.dtype, device=device)
    )
