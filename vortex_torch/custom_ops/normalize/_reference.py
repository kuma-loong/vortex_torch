"""Pure-torch reference for ``normalize`` — slow but obviously correct.

The kernel does **L2** normalize (sqrt of sum of squares per ``(D0, D1)``
lane), not L1. Per row, the middle ``[bos, num_blocks - eos)`` blocks
are divided by the L2 norm computed across that segment. The first
``bos`` and last ``eos`` blocks are passed through unchanged. Rows with
``num_blocks <= bos + eos + topk_val`` are skipped entirely (no write).
"""
from __future__ import annotations

import torch

_EPS = 1e-12


def _normalize_row(x_seg: torch.Tensor, out_seg: torch.Tensor):
    if x_seg.shape[0] == 0:
        return
    xf = x_seg.float()
    norm = (xf * xf).sum(dim=0, keepdim=True).sqrt().clamp_min(_EPS)
    out_seg.copy_((xf / norm).to(out_seg.dtype))


def reference_flashinfer(
    x, out, indptr, bos, eos, topk_val, D0, D1, eff_batch_size,
):
    threshold = bos + eos + topk_val
    for pid in range(eff_batch_size):
        start = int(indptr[pid].item())
        end = int(indptr[pid + 1].item())
        num_blocks = end - start
        if num_blocks <= threshold:
            continue
        mid_lo = start + bos
        mid_hi = end - eos
        if mid_hi <= mid_lo:
            continue
        _normalize_row(x[mid_lo:mid_hi], out[mid_lo:mid_hi])


def reference_trtllm(
    x, out, seqlens, bos, eos, topk_val, D0, D1,
    block_size, max_blocks_per_seq, eff_batch_size,
):
    threshold = bos + eos + topk_val
    for pid in range(eff_batch_size):
        tokens = int(seqlens[pid].item())
        num_blocks = (tokens + block_size - 1) // block_size
        if num_blocks <= threshold:
            continue
        start = pid * max_blocks_per_seq
        mid_lo = start + bos
        mid_hi = start + num_blocks - eos
        if mid_hi <= mid_lo:
            continue
        _normalize_row(x[mid_lo:mid_hi], out[mid_lo:mid_hi])
