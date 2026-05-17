"""Pure-torch reference for ``conv_1d`` — slow but obviously correct.

Per (req, kv_head) row, slide a ``K``-tap causal 1D convolution along
the block axis (axis 0) of the middle ``[bos, num_blocks - eos)`` slice.
``out[p] = sum_k weight[k] * x[p - k]`` per ``(D0, D1)`` lane; left
padding is zeros (causal). Rows with ``num_blocks <= bos + eos + topk_val``
are skipped.
"""
from __future__ import annotations

import torch


def _conv1d_row(x_seg: torch.Tensor, out_seg: torch.Tensor, weight: torch.Tensor, K: int):
    """``x_seg``: [num_blocks, D0, D1]; ``weight``: [K, D0, D1]."""
    if x_seg.shape[0] == 0:
        return
    P = x_seg.shape[0]
    xf = x_seg.float()
    wf = weight.float()
    acc = torch.zeros_like(xf)
    for k in range(K):
        if k == 0:
            acc = acc + xf * wf[k]
        else:
            # shift right by k (zero left-pad)
            shifted = torch.zeros_like(xf)
            shifted[k:] = xf[: P - k]
            acc = acc + shifted * wf[k]
    out_seg.copy_(acc.to(out_seg.dtype))


def reference_flashinfer(
    x, out, weight, indptr, bos, eos, topk_val, K, D0, D1, eff_batch_size,
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
        _conv1d_row(x[mid_lo:mid_hi], out[mid_lo:mid_hi], weight, K)


def reference_trtllm(
    x, out, weight, seqlens, bos, eos, topk_val, K, D0, D1,
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
        _conv1d_row(x[mid_lo:mid_hi], out[mid_lo:mid_hi], weight, K)
