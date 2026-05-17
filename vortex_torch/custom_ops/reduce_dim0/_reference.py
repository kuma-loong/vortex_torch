"""Pure-torch reference for ``reduce_dim0`` — slow but obviously correct.

Per (req, kv_head) row, reduce the middle ``[bos, num_blocks - eos)``
blocks along axis 0 into a single ``(D0, D1)`` tile, writing into the
BATCHED output at index ``pid``. The reduction kind is selected by
``variant`` ∈ ``{"sum", "mean", "max", "min", "l2norm"}`` — one
reference function per kind for clarity.

Empty middle (``num_blocks <= bos + eos``) ⇒ write zeros.
"""
from __future__ import annotations

import torch


def _reduce(x_seg: torch.Tensor, variant: str) -> torch.Tensor:
    if x_seg.shape[0] == 0:
        return torch.zeros(x_seg.shape[1:], dtype=torch.float32, device=x_seg.device)
    xf = x_seg.float()
    if variant == "sum":
        return xf.sum(dim=0)
    if variant == "mean":
        return xf.mean(dim=0)
    if variant == "max":
        return xf.amax(dim=0)
    if variant == "min":
        return xf.amin(dim=0)
    if variant == "l2norm":
        return (xf * xf).sum(dim=0).sqrt()
    raise ValueError(f"reduce_dim0: unknown variant {variant!r}")


def _row_segment_flashinfer(x, indptr, pid: int, bos: int, eos: int):
    start = int(indptr[pid].item())
    end = int(indptr[pid + 1].item())
    num_blocks = end - start
    if num_blocks <= bos + eos:
        return x[0:0]  # empty view
    return x[start + bos : end - eos]


def _row_segment_trtllm(x, seqlens, block_size, max_blocks_per_seq, pid, bos, eos):
    tokens = int(seqlens[pid].item())
    num_blocks = (tokens + block_size - 1) // block_size
    if num_blocks <= bos + eos:
        return x[0:0]
    start = pid * max_blocks_per_seq
    return x[start + bos : start + num_blocks - eos]


def reference_flashinfer(
    x, out, indptr, bos, eos, D0, D1, eff_batch_size, *, variant: str,
):
    """Write ``out[pid] = reduce(x[indptr[pid]+bos : indptr[pid+1]-eos], variant)``."""
    for pid in range(eff_batch_size):
        seg = _row_segment_flashinfer(x, indptr, pid, bos, eos)
        out[pid].copy_(_reduce(seg, variant).to(out.dtype))


def reference_trtllm(
    x, out, seqlens, bos, eos, D0, D1, block_size, max_blocks_per_seq,
    eff_batch_size, *, variant: str,
):
    for pid in range(eff_batch_size):
        seg = _row_segment_trtllm(x, seqlens, block_size, max_blocks_per_seq, pid, bos, eos)
        out[pid].copy_(_reduce(seg, variant).to(out.dtype))
