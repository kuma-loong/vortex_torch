"""Pure-torch reference for ``softmax`` — slow but obviously correct.

Mirrors the kernel semantics documented in ``SPEC.md``:

  * For each (req, kv_head) row, gather the RAGGED block segment.
  * If ``num_blocks <= bos + eos + topk_val``: pass through (no write).
  * Otherwise, softmax along the block axis (axis 0) over the middle
    ``[bos, num_blocks - eos)`` blocks; write into ``out`` in place.

Both backends share the same op semantics; only the per-row
addressing differs (CSR ``indptr`` vs trtllm ``seqlens`` +
``block_size`` + ``max_blocks_per_seq``).

Use as a correctness oracle when validating a new leaf:

    from vortex_torch.custom_ops.softmax._reference import reference_flashinfer
    out_ref = torch.empty_like(x)
    reference_flashinfer(x, out_ref, indptr, scale, bos, eos, topk_val,
                         D0, D1, eff_batch_size)
    diff = (out_ref.float() - out_kernel.float()).abs().max()
    assert diff < 1e-2
"""
from __future__ import annotations

import torch


def _softmax_row(x_seg: torch.Tensor, out_seg: torch.Tensor, scale: float):
    """Softmax along axis 0 over a ``[num_blocks, D0, D1]`` slice."""
    if x_seg.shape[0] == 0:
        return
    scaled = scale * x_seg.float()
    m = scaled.amax(dim=0, keepdim=True)
    e = (scaled - m).exp()
    s = e.sum(dim=0, keepdim=True)
    out_seg.copy_((e / s).to(out_seg.dtype))


def reference_flashinfer(
    x, out, indptr, scale, bos, eos, topk_val, D0, D1, eff_batch_size,
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
        _softmax_row(x[mid_lo:mid_hi], out[mid_lo:mid_hi], scale)


def reference_trtllm(
    x, out, seqlens, scale, bos, eos, topk_val, D0, D1,
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
        _softmax_row(x[mid_lo:mid_hi], out[mid_lo:mid_hi], scale)
