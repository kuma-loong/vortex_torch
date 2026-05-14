"""Radix top-k Python wrapper.

CUDA source lives in ``radix_topk.cu``; compilation goes through
``dispatcher.get_kernel``.
"""

from __future__ import annotations

import torch

from .dispatcher import get_kernel


_SOURCE = "radix_topk.cu"

DEFAULT_THREADS_PER_BLOCK = 1024
DEFAULT_MAX_TOPK = 2048
DEFAULT_SMEM_BYTES = 8 * 1024 * 4  # 32 KB

# Sentinel names must match the placeholders in radix_topk.cu.
_SENTINELS = {
    "threads_per_block": "__THREADS_PER_BLOCK__",
    "max_topk": "__VORTEX_MAX_TOPK__",
    "smem_bytes": "__SMEM_BYTES__",
}

_MIN_THREADS_PER_BLOCK = 256
_MAX_THREADS_PER_BLOCK = 1024
_SMEM_ALIGN = 2 * 4


def _validate(threads_per_block: int, max_topk: int, smem_bytes: int) -> None:
    if not (_MIN_THREADS_PER_BLOCK <= threads_per_block <= _MAX_THREADS_PER_BLOCK):
        raise ValueError(
            f"threads_per_block must be in [{_MIN_THREADS_PER_BLOCK}, "
            f"{_MAX_THREADS_PER_BLOCK}], got {threads_per_block}"
        )
    if threads_per_block % 32 != 0:
        raise ValueError(
            f"threads_per_block must be a multiple of 32 (warp size), got {threads_per_block}"
        )
    if max_topk <= 0:
        raise ValueError(f"max_topk must be positive, got {max_topk}")
    if smem_bytes <= 0:
        raise ValueError(f"smem_bytes must be positive, got {smem_bytes}")
    if smem_bytes % _SMEM_ALIGN != 0:
        raise ValueError(
            f"smem_bytes must be a multiple of {_SMEM_ALIGN} "
            f"(two banks of int32), got {smem_bytes}"
        )


def topk(
    x: torch.Tensor,
    dense_kv_indptr: torch.Tensor,
    sparse_kv_indptr: torch.Tensor,
    dense_kv_indices: torch.Tensor,
    sparse_kv_indices: torch.Tensor,
    eff_batch_size: int,
    reserved_bos: int,
    reserved_eos: int,
    max_num_pages: int,
    *,
    threads_per_block: int = DEFAULT_THREADS_PER_BLOCK,
    max_topk: int = DEFAULT_MAX_TOPK,
    smem_bytes: int = DEFAULT_SMEM_BYTES,
    verbose: bool = True,
) -> None:
    """BOS/EOS-aware segmented radix top-k with index remap.

    Compile-time knobs:
      - ``threads_per_block`` — CTA width (multiple of 32, in [256, 1024]).
      - ``max_topk`` — upper bound on per-block ``topk_val``; sizes the
        static ``s_indices`` smem buffer.
      - ``smem_bytes`` — dynamic smem budget for the two-bank candidate
        buffer (``SMEM_INPUT_SIZE = smem_bytes / 8`` per bank).
    """
    _validate(threads_per_block, max_topk, smem_bytes)
    module = get_kernel(
        _SOURCE,
        substitutions={
            _SENTINELS["threads_per_block"]: threads_per_block,
            _SENTINELS["max_topk"]: max_topk,
            _SENTINELS["smem_bytes"]: smem_bytes,
        },
        verbose=verbose,
    )
    module.topk(
        x,
        dense_kv_indptr,
        sparse_kv_indptr,
        dense_kv_indices,
        sparse_kv_indices,
        int(eff_batch_size),
        int(reserved_bos),
        int(reserved_eos),
        int(max_num_pages),
    )
