"""Sort top-k Python wrapper.

CUDA source lives in ``sort_topk.cu``; compilation goes through
``dispatcher.get_kernel``.
"""

from __future__ import annotations

import torch

from .dispatcher import get_kernel


_SOURCE = "sort_topk.cu"

DEFAULT_ENABLE_FP8 = False


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
    enable_fp8: bool = DEFAULT_ENABLE_FP8,
    verbose: bool = False,
) -> None:
    """BOS/EOS-aware segmented top-k via CUB ``BlockRadixSort``.

    ``max_num_pages`` selects the ``(THREADS, ITEM_PER_THREAD)`` instance
    at host dispatch; supported range is ``[1, 4096]``.

    Compile-time knobs:
      - ``enable_fp8`` — compile in support for ``Float8_e4m3fn`` and
        ``Float8_e5m2`` scores. Requires ``<cuda_fp8.h>``.
    """
    extra_cuda_cflags: tuple[str, ...] = (
        ("-DVORTEX_ENABLE_FP8",) if enable_fp8 else ()
    )
    module = get_kernel(
        _SOURCE,
        extra_cuda_cflags=extra_cuda_cflags,
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
