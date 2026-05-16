"""Dtype ↔ CUDA cast expression helpers.

Lives in its own module so :mod:`kernel_gen` (and any per-op codegen
under :mod:`cuda_impl`) can use it without creating an import cycle
through :mod:`register` — exactly the same reason
:mod:`triton_impl.dtype_cast` is split out on the Triton side.

The helpers mirror the cache-side convention:

  * **smem -> float** — narrow storage dtype is upcast to ``float`` via
    the matching CUDA intrinsic (``__half2float`` / ``__bfloat162float``)
    before fp32 compute.
  * **float -> smem** — fp32 compute result is rounded to the storage
    dtype (``__float2half_rn`` / ``__float2bfloat16_rn``).

fp8 storage isn't covered here — the load / store codegens own the
fp8 ↔ uint8 boundary, and compute ops currently assume their smem
tiles are bf16 / fp16 / fp32.
"""

import torch


def cast_smem_to_float(expr: str, dtype) -> str:
    """Cast a storage-dtype smem value (``expr``) to ``float`` for fp32
    compute.
    """
    if dtype is torch.float32:
        return expr
    if dtype is torch.float16:
        return f"__half2float({expr})"
    if dtype is torch.bfloat16:
        return f"__bfloat162float({expr})"
    raise NotImplementedError(
        f"cuda_impl.dtype_cast: no smem -> float cast for dtype {dtype!r}"
    )


def cast_float_to_smem(expr: str, dtype) -> str:
    """Cast an fp32 compute value (``expr``) to a storage dtype for
    writing back into a smem tile. Rounding is ``rn`` (round to
    nearest even), matching Triton's default narrow-dtype cast.
    """
    if dtype is torch.float32:
        return expr
    if dtype is torch.float16:
        return f"__float2half_rn({expr})"
    if dtype is torch.bfloat16:
        return f"__float2bfloat16_rn({expr})"
    raise NotImplementedError(
        f"cuda_impl.dtype_cast: no float -> smem cast for dtype {dtype!r}"
    )
