"""Dtype <-> Triton cast expression helpers.

Lives in its own module so both the per-block kernel generator
(:mod:`kernel_gen`) and Schedule.S codegens (e.g. :mod:`reduce`) can use
it without creating an import cycle through :mod:`register`.

The helpers mirror the cache-side convention:

  * **Load**  — fp8 inputs come in as a ``uint8`` view; bitcast back to
    ``tl.float8eX`` before the fp32 upcast for compute.
  * **Store** — fp32 → narrow dtype; fp8 paths clamp to the representable
    range first to avoid saturating to ``inf``, then bitcast to ``uint8``.
"""

import torch


_TORCH_TO_TL = {
    torch.bfloat16: "tl.bfloat16",
    torch.float16:  "tl.float16",
    torch.float32:  "tl.float32",
}
_FP8_DTYPES = (torch.float8_e5m2, torch.float8_e4m3fn)


def is_fp8(t) -> bool:
    return t.dtype in _FP8_DTYPES


def load_cast_expr(load_expr: str, t) -> str:
    """Return the fp32 block expression for a ``tl.load(...)`` call.

    FP8 inputs are bitcast back from ``uint8`` to ``tl.float8eX`` before
    the fp32 upcast; everything else casts straight to fp32.
    """
    if t.dtype == torch.float8_e5m2:
        return f"{load_expr}.to(tl.float8e5, bitcast=True).to(tl.float32)"
    if t.dtype == torch.float8_e4m3fn:
        return f"{load_expr}.to(tl.float8e4nv, bitcast=True).to(tl.float32)"
    return f"{load_expr}.to(tl.float32)"


def store_cast_expr(block_expr: str, dtype) -> str:
    """Cast an fp32 compute block to the output tensor's storage dtype.

    FP8 stores clamp to the representable range (e5m2 = 57344,
    e4m3fn = 448) before the narrow cast to avoid saturating to ``inf``,
    then bitcast to ``uint8`` so the wrapper can pass a uint8 view.
    """
    if dtype == torch.float8_e5m2:
        clamped = f"tl.minimum(tl.maximum({block_expr}, -57344.0), 57344.0)"
        return f"{clamped}.to(tl.float8e5).to(tl.uint8, bitcast=True)"
    if dtype == torch.float8_e4m3fn:
        clamped = f"tl.minimum(tl.maximum({block_expr}, -448.0), 448.0)"
        return f"{clamped}.to(tl.float8e4nv).to(tl.uint8, bitcast=True)"
    tl_name = _TORCH_TO_TL.get(dtype)
    if tl_name is None:
        raise NotImplementedError(
            f"indexer.kernel_gen: unsupported store dtype {dtype!r}"
        )
    return f"{block_expr}.to({tl_name})"
