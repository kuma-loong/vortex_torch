"""Custom-op dispatch tree for Schedule.S indexer implementations.

Layout — every op gets the same 4-level shape so future per-backend
or per-parameter buckets can land without touching the upper layers::

    vortex_torch/custom_ops/
        __init__.py             # level-0: route on ``op_name``
        _jit.py                 # CUDA JIT helper (load_inline cache)
        <op_name>/
            dispatch.py         # level-1: route on ``backend``
            <backend>/
                _abi.py         # C ABI string (CUDA leaves only)
                dispatch.py     # level-2: pick a leaf bucket from kwargs
                <bucket>/       # level-3: the actual implementation
                    dispatch.py # ``dispatch(**leaf_kwargs)`` -> kernel
                    kernel.py   #   (Triton leaves)
                    kernel.cu   #   (CUDA leaves)
                    config.json #   (CUDA leaves)

Each ``dispatch.py`` exports either:

  * ``find(**kwargs)`` — intermediate routers. Returns the *next*
    layer's ``find`` callable's return value (i.e. recurses).
  * ``dispatch(**leaf_kwargs)`` — leaves. Returns a kernel callable:
    a JIT-compiled torch-extension ``topk`` (CUDA leaves) or a
    ``@triton.jit`` kernel object (Triton leaves).

Public entry point::

    from vortex_torch.custom_ops import find

    # CUDA op with bucket selection by ``max_topk_val``:
    topk_fn = find("topk_output", "trtllm", max_topk_val=256)()

    # Triton op (single bucket today):
    softmax_kernel = find("softmax", "flashinfer")()

    # Reduce_dim0 — bucket = reduce_type:
    mean_kernel = find("reduce_dim0", "trtllm", variant="mean")()

    # approxTopK — leaf takes a per-call knob:
    approx_fn = find("topk_output", "flashinfer", approx=True)(
        tolerate_ratio=0.05
    )

Even ops with a single backend (``topk`` / ``union`` — trtllm only) or
a single bucket (``softmax`` / ``normalize`` / ``conv_1d``) still go
through every layer so adding a new backend or bucket is a pure
``mkdir`` + drop-in, never a refactor of the routing surface.
"""
from __future__ import annotations

import importlib
from typing import Callable


def find(op_name: str, backend: str = "flashinfer", **kwargs) -> Callable[..., object]:
    """Resolve ``(op_name, backend, **routing kwargs)`` to a leaf's
    ``dispatch`` callable.

    Routing recurses through three layers:

      1. ``op_name`` → ``custom_ops/<op_name>/dispatch.py::find``
      2. ``backend`` → ``custom_ops/<op_name>/<backend>/dispatch.py::find``
      3. remaining kwargs (``max_topk_val`` / ``approx`` / ``variant`` /
         …) → ``custom_ops/<op_name>/<backend>/<bucket>/dispatch.py::dispatch``

    The returned callable is the leaf's ``dispatch`` function
    **uninvoked** — callers invoke it with the leaf's per-call args
    (e.g. ``tolerate_ratio=`` for approxTopK, ``verbose=`` for CUDA
    leaves; nothing for Triton-kernel leaves).
    """
    try:
        op_mod = importlib.import_module(f"vortex_torch.custom_ops.{op_name}.dispatch")
    except ModuleNotFoundError as exc:
        raise ValueError(
            f"custom_ops.find: unknown op_name {op_name!r} "
            f"(no module vortex_torch.custom_ops.{op_name}.dispatch)"
        ) from exc
    return op_mod.find(backend, **kwargs)


__all__ = ["find"]
