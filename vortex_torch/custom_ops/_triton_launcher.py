"""Adapter to expose ``@triton.jit`` kernels as plain Python callables.

Without this wrapper, Triton-backed leaves return raw ``@triton.jit``
objects that are launched with ``kernel[(grid,)](*args, num_warps=...,
num_stages=...)`` — a launch convention CUDA-backed leaves do not
share (their ``load_inline``'d callable is invoked as a regular
function).

``make_launcher`` collapses that difference: every Triton leaf's
``dispatch()`` returns a plain callable ``launch(*args)`` whose contract
matches the CUDA leaves' — the caller never indexes ``[grid]`` and
never sets ``num_warps`` / ``num_stages`` themselves. The grid is the
trailing positional argument ``eff_batch_size``; everything before it
maps 1-to-1 onto the underlying kernel's positional parameters
(constexpr or not — Triton handles constexpr inference from values).

This keeps the compiler-side launcher emitters identical across
Triton and CUDA backends: emit a single plain function call with
``eff_batch_size`` as the trailing argument, regardless of impl.
"""
from __future__ import annotations

from typing import Callable


def make_launcher(
    kernel,
    *,
    num_warps: int = 4,
    num_stages: int = 1,
) -> Callable[..., None]:
    """Wrap ``kernel`` (a ``@triton.jit`` function) as ``launch(*args)``.

    Convention:

      * ``args[-1]`` is ``eff_batch_size`` (the 1D grid size).
      * ``args[:-1]`` are the kernel's positional arguments, in the
        order declared in the kernel signature.

    ``num_warps`` / ``num_stages`` are baked into the wrapper at
    ``dispatch()`` time; the caller does not pass them.
    """
    def launch(*args):
        eff_batch_size = args[-1]
        kernel_args = args[:-1]
        kernel[(eff_batch_size,)](
            *kernel_args,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return launch


__all__ = ["make_launcher"]
