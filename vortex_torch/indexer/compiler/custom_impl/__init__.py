"""Schedule.S codegen for the indexer compiler.

Schedule.S ops (``topK`` / ``approxTopK`` / ``TopK`` / ``Union`` /
``Softmax`` / ``Normalize`` / ``Conv1d`` / ``Reduce(dim=0)``) emit a
Python wrapper body that resolves an underlying custom kernel via
:func:`vortex_torch.custom_ops.find` at module-import time. The emitted
code is **backend-agnostic** — neither the Triton nor the CUDA Schedule.W
backend is involved — so the same generator is used regardless of
``ctx.impl_backend``.

Public surface:

  * :func:`get_impl_func(op)` — return the codegen function registered
    for ``op``'s class at ``Schedule.S``.
  * :func:`register_headers(ctx)` — append the module-level headers any
    Schedule.S codegen may need to ``ctx.compilation_header_lines``
    (idempotent — ``interface.generate_interface`` dedups via
    ``dict.fromkeys``).
"""
from .register import get_impl_func


def register_headers(ctx) -> None:
    """Append Schedule.S module-level headers to ``ctx``.

    The Schedule.S codegens emit ``@triton.jit``-backed launchers
    resolved via ``custom_ops.find``, so the generated module must
    have ``torch`` and ``triton`` / ``triton.language`` imported at
    the top. ``interface.generate_interface`` runs
    ``dict.fromkeys(ctx.compilation_header_lines)`` before writing,
    so repeated registration is harmless.
    """
    ctx.compilation_header_lines.extend([
        "import torch",
        "import triton",
        "import triton.language as tl",
    ])


__all__ = ["get_impl_func", "register_headers"]
