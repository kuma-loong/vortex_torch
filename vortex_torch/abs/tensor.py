"""Pure-metadata tensor type used throughout the vortex graph + compiler.

A :class:`vTensor` does **not** carry any real storage. It is a small
descriptor that records exactly what the rest of the system needs to
reason about a tensor at graph-construction and codegen time:

  * ``shape``     — tuple of ints
  * ``dtype``     — :class:`torch.dtype`
  * ``device``    — :class:`torch.device` / str / ``None``
  * ``_format``   — :class:`FORMAT` (BATCHED / RAGGED / PAGED)
  * ``tensor_id`` — int, the graph-level identity used by the compiler

It also exposes ``dim()`` for parity with ``torch.Tensor`` so existing
profile-time validation code (``assert x.dim() == 3``) keeps working.

There is intentionally **no** torch op support, no ``__torch_function__``
override, no parent ``torch.Tensor`` class. ``vTensor`` is just metadata.
Real tensors used by the runtime/execute path stay as plain
``torch.Tensor`` instances.
"""

from __future__ import annotations
import torch
from enum import Enum
from typing import Any, Optional, Sequence, Union


class FORMAT(Enum):
    """Tensor storage/layout format.

    Attributes:
        BATCHED: Standard dense batched tensors (e.g., ``[B, N, D]``).
        RAGGED: Ragged tensors with variable-length sequences or elements per batch.
        PAGED: Paged tensors used for large or streaming data split into pages/chunks.
    """

    BATCHED = 0
    RAGGED = 1
    PAGED = 2


class vTensor:
    """Pure-metadata virtual tensor.

    Carries the descriptor fields used by the graph builder, the
    compiler, and the codegen layer. It does not own any GPU / CPU
    memory and intentionally cannot participate in torch ops — code
    that wants to compute on real data should hold a ``torch.Tensor``
    separately and use the ``vTensor`` only for graph bookkeeping.
    """

    __slots__ = ("shape", "dtype", "device", "_format", "tensor_id")

    shape: tuple
    dtype: torch.dtype
    device: Optional[Union[torch.device, str]]
    _format: FORMAT
    tensor_id: int

    def __init__(
        self,
        shape: Sequence[int] = (),
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[Union[torch.device, str]] = None,
        _format: FORMAT = FORMAT.BATCHED,
        tensor_id: int = -1,
    ) -> None:
        if not isinstance(tensor_id, int):
            raise TypeError(f"tensor_id must be int, got {type(tensor_id).__name__}")
        if not isinstance(_format, FORMAT):
            raise TypeError(f"_format must be a FORMAT enum, got {type(_format).__name__}")

        # Normalize ``shape`` so ``shape[i]``, ``len(shape)`` and ``tuple(shape)``
        # all behave like ``torch.Tensor.shape``.
        self.shape = tuple(int(s) for s in shape)
        self.dtype = dtype
        self.device = device
        self._format = _format
        self.tensor_id = tensor_id

    # -------- shape helpers --------
    def dim(self) -> int:
        """Number of dimensions; mirrors :meth:`torch.Tensor.dim`."""
        return len(self.shape)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def numel(self) -> int:
        n = 1
        for s in self.shape:
            n *= int(s)
        return n

    def size(self, dim: Optional[int] = None):
        """Mirror of :meth:`torch.Tensor.size`."""
        if dim is None:
            return self.shape
        return self.shape[dim]

    # -------- repr --------
    def __repr__(self) -> str:
        return (
            f"vTensor(shape={self.shape}, dtype={self.dtype}, "
            f"device={self.device}, _format={self._format}, "
            f"tensor_id={self.tensor_id})"
        )

    # -------- pickle / copy --------
    def __reduce__(self):
        return (
            _rebuild_vtensor,
            (self.shape, self.dtype, self.device, self._format, self.tensor_id),
        )


def _rebuild_vtensor(shape, dtype, device, _format, tensor_id):
    return vTensor(
        shape=shape, dtype=dtype, device=device, _format=_format, tensor_id=tensor_id,
    )


# -------- convenience factory --------
def as_vtensor(
    x: Any = None,
    _format: FORMAT = FORMAT.BATCHED,
    tensor_id: int = -1,
    *,
    shape: Optional[Sequence[int]] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[torch.device, str]] = None,
) -> vTensor:
    """Build a :class:`vTensor`.

    Three calling styles, all returning a fresh ``vTensor`` (or, for an
    existing ``vTensor``, the same object re-tagged):

    1. **Re-tag an existing vTensor** — ``as_vtensor(vt, fmt, tid)``
       overwrites ``vt._format`` and ``vt.tensor_id`` in place and
       returns ``vt``. Useful when the caller wants to add an existing
       tensor descriptor to the graph under a fresh id.

    2. **Extract metadata from a torch.Tensor** — ``as_vtensor(real, fmt, tid)``
       reads ``shape``, ``dtype``, ``device`` from ``real`` and returns
       a brand-new ``vTensor``. The original tensor is **not** retained
       — vTensor is pure metadata.

    3. **Direct construction by kwargs** —
       ``as_vtensor(_format=fmt, tensor_id=tid, shape=..., dtype=..., device=...)``.
       Use this when no real torch tensor is available (the common case
       once the compile path is fully virtualized).
    """
    if isinstance(x, vTensor):
        x._format = _format
        x.tensor_id = tensor_id
        return x

    if isinstance(x, torch.Tensor):
        return vTensor(
            shape=tuple(x.shape),
            dtype=x.dtype,
            device=x.device,
            _format=_format,
            tensor_id=tensor_id,
        )

    if x is None:
        return vTensor(
            shape=shape if shape is not None else (),
            dtype=dtype if dtype is not None else torch.bfloat16,
            device=device,
            _format=_format,
            tensor_id=tensor_id,
        )

    raise TypeError(
        f"as_vtensor: cannot convert {type(x).__name__} to vTensor; "
        "pass a torch.Tensor, an existing vTensor, or shape/dtype/device kwargs."
    )


if __name__ == "__main__":
    # Direct construction
    a = vTensor(shape=(2, 3, 4), dtype=torch.bfloat16, device="cuda:0",
                _format=FORMAT.RAGGED, tensor_id=0)
    print("a:", a)
    print("dim:", a.dim(), "ndim:", a.ndim, "numel:", a.numel(), "size(1):", a.size(1))

    # Extract metadata from a real torch.Tensor
    real = torch.empty(2, 3, 4, dtype=torch.float8_e4m3fn)
    b = as_vtensor(real, FORMAT.PAGED, tensor_id=1)
    print("b:", b)

    # Re-tag an existing vTensor
    c = as_vtensor(a, FORMAT.BATCHED, tensor_id=42)
    assert c is a
    print("c (same as a, re-tagged):", c)

    # Direct construction via kwargs only
    d = as_vtensor(_format=FORMAT.RAGGED, tensor_id=2,
                   shape=(0, 4, 128), dtype=torch.bfloat16, device="cuda:0")
    print("d:", d)
