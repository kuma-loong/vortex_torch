import torch
from enum import Enum
from typing import Any, Iterable
import numbers


class FORMAT(Enum):
    # Standard dense batched tensors (e.g., [B, N, D])
    BATCHED = 0
    # Ragged tensors with variable-length sequences or elements per batch
    RAGGED = 1
    # Paged tensors used for large or streaming data split into pages/chunks
    PAGED = 2


class vTensor(torch.Tensor):
    """Tensor subclass with a `_format` metadata field.

    Rules:
    - Torch ops do NOT change `_format`; it must be consistent across all vTensors in the op.
    - vTensor CANNOT participate in ops with plain torch.Tensors (raise RuntimeError).
    - vTensor CAN participate in ops with Python scalars (int/float/bool).
    """

    # -------- construction & (de)serialization --------
    def __new__(cls, data, _format: FORMAT = FORMAT.BATCHED, **kwargs):
        # Accept scalar/iterable/torch.Tensor inputs
        if not isinstance(data, torch.Tensor):
            data = torch.as_tensor(data, **kwargs)
        # Create subclass without copying storage
        out = torch.Tensor._make_subclass(cls, data, require_grad=data.requires_grad)
        out._format = _format
        return out

    def __repr__(self):
        base = super().__repr__().replace(self.__class__.__name__, "Tensor")
        return f"vTensor(format={self._format}, {base})"

    # Pickle support
    def __reduce_ex__(self, protocol):
        return (self._rebuild_from_tensor, (torch.Tensor(self), self._format, self.requires_grad))

    @staticmethod
    def _rebuild_from_tensor(t: torch.Tensor, fmt: FORMAT, requires_grad: bool):
        out = torch.Tensor._make_subclass(vTensor, t, require_grad=requires_grad)
        out._format = fmt
        return out

    # -------- helpers --------
    @classmethod
    def _from_base(cls, base: torch.Tensor, fmt: FORMAT):
        out = torch.Tensor._make_subclass(cls, base, require_grad=base.requires_grad)
        out._format = fmt
        return out

    @staticmethod
    def _iter_tensors(obj: Any) -> Iterable[torch.Tensor]:
        """Recursively yield tensors from nested args/kwargs."""
        if isinstance(obj, torch.Tensor):
            yield obj
        elif isinstance(obj, (list, tuple)):
            for x in obj:
                yield from vTensor._iter_tensors(x)
        elif isinstance(obj, dict):
            for x in obj.values():
                yield from vTensor._iter_tensors(x)

    @staticmethod
    def _extract_format_and_validate(args, kwargs) -> FORMAT | None:
        """Collect vTensor formats, ensure no mixing with plain Tensors, and return the op format."""
        tensors = list(vTensor._iter_tensors((args, kwargs)))
        # Disallow presence of any plain torch.Tensor alongside vTensor
        has_v = any(isinstance(t, vTensor) for t in tensors)
        has_plain = any(isinstance(t, torch.Tensor) and not isinstance(t, vTensor) for t in tensors)
        if has_v and has_plain:
            raise RuntimeError("vTensor cannot operate with plain torch.Tensor in the same op.")

        # If only scalars and/or vTensors are present, enforce format consistency
        fmts = {t._format for t in tensors if isinstance(t, vTensor)}
        if len(fmts) > 1:
            raise RuntimeError(f"vTensor _format mismatch in op: {fmts}")
        return next(iter(fmts)) if fmts else None

    @classmethod
    def _propagate_format(cls, result, fmt: FORMAT):
        """Propagate fmt into results (wrap plain Tensors as vTensor)."""
        if fmt is None:
            return result  # No vTensor involved => nothing to propagate

        if isinstance(result, torch.Tensor):
            if isinstance(result, vTensor):
                result._format = fmt
                return result
            else:
                return cls._from_base(result, fmt)
        elif isinstance(result, (list, tuple)):
            t = type(result)
            return t(cls._propagate_format(x, fmt) for x in result)
        elif isinstance(result, dict):
            return {k: cls._propagate_format(v, fmt) for k, v in result.items()}
        else:
            # Non-tensor outputs (numbers, bools, None, etc.) are returned as-is
            return result

    @staticmethod
    def _is_allowed_scalar(x: Any) -> bool:
        """Return True if x is a Python numeric/bool scalar allowed to mix with vTensor."""
        return isinstance(x, numbers.Number)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        # If no vTensor types are involved, fall back to default behavior.
        if not any(issubclass(t, vTensor) for t in types):
            return super().__torch_function__(func, types, args, kwargs)

        # Validate: no plain torch.Tensor with vTensor; allow scalars.
        # (Scalars aren't torch.Tensors, so they won't appear in _iter_tensors; that's fine.)
        fmt = cls._extract_format_and_validate(args, kwargs)

        # Execute op via parent dispatch
        out = super().__torch_function__(func, types, args, kwargs)

        # Propagate format into outputs
        return cls._propagate_format(out, fmt)


# -------- convenience factory --------
def as_vtensor(x: torch.Tensor | Any, _format: FORMAT = FORMAT.BATCHED) -> vTensor:
    """Wrap an input as vTensor (no storage copy). If already vTensor, just update format."""
    if isinstance(x, vTensor):
        x._format = _format
        return x
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    return vTensor._from_base(x, _format)


if __name__ == "__main__":
    a = as_vtensor(torch.randn(2, 3), FORMAT.RAGGED)
    b = as_vtensor(torch.ones(2, 3), FORMAT.RAGGED)

    # Allowed: vTensor with scalar
    y = a + 2.0
    print("ok scalar:", isinstance(y, vTensor), y._format)

    # Allowed: vTensor with vTensor (same format)
    z = a * b
    print("ok same-format:", isinstance(z, vTensor), z._format)

    # Disallowed: vTensor with plain Tensor
    try:
        _ = a + torch.randn(2, 3)
    except RuntimeError as e:
        print("expected error (plain tensor):", e)

    # Disallowed: different formats
    c = as_vtensor(torch.randn(2, 3), FORMAT.PAGED)
    try:
        _ = a + c
    except RuntimeError as e:
        print("expected error (format mismatch):", e)