import torch
from typing import Tuple, Dict, Callable, Optional
from .context import Context
from ..abs import vTensor, FORMAT, vOp
from .triton_kernels.softmax_impl import softmax_inplace_r
from .triton_kernels.normalize_impl import normalize_inplace_r

class Softmax(vOp):
    """
    In-place softmax dispatcher.
    - Only supports dim == 0 by design.
    - Dispatch is keyed by the input vTensor's format.
    - No output buffer is allocated; the op is in-place on `x`.
    """

    # Implementation dispatch table: keyed by x_format.
    # Value: (callable_impl, resolved_output_format)
    _impl_map: Dict[FORMAT, Tuple[Callable, FORMAT]] = {
        FORMAT.RAGGED: (softmax_inplace_r, FORMAT.RAGGED),
        # Extend with other formats if you add more kernels, e.g.:
        # FORMAT.PAGED: (softmax_inplace_p, FORMAT.PAGED),
    }

    def __init__(self, dim: int = 0, scale: float = 1.0):
        super().__init__()
        self.dim = dim
        self.scale = scale
        self.impl: Optional[Callable] = None
        self.output_format: Optional[FORMAT] = None
        # Validate dim at construction
        prefix = self._prefix()
        assert self.dim in (0,), f"{prefix}__init__: dim must be 0, got dim={self.dim}"

    # ---------------- profile ----------------
    def profile(self, x: vTensor, ctx: Context) -> vTensor:
        """
        Validate inputs and select implementation. Since the op is in-place,
        no output buffer is allocated; returns `x` as a vTensor view.
        """
        prefix = self._prefix()

        # Type & rank checks
        assert isinstance(x, vTensor), f"{prefix}profile expects x to be vTensor, got {type(x)}"
        assert x.dim() == 3, f"{prefix}expected 3D input [B, D0, D1], got ndim={x.dim()} shape={tuple(x.shape)}"

        # Dispatch by input format
        x_fmt = x._format
        assert x_fmt in self._impl_map, (
            f"{prefix}no implementation for x_fmt={x_fmt}. "
            f"Available keys: {list(self._impl_map.keys())}"
        )
        self.impl, self.output_format = self._impl_map[x_fmt]

        # In-place: return the same vTensor view
        return x

    # ---------------- execute ----------------
    def execute(self, x: torch.Tensor, ctx: Context) -> torch.Tensor:
        """
        Run the selected in-place implementation and return `x`.
        """
        prefix = self._prefix()
        assert self.impl is not None, f"{prefix}execute called before profile() (impl is None)"

        # Expected signature: impl(x, dim, scale, ctx)
        self.impl(x, self.dim, self.scale, ctx)
        return x
    

class Normalize(vOp):
    """
    In-place normalization dispatcher.
    - Only supports dim == 0 by design.
    - Dispatch is keyed by the input vTensor's format.
    - No output buffer is allocated; normalization happens in-place on `x`.
    """

    # Implementation dispatch table: keyed by x_format.
    # Value: (callable_impl, resolved_output_format)
    _impl_map: Dict[FORMAT, Tuple[Callable, FORMAT]] = {
        FORMAT.RAGGED: (normalize_inplace_r, FORMAT.RAGGED),
        # Extend with other formats if you add more kernels, e.g.:
        # FORMAT.PAGED: (normalize_inplace_p, FORMAT.PAGED),
    }

    def __init__(self, dim: int = 0):
        super().__init__()
        self.dim = dim
        self.impl: Optional[Callable] = None
        self.output_format: Optional[FORMAT] = None
        # Validate dim at construction
        prefix = self._prefix()
        assert self.dim in (0,), f"{prefix}__init__: dim must be 0, got dim={self.dim}"

    # ---------------- profile ----------------
    def profile(self, x: vTensor, ctx: Context) -> vTensor:
        """
        Validate inputs and select implementation. Since the op is in-place,
        no output buffer is allocated; returns `x` as a vTensor view.
        """
        prefix = self._prefix()

        # Type & rank checks
        assert isinstance(x, vTensor), f"{prefix}profile expects x to be vTensor, got {type(x)}"
        assert x.dim() == 3, f"{prefix}expected 3D input [B, D0, D1], got ndim={x.dim()} shape={tuple(x.shape)}"

        # Dispatch by input format
        x_fmt = x._format
        assert x_fmt in self._impl_map, (
            f"{prefix}no implementation for x_fmt={x_fmt}. "
            f"Available keys: {list(self._impl_map.keys())}"
        )
        self.impl, self.output_format = self._impl_map[x_fmt]

        # In-place: return the same vTensor view
        return x

    # ---------------- execute ----------------
    def execute(self, x: torch.Tensor, ctx: Context) -> torch.Tensor:
        """
        Run the selected in-place implementation and return `x`.
        """
        prefix = self._prefix()
        assert self.impl is not None, f"{prefix}execute called before profile() (impl is None)"

        # Expected signature: impl(x, dim, ctx)
        self.impl(x, self.dim, ctx)
        return x


