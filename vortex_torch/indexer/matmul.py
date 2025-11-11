import torch
from typing import Tuple, Dict, Callable, Optional
from .context import Context
from ..abs import vTensor, as_vtensor, FORMAT, vOp

from .triton_kernels.mv_impl import mv_bpr
from .triton_kernels.matmul_impl import mm_bpr, mm_rrr, mm_rpr

# ------------------------------ GeMV ------------------------------ #
class GeMV(vOp):
    """
    General matrix–vector multiplication (GEMV) dispatcher:
      O = Y @ X^T with logical shapes:
        X: [B, 1, D]
        Y: [S_pack, 1, D]
        O: [S_pack, 1, 1]
    Dispatch is keyed by (x_format, y_format).
    """

    # Implementation dispatch table: keyed by (x_format, y_format).
    # Value: (callable_impl, resolved_output_format)
    _impl_map: Dict[Tuple[FORMAT, FORMAT], Tuple[Callable, FORMAT]] = {
        (FORMAT.BATCHED, FORMAT.PAGED): (mv_bpr, FORMAT.RAGGED),
        # Extend with more pairs as needed.
    }

    def __init__(self):
        super().__init__()
        self.impl: Optional[Callable] = None
        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[torch.Tensor] = None

    # ---------------- profile ----------------
    def profile(self, x: vTensor, y: vTensor, ctx: Context) -> vTensor:
        """
        Validate inputs, select implementation, allocate output buffer,
        and return an as_vtensor view with the resolved format.
        """
        prefix = self._prefix()

        # Type checks
        assert isinstance(x, vTensor), f"{prefix}profile expects x to be vTensor, got {type(x)}"
        assert isinstance(y, vTensor), f"{prefix}profile expects y to be vTensor, got {type(y)}"

        # Rank/shape checks
        assert x.dim() == 3 and y.dim() == 3, (
            f"{prefix}expected 3D inputs; got x.ndim={x.dim()}, y.ndim={y.dim()}"
        )
        assert x.shape[1] == 1, f"{prefix}expected x.shape[1] == 1, got {tuple(x.shape)}"
        assert y.shape[1] == 1, f"{prefix}expected y.shape[1] == 1, got {tuple(y.shape)}"
        assert x.shape[2] == y.shape[2], (
            f"{prefix}last dimension mismatch: x.shape[2]={x.shape[2]} vs y.shape[2]={y.shape[2]}"
        )

        # Dispatch
        x_fmt, y_fmt = x._format, y._format
        key = (x_fmt, y_fmt)
        assert key in self._impl_map, (
            f"{prefix}no implementation for (x_fmt={x_fmt}, y_fmt={y_fmt}). "
            f"Available: {list(self._impl_map.keys())}"
        )
        self.impl, self.output_format = self._impl_map[key]

        # Allocate output buffer on x.device/x.dtype
        S_out = ctx.max_num_pages          # logical "S_pack" per your runtime
        self.output_buffer = torch.empty(
            (S_out, 1, 1),
            device=x.device,
            dtype=x.dtype,
        )
        ctx.add_aux_memory(self.output_buffer)

        return as_vtensor(self.output_buffer, self.output_format)

    # ---------------- execute ----------------
    def execute(self, x: torch.Tensor, y: torch.Tensor, ctx: Context) -> torch.Tensor:
        """
        Launch the selected implementation into the internal buffer and return it.
        Expected signature: impl(x, y, output, ctx)
        """
        prefix = self._prefix()
        assert self.impl is not None, f"{prefix}execute called before profile() (impl is None)"
        assert self.output_buffer is not None, f"{prefix}output buffer is None; did profile() run?"
        assert x.device == y.device == self.output_buffer.device, (
            f"{prefix}device mismatch: "
            f"x={x.device}, y={y.device}, o={self.output_buffer.device}"
        )

        self.impl(x, y, self.output_buffer, ctx)
        return self.output_buffer


# ------------------------------ GeMM ------------------------------ #
class GeMM(vOp):
    """
    General matrix–matrix multiplication dispatcher:
      O = Y @ X^T with logical shapes:
        X: [B/S, Nx, K]
        Y: [S, Ny, K]
        O: [S, Ny, Nx]
    Dispatch is keyed by (x_format, y_format).
    """

    # Implementation dispatch table: keyed by (x_format, y_format).
    # Value: (callable_impl, resolved_output_format)
    _impl_map: Dict[Tuple[FORMAT, FORMAT], Tuple[Callable, FORMAT]] = {
        (FORMAT.BATCHED, FORMAT.PAGED): (mm_bpr, FORMAT.RAGGED),
        (FORMAT.RAGGED, FORMAT.RAGGED): (mm_rrr, FORMAT.RAGGED),
        (FORMAT.RAGGED, FORMAT.PAGED):  (mm_rpr, FORMAT.RAGGED),
        # Extend with more pairs as needed.
    }

    def __init__(self):
        super().__init__()
        self.impl: Optional[Callable] = None
        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[torch.Tensor] = None

    # ---------------- profile ----------------
    def profile(self, x: vTensor, y: vTensor, ctx: Context) -> vTensor:
        """
        Validate inputs, select implementation, allocate output buffer,
        and return an as_vtensor view with the resolved format.
        """
        prefix = self._prefix()

        # Type checks
        assert isinstance(x, vTensor), f"{prefix}profile expects x to be vTensor, got {type(x)}"
        assert isinstance(y, vTensor), f"{prefix}profile expects y to be vTensor, got {type(y)}"

        # Rank/shape checks
        assert x.dim() == 3 and y.dim() == 3, (
            f"{prefix}expected 3D inputs; got x.ndim={x.dim()}, y.ndim={y.dim()}"
        )
        # K must match
        assert x.shape[2] == y.shape[2], (
            f"{prefix}last dimension mismatch: x.shape[2]={x.shape[2]} vs y.shape[2]={y.shape[2]}"
        )

        # Dispatch
        x_fmt, y_fmt = x._format, y._format
        key = (x_fmt, y_fmt)
        assert key in self._impl_map, (
            f"{prefix}no implementation for (x_fmt={x_fmt}, y_fmt={y_fmt}). "
            f"Available: {list(self._impl_map.keys())}"
        )
        self.impl, self.output_format = self._impl_map[key]

        # Output logical sizes: Ny x Nx
        Ny, Nx = y.shape[1], x.shape[1]
        B_out  = ctx.max_num_pages

        # Allocate output buffer on x.device/x.dtype
        self.output_buffer = torch.empty(
            (B_out, Ny, Nx),
            device=x.device,
            dtype=x.dtype,
        )
        ctx.add_aux_memory(self.output_buffer)

        return as_vtensor(self.output_buffer, self.output_format)

    # ---------------- execute ----------------
    def execute(self, x: torch.Tensor, y: torch.Tensor, ctx: Context) -> torch.Tensor:
        """
        Launch the selected implementation into the internal buffer and return it.
        Expected signature: impl(x, y, output, ctx)
        """
        prefix = self._prefix()
        assert self.impl is not None, f"{prefix}execute called before profile() (impl is None)"
        assert self.output_buffer is not None, f"{prefix}output buffer is None; did profile() run?"
        assert x.device == y.device == self.output_buffer.device, (
            f"{prefix}device mismatch: "
            f"x={x.device}, y={y.device}, o={self.output_buffer.device}"
        )

        self.impl(x, y, self.output_buffer, ctx)
        return self.output_buffer