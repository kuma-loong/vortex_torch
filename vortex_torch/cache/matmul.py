import torch
from .context import Context
from ..abs import vOp
from .triton_kernels.matmul_impl import gemm_ppp, gemm_rrp, gemm_ppr, gemm_rrr
from ..abs import vTensor, FORMAT, as_vtensor
from typing import Tuple, Dict, Callable, Optional


class GeMM(vOp):
    """
    GEMM dispatcher for O = Y @ X^T on page/token tiled layouts.

    Layout/shape convention (logical 3D tensors):
      x: [B, Nx, K]
      y: [B, Ny, K]
      o: [B, Ny, Nx]   (i.e., rows from y, cols from x)

    Dispatch key:
      (x_format, y_format, o_format) -> (impl, resolved_output_format)

    Policy:
      - If `output` is None, prefer implementations with o_format == FORMAT.RAGGED.
      - Otherwise, require an exact (x_fmt, y_fmt, o_fmt) mapping.
      - K dimension must match: x.shape[2] == y.shape[2].
    """

    # Implementation registry:
    #   value impl is a Python wrapper that launches the corresponding kernel, e.g.:
    #   def gemm_ppp(x, y, output, loc, ctx): ...
    _impl_map: Dict[Tuple[FORMAT, FORMAT, FORMAT], Tuple[Callable, FORMAT]] = {
        (FORMAT.PAGED,  FORMAT.PAGED,  FORMAT.PAGED):  (gemm_ppp, FORMAT.PAGED),
        (FORMAT.PAGED,  FORMAT.PAGED,  FORMAT.RAGGED): (gemm_ppr, FORMAT.RAGGED),
        (FORMAT.RAGGED, FORMAT.RAGGED, FORMAT.PAGED):  (gemm_rrp, FORMAT.PAGED),
        (FORMAT.RAGGED, FORMAT.RAGGED, FORMAT.RAGGED): (gemm_rrr, FORMAT.RAGGED),
    }

    def __init__(self):
        
        super().__init__()
        self.impl: Optional[Callable] = None
        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[torch.Tensor] = None

    
    def _infer_impl_ragged(
        self, x_fmt: FORMAT, y_fmt: FORMAT
    ) -> Tuple[Callable, FORMAT]:
        """
        Default inference when output is None:
        choose (x_fmt, y_fmt, FORMAT.RAGGED). Raise if missing.
        """
        key = (x_fmt, y_fmt, FORMAT.RAGGED)
        assert key in self._impl_map, (
            f"{self._prefix()}no RAGGED-output implementation for "
            f"(x_fmt={x_fmt}, y_fmt={y_fmt}). "
            f"Available keys: {list(self._impl_map.keys())}"
        )
        return self._impl_map[key]  # -> (impl, out_fmt)

    # --------------------------------------------------------------------- #
    # profile: validate, select impl/format, and optionally allocate output
    # --------------------------------------------------------------------- #
    def profile(
        self, x: vTensor, y: vTensor, output: Optional[vTensor], loc: torch.Tensor, ctx: Context
    ) -> vTensor:
        prefix = self._prefix()

        # --- type & rank checks ---
        assert isinstance(x, vTensor), f"{prefix}x must be vTensor, got {type(x)}"
        assert isinstance(y, vTensor), f"{prefix}y must be vTensor, got {type(y)}"
        assert isinstance(loc, torch.Tensor), f"{prefix}loc must be torch.Tensor, got {type(loc)}"

        assert x.dim() == 3, f"{prefix}x must be 3D, got ndim={x.dim()} shape={tuple(x.shape)}"
        assert y.dim() == 3, f"{prefix}y must be 3D, got ndim={y.dim()} shape={tuple(y.shape)}"

        # --- K match: x[..., K] == y[..., K] ---
        Kx, Ky = x.shape[2], y.shape[2]
        assert Kx == Ky, f"{prefix}K mismatch: x.shape[2]={Kx} vs y.shape[2]={Ky}"

        # Output logical shape: [B, Ny, Nx]
        Ny, Nx = y.shape[1], x.shape[1]
        x_fmt, y_fmt = x._format, y._format

        # Case A: output not provided -> choose RAGGED impl and allocate buffer
        if output is None:
            self.impl, self.output_format = self._infer_impl_ragged(x_fmt, y_fmt)

            # Allocate on x.device/x.dtype; B comes from runtime context
            B = ctx.max_new_tokens_per_batch * ctx.head_num
            self.output_buffer = torch.empty(
                (B, Ny, Nx),
                device=x.device,
                dtype=x.dtype,
            )
            ctx.add_aux_memory(self.output_buffer)
            return as_vtensor(self.output_buffer, self.output_format)

        # Case B: output provided -> validate and select exact impl
        assert isinstance(output, vTensor), f"{prefix}output must be vTensor, got {type(output)}"
        assert output.dim() == 3, f"{prefix}output must be 3D, got ndim={output.dim()} shape={tuple(output.shape)}"

        o_fmt = output._format
        key = (x_fmt, y_fmt, o_fmt)
        assert key in self._impl_map, (
            f"{prefix}no implementation for (x_fmt={x_fmt}, y_fmt={y_fmt}, o_fmt={o_fmt}). "
            f"Available keys: {list(self._impl_map.keys())}"
        )
        self.impl, self.output_format = self._impl_map[key]

        # Shape consistency: GEMM yields [*, Ny, Nx]
        assert output.shape[1] == Ny and output.shape[2] == Nx, (
            f"{prefix}output shape mismatch. Expected (*,{Ny},{Nx}), got {tuple(output.shape)}"
        )

        # Device consistency check
        assert x.device == y.device == output.device, (
            f"{prefix}x, y, and output must be on the same device "
            f"(x.device={x.device}, y.device={y.device}, output.device={output.device})"
        )

        return output

    # --------------------------------------------------------------------- #
    # execute: run selected impl and return the plain output tensor
    # --------------------------------------------------------------------- #
    def execute(
        self, x: torch.Tensor, y: torch.Tensor, output: Optional[torch.Tensor], loc: torch.Tensor, ctx: Context
    ) -> torch.Tensor:
        prefix = self._prefix()
        assert self.impl is not None, f"{prefix}called before profile() (impl is None)"

        if output is None:
            assert self.output_buffer is not None, f"{prefix}internal output buffer is None; did profile() run?"
            output = self.output_buffer

        # Expected signature for impl: impl(x, y, output, loc, ctx)
        self.impl(x, y, output, loc, ctx)
        return output
