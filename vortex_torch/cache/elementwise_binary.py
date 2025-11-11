import torch
from ..abs import vOp
from .context import Context
from .triton_kernels.elementwise_binary_impl import elementwise_binary_ppp, elementwise_binary_rrp, elementwise_binary_ppr, elementwise_binary_rrr
from ..abs import vTensor, FORMAT, as_vtensor
from ..utils import ElementwiseBinaryOpType
from typing import Tuple, Dict, Callable, Optional


class Elementwise_Binary(vOp):
    """
    Elementwise binary op dispatcher (Maximum / Minimum / AXPBY / Mul).

    Dispatch key: (x_format, y_format, o_format) -> (impl, resolved_output_format)

    Policy:
      - If `output` is None, choose the mapping with o_format == FORMAT.RAGGED.
      - Otherwise, require an exact (x_fmt, y_fmt, o_fmt) match.
      - Shapes are rank-3: [B, N, D], with broadcasting allowed on N/D.
    """

    _impl_map: Dict[Tuple[FORMAT, FORMAT, FORMAT], Tuple[Callable, FORMAT]] = {
        (FORMAT.PAGED,  FORMAT.PAGED,  FORMAT.PAGED):  (elementwise_binary_ppp, FORMAT.PAGED),
        (FORMAT.PAGED,  FORMAT.PAGED,  FORMAT.RAGGED): (elementwise_binary_ppr, FORMAT.RAGGED),
        (FORMAT.RAGGED, FORMAT.RAGGED, FORMAT.PAGED):  (elementwise_binary_rrp, FORMAT.PAGED),
        (FORMAT.RAGGED, FORMAT.RAGGED, FORMAT.RAGGED): (elementwise_binary_rrr, FORMAT.RAGGED),
    }

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.op_type: Optional[ElementwiseBinaryOpType] = None
        self.impl: Optional[Callable] = None
        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[torch.Tensor] = None

    # ------------------------------ helpers ------------------------------ #
    def _infer_impl_ragged(
        self, x_fmt: FORMAT, y_fmt: FORMAT
    ) -> Tuple[Callable, FORMAT]:
        """Prefer (x_fmt, y_fmt, FORMAT.RAGGED)."""
        key = (x_fmt, y_fmt, FORMAT.RAGGED)
        assert key in self._impl_map, (
            f"{self._prefix()}no RAGGED-output implementation for "
            f"(x_fmt={x_fmt}, y_fmt={y_fmt}). Available keys: {list(self._impl_map.keys())}"
        )
        return self._impl_map[key]

    # ------------------------------------------------------------------ #
    def profile(
        self, x: vTensor, y: vTensor, output: Optional[vTensor], loc: torch.Tensor, ctx: Context
    ) -> vTensor:
        prefix = self._prefix()

        # --- type checks ---
        assert isinstance(x, vTensor), f"{prefix}x must be vTensor, got {type(x)}"
        assert isinstance(y, vTensor), f"{prefix}y must be vTensor, got {type(y)}"
        assert isinstance(loc, torch.Tensor), f"{prefix}loc must be torch.Tensor, got {type(loc)}"

        assert x.dim() == 3, f"{prefix}x must be 3D, got ndim={x.dim()} shape={tuple(x.shape)}"
        assert y.dim() == 3, f"{prefix}y must be 3D, got ndim={y.dim()} shape={tuple(y.shape)}"

        # --- broadcastability checks ---
        assert (
            x.shape[1] == y.shape[1] or x.shape[1] == 1 or y.shape[1] == 1
        ), f"{prefix}dim-1 not broadcastable: x={x.shape}, y={y.shape}"
        assert (
            x.shape[2] == y.shape[2] or x.shape[2] == 1 or y.shape[2] == 1
        ), f"{prefix}dim-2 not broadcastable: x={x.shape}, y={y.shape}"

        x_fmt, y_fmt = x._format, y._format
        exp_N, exp_D = max(x.shape[1], y.shape[1]), max(x.shape[2], y.shape[2])

        # Case A: output None → choose RAGGED impl
        if output is None:
            self.impl, self.output_format = self._infer_impl_ragged(x_fmt, y_fmt)
            B = ctx.max_new_tokens_per_batch * ctx.head_num
            self.output_buffer = torch.empty((B, exp_N, exp_D), device=x.device, dtype=x.dtype)
            ctx.add_aux_memory(self.output_buffer)
            return as_vtensor(self.output_buffer, self.output_format)

        # Case B: output provided → exact match
        assert isinstance(output, vTensor), f"{prefix}output must be vTensor, got {type(output)}"
        assert output.dim() == 3, f"{prefix}output must be 3D, got ndim={output.dim()} shape={tuple(output.shape)}"

        o_fmt = output._format
        key = (x_fmt, y_fmt, o_fmt)
        assert key in self._impl_map, (
            f"{prefix}no implementation for (x_fmt={x_fmt}, y_fmt={y_fmt}, o_fmt={o_fmt}). "
            f"Available keys: {list(self._impl_map.keys())}"
        )
        self.impl, self.output_format = self._impl_map[key]

        assert output.shape[1] == exp_N and output.shape[2] == exp_D, (
            f"{prefix}output shape mismatch. Expected (*,{exp_N},{exp_D}), got {tuple(output.shape)}"
        )
        assert x.device == y.device == output.device, (
            f"{prefix}x, y, and output must be on the same device "
            f"(x.device={x.device}, y.device={y.device}, output.device={output.device})"
        )

        return output

    # ------------------------------------------------------------------ #
    def execute(
        self, x: torch.Tensor, y: torch.Tensor, output: Optional[torch.Tensor], loc: torch.Tensor, ctx: Context
    ) -> torch.Tensor:
        prefix = self._prefix()
        assert self.impl is not None, f"{prefix}called before profile() (impl is None)"

        if output is None:
            assert self.output_buffer is not None, f"{prefix}internal output buffer missing; did profile() run?"
            output = self.output_buffer

        # Launch impl
        self.impl(x, y, output, loc, ctx, self.op_type, self.alpha, self.beta)
        return output

class Maximum(Elementwise_Binary):
    def __init__(self, alpha = 1, beta = 1):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseBinaryOpType.Maximum


class Minimum(Elementwise_Binary):
    def __init__(self, alpha = 1, beta = 1):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseBinaryOpType.Minimum
        

class Add(Elementwise_Binary):
    def __init__(self, alpha = 1, beta = 1):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseBinaryOpType.Add


class Multiply(Elementwise_Binary):
    def __init__(self, alpha = 1, beta = 1):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseBinaryOpType.Mul