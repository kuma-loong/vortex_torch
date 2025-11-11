import torch
from ..abs import vOp
from .context import Context
from .triton_kernels.elementwise_impl import elementwise_pp, elementwise_rp, elementwise_pr, elementwise_rr
from ..abs import vTensor, FORMAT, as_vtensor
from ..utils import ElementwiseOpType
from typing import Tuple, Dict, Callable, Optional

class Elementwise(vOp):
    """
    Unary elementwise op dispatcher (e.g., piecewise/sigmoid/silu-like/abs/affine).
    Dispatch key: (x_format, o_format) -> (impl, resolved_output_format)

    Policy:
      - If `output` is None, pick implementation with o_format == FORMAT.RAGGED.
      - Otherwise, require an exact (x_fmt, o_fmt) mapping.
      - Input/Output logical shapes are rank-3: [B, N, D].
    """

    # Implementation registry:
    #   key   = (x_format, o_format)
    #   value = (callable_impl, resolved_output_format)
    _impl_map: Dict[Tuple[FORMAT, FORMAT], Tuple[Callable, FORMAT]] = {
        (FORMAT.PAGED,  FORMAT.PAGED):  (elementwise_pp, FORMAT.PAGED),
        (FORMAT.PAGED,  FORMAT.RAGGED): (elementwise_pr, FORMAT.RAGGED),
        (FORMAT.RAGGED, FORMAT.PAGED):  (elementwise_rp, FORMAT.PAGED),
        (FORMAT.RAGGED, FORMAT.RAGGED): (elementwise_rr, FORMAT.RAGGED),
    }

    def __init__(self, alpha: float = 0.0, beta: float = 1.0):
        """
        Parameters
        ----------
        alpha, beta : scalars used by certain unary ops (e.g., affine/activation variants).
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.op_type: Optional[int] = None          # runtime-set enum/int for the op
        self.impl: Optional[Callable] = None
        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[torch.Tensor] = None

    # ------------------------------ helpers ------------------------------ #
    def _infer_impl_ragged(self, x_fmt: FORMAT) -> Tuple[Callable, FORMAT]:
        """
        Default inference when output is None:
        choose (x_fmt, FORMAT.RAGGED). Raise if missing.
        """
        key = (x_fmt, FORMAT.RAGGED)
        assert key in self._impl_map, (
            f"{self._prefix()}no RAGGED-output implementation for x_fmt={x_fmt}. "
            f"Available keys: {list(self._impl_map.keys())}"
        )
        return self._impl_map[key]  # -> (impl, out_fmt)

    # --------------------------------------------------------------------- #
    # profile: validate, select impl/format, and optionally allocate output
    # --------------------------------------------------------------------- #
    def profile(
        self, x: vTensor, output: Optional[vTensor], loc: torch.Tensor, ctx: Context
    ) -> vTensor:
        prefix = self._prefix()

        # --- type & rank checks ---
        assert isinstance(x, vTensor), f"{prefix}x must be vTensor, got {type(x)}"
        assert isinstance(loc, torch.Tensor), f"{prefix}loc must be torch.Tensor, got {type(loc)}"
        assert x.dim() == 3, f"{prefix}x must be 3D, got ndim={x.dim()} shape={tuple(x.shape)}"

        x_fmt = x._format
        N, D = x.shape[1], x.shape[2]

        # Case A: output not provided -> choose RAGGED impl and allocate buffer
        if output is None:
            self.impl, self.output_format = self._infer_impl_ragged(x_fmt)

            # Allocate on x.device/x.dtype; B comes from runtime context
            B = ctx.max_new_tokens_per_batch * ctx.head_num
            self.output_buffer = torch.empty(
                (B, N, D),
                device=x.device,
                dtype=x.dtype,
            )
            ctx.add_aux_memory(self.output_buffer)
            return as_vtensor(self.output_buffer, self.output_format)

        # Case B: output provided -> validate and select exact impl
        assert isinstance(output, vTensor), f"{prefix}output must be vTensor, got {type(output)}"
        assert output.dim() == 3, f"{prefix}output must be 3D, got ndim={output.dim()} shape={tuple(output.shape)}"

        o_fmt = output._format
        key = (x_fmt, o_fmt)
        assert key in self._impl_map, (
            f"{prefix}no implementation for (x_fmt={x_fmt}, o_fmt={o_fmt}). "
            f"Available keys: {list(self._impl_map.keys())}"
        )
        self.impl, self.output_format = self._impl_map[key]

        # Shape consistency: unary elementwise keeps (N,D)
        assert output.shape[1] == N and output.shape[2] == D, (
            f"{prefix}output shape mismatch. Expected (*,{N},{D}), got {tuple(output.shape)}"
        )

        # Device consistency check
        assert x.device == output.device, (
            f"{prefix}x and output must be on the same device "
            f"(x.device={x.device}, output.device={output.device})"
        )

        return output

    # --------------------------------------------------------------------- #
    # execute: run selected impl and return the plain output tensor
    # --------------------------------------------------------------------- #
    def execute(
        self, x: torch.Tensor, output: Optional[torch.Tensor], loc: torch.Tensor, ctx: Context
    ) -> torch.Tensor:
        prefix = self._prefix()
        assert self.impl is not None, f"{prefix}called before profile() (impl is None)"

        if output is None:
            assert self.output_buffer is not None, f"{prefix}internal output buffer is None; did profile() run?"
            output = self.output_buffer

        # Expected signature for impl: impl(x, output, loc, ctx, op_type, alpha, beta)
        self.impl(x, output, loc, ctx, self.op_type, self.alpha, self.beta)
        return output


class Relu(Elementwise):
    
    def __init__(self, alpha = 0.0, beta = 0.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseOpType.Relu

class Silu(Elementwise):
    
    def __init__(self, alpha = 0.0, beta = 0.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseOpType.Silu
        

class Sigmoid(Elementwise):
    
    def __init__(self, alpha = 0.0, beta = 0.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseOpType.Sigmoid
        

class Add_Mul(Elementwise):
    
    def __init__(self, alpha = 0.0, beta = 1.0):
        super().__init__(alpha, beta)
        self.op_type =  ElementwiseOpType.Add_Mul


class Abs(Elementwise):
    
    def __init__(self, alpha = 0.0, beta = 1.0):
        super().__init__(alpha, beta)
        self.op_type =  ElementwiseOpType.Abs








