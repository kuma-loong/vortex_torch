import torch
from typing import Tuple, Dict, Callable, Optional
from .context import Context
from ..abs import vTensor, as_vtensor, FORMAT, vOp
from .triton_kernels.elementwise_impl import elementwise_rr
from ..utils import ElementwiseOpType

class Elementwise(vOp):
    """
    Unary elementwise dispatcher for rank-3 logical tensors [S, C, D].
    - Dispatch is keyed only by the input format (x._format).
    - Output has the same logical shape as input.
    - Alpha/Beta are scalar params used by certain ops.
    """

    # Implementation dispatch table: keyed only by x_format.
    # Value: (callable_impl, resolved_output_format)
    _impl_map: Dict[FORMAT, Tuple[Callable, FORMAT]] = {
        FORMAT.RAGGED: (elementwise_rr, FORMAT.RAGGED),
        # Add more entries if you support other formats, e.g.:
        # FORMAT.PAGED: (elementwise_pp, FORMAT.PAGED),
    }

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__()
        self.impl: Optional[Callable] = None
        self.op_type: Optional[ElementwiseOpType] = None
        self.alpha = alpha
        self.beta = beta
        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[torch.Tensor] = None

    # ---------------- profile ----------------
    def profile(self, x: vTensor, ctx: Context) -> vTensor:
        """
        Validate input, select implementation by x._format,
        allocate output buffer, and return a vTensor view with resolved format.
        """
        prefix = self._prefix()

        # Type & rank checks
        assert isinstance(x, vTensor), f"{prefix}profile expects x to be vTensor, got {type(x)}"
        assert x.dim() == 3, (
            f"{prefix}expected 3D input [S, C, D]. Got ndim={x.dim()} shape={tuple(x.shape)}"
        )

        # Dispatch by input format
        x_fmt = x._format
        assert x_fmt in self._impl_map, (
            f"{prefix}no implementation for x_fmt={x_fmt}. "
            f"Available keys: {list(self._impl_map.keys())}"
        )
        self.impl, self.output_format = self._impl_map[x_fmt]

        # Allocate output buffer on x.device with x.dtype
        S = ctx.max_num_pages             # runtime "S" axis from context
        C, D = x.shape[1], x.shape[2]
        self.output_buffer = torch.empty(
            (S, C, D),
            device=x.device,
            dtype=x.dtype,
        )

        # Account auxiliary memory
        ctx.add_aux_memory(self.output_buffer)

        # Return vTensor view with dispatched output format
        return as_vtensor(self.output_buffer, self.output_format)

    # ---------------- execute ----------------
    def execute(self, x: torch.Tensor, ctx: Context) -> torch.Tensor:
        """
        Run the selected implementation into the internal buffer and return it.
        Expected signature: impl(x, output, op_type, alpha, beta, ctx)
        """
        prefix = self._prefix()
        assert self.impl is not None, f"{prefix}execute called before profile() (impl is None)"
        assert self.output_buffer is not None, f"{prefix}output buffer is None; did profile() run?"
        assert x.device == self.output_buffer.device, (
            f"{prefix}device mismatch: x.device={x.device}, o.device={self.output_buffer.device}"
        )

        self.impl(x, self.output_buffer, self.op_type, self.alpha, self.beta, ctx)
        return self.output_buffer
    

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
        self.op_type = ElementwiseOpType.Add_Mul


class Abs(Elementwise):
    
    def __init__(self, alpha = 0.0, beta = 1.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseOpType.Abs








