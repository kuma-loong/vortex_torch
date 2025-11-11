import torch
from typing import Tuple, Dict, Callable, Optional
from .context import Context
from ..abs import vTensor, as_vtensor, FORMAT, vOp
from .triton_kernels.elementwise_binary_impl import elementwise_binary_bpr, elementwise_binary_rrr, elementwise_binary_rpr
from ..utils import ElementwiseBinaryOpType

class Elementwise_Binary(vOp):
    """
    Binary elementwise dispatcher for rank-3 logical tensors [S, C, D].
    - Dispatch is keyed by (x_format, y_format).
    - Output logical shape is broadcast over (C, D) and keeps S from runtime context.
    - Alpha/Beta are scalar params used by certain ops (e.g., axpby).
    """

    # Implementation dispatch table: keyed by (x_format, y_format).
    # Value: (callable_impl, resolved_output_format)
    _impl_map: Dict[Tuple[FORMAT, FORMAT], Tuple[Callable, FORMAT]] = {
        (FORMAT.RAGGED,  FORMAT.RAGGED): (elementwise_binary_rrr, FORMAT.RAGGED),
        (FORMAT.BATCHED, FORMAT.PAGED):  (elementwise_binary_bpr, FORMAT.RAGGED),
        (FORMAT.RAGGED,  FORMAT.PAGED):  (elementwise_binary_rpr, FORMAT.RAGGED),
        # Add more pairs as needed.
    }

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__()
        self.impl: Optional[Callable] = None
        self.op_type: Optional[ElementwiseBinaryOpType] = None
        self.alpha = alpha
        self.beta = beta
        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[torch.Tensor] = None

    # ---------------- profile ----------------
    def profile(self, x: vTensor, y: vTensor, ctx: Context) -> vTensor:
        """
        Validate inputs, select implementation by (x._format, y._format),
        allocate output buffer with broadcasted (C, D), and return a vTensor view.
        """
        prefix = self._prefix()

        # Type checks
        assert isinstance(x, vTensor), f"{prefix}profile expects x to be vTensor, got {type(x)}"
        assert isinstance(y, vTensor), f"{prefix}profile expects y to be vTensor, got {type(y)}"

        # Rank & basic shape checks
        assert x.dim() == 3 and y.dim() == 3, (
            f"{prefix}expected 3D inputs [S, C, D]; got x.ndim={x.dim()}, y.ndim={y.dim()}"
        )

        # Broadcastability on C/D
        assert (x.shape[1] == y.shape[1] or x.shape[1] == 1 or y.shape[1] == 1), (
            f"{prefix}dim-1 not broadcastable: x.shape={tuple(x.shape)}, y.shape={tuple(y.shape)}"
        )
        assert (x.shape[2] == y.shape[2] or x.shape[2] == 1 or y.shape[2] == 1), (
            f"{prefix}dim-2 not broadcastable: x.shape={tuple(x.shape)}, y.shape={tuple(y.shape)}"
        )

        # Dispatch
        x_fmt, y_fmt = x._format, y._format
        key = (x_fmt, y_fmt)
        assert key in self._impl_map, (
            f"{prefix}no implementation for (x_fmt={x_fmt}, y_fmt={y_fmt}). "
            f"Available: {list(self._impl_map.keys())}"
        )
        self.impl, self.output_format = self._impl_map[key]

        # Device consistency
        assert x.device == y.device, (
            f"{prefix}x and y must be on the same device "
            f"(x.device={x.device}, y.device={y.device})"
        )

        # Broadcasted output (C, D)
        C_out = max(x.shape[1], y.shape[1])
        D_out = max(x.shape[2], y.shape[2])

        # Allocate output buffer on x.device with x.dtype
        S = ctx.max_num_pages
        self.output_buffer = torch.empty(
            (S, C_out, D_out),
            device=x.device,
            dtype=x.dtype,
        )
        ctx.add_aux_memory(self.output_buffer)

        # Return vTensor view with dispatched output format
        return as_vtensor(self.output_buffer, self.output_format)

    # ---------------- execute ----------------
    def execute(self, x: torch.Tensor, y: torch.Tensor, ctx: Context) -> torch.Tensor:
        """
        Run the selected implementation into the internal buffer and return it.
        Expected signature: impl(x, y, output, op_type, alpha, beta, ctx)
        """
        prefix = self._prefix()
        assert self.impl is not None, f"{prefix}execute called before profile() (impl is None)"
        assert self.output_buffer is not None, f"{prefix}output buffer is None; did profile() run?"
        assert x.device == y.device == self.output_buffer.device, (
            f"{prefix}device mismatch: "
            f"x={x.device}, y={y.device}, o={self.output_buffer.device}"
        )

        self.impl(x, y, self.output_buffer, self.op_type, self.alpha, self.beta, ctx)
        return self.output_buffer


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



