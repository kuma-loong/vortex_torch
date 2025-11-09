import torch
from typing import Tuple, Dict, Callable
from .context import Context
from ..abs import vTensor, as_vtensor, FORMAT, vOp
from .triton_kernels.elementwise_binary_impl import elementwise_binary_bpr, elementwise_binary_rrr, elementwise_binary_rpr

class Elementwise_Binary(vOp):
    
    # Implementation dispatch table: keyed only by (x_format, y_format).
    _impl_map: Dict[Tuple[FORMAT, FORMAT], Tuple[Callable, FORMAT]] = {
        (FORMAT.RAGGED, FORMAT.RAGGED): (elementwise_binary_rrr, FORMAT.RAGGED),
        (FORMAT.BATCHED, FORMAT.PAGED): (elementwise_binary_bpr, FORMAT.RAGGED),
        (FORMAT.RAGGED, FORMAT.PAGED): (elementwise_binary_rpr, FORMAT.RAGGED),
        # Add more (x_fmt, y_fmt) -> (impl, out_fmt) pairs as needed.
    }

    def __init__(self, alpha: float=1.0, beta: float=1.0):
        super().__init__()
        self.impl: Callable = None
        self.op_type: str = None
        self.alpha = alpha
        self.beta = beta
        self.output_format: FORMAT | None = None
        self.output_buffer: torch.Tensor | None = None

    # ---------------- profile ----------------
    def profile(self, x: vTensor, y: vTensor, ctx: Context) -> vTensor:
        # Inputs must be vTensor per the system rule (no mixing with plain tensors).
        assert isinstance(x, vTensor), f"Elementwise_Binary.profile expects x to be vTensor, got {type(x)}"
        assert isinstance(y, vTensor), f"Elementwise_Binary.profile expects y to be vTensor, got {type(y)}"

        # Dimensionality checks (expect [*, 1, *] with both tensors being 3D).
        assert len(x.shape) == len(y.shape) == 3, (
            f"Expected 3D inputs (S, C/1, D/1). Got x.ndim={x.dim()}, y.ndim={y.dim()}"
        )
        
        assert (
            x.shape[1] == y.shape[1] or
            x.shape[1] == 1 or
            y.shape[1] == 1
        ), f"Expected x.shape[1] == y.shape[1] or one of them to be 1 (broadcastable), "
        f"got x.shape={tuple(x.shape)}, y.shape={tuple(y.shape)}"

        assert (
            x.shape[2] == y.shape[2] or
            x.shape[2] == 1 or
            y.shape[2] == 1
        ), f"Expected x.shape[2] == y.shape[2] or one of them to be 1 (broadcastable), "
        f"got x.shape={tuple(x.shape)}, y.shape={tuple(y.shape)}"
        

        # Dispatch by (format_x, format_y).
        x_fmt, y_fmt = x._format, y._format
        assert (x_fmt, y_fmt) in self._impl_map, (
            f"No implementation for (x_fmt={x_fmt}, y_fmt={y_fmt}). "
            f"Available keys: {list(self._impl_map.keys())}"
        )

        self.impl, self.output_format = self._impl_map[(x_fmt, y_fmt)]

        # Allocate output buffer on x.device, with x.dtype (as per your original logic).
        self.output_buffer = torch.empty(
            (ctx.max_num_pages, max(x.shape[1], y.shape[1]), max(x.shape[2], y.shape[2])),
            device=x.device,
            dtype=x.dtype,
        )

        # Account auxiliary memory.
        ctx.add_aux_memory(self.output_buffer)

        # Return vTensor view carrying the dispatched output format.
        return as_vtensor(self.output_buffer, self.output_format)

    # ---------------- execute ----------------
    def execute(self, x: torch.Tensor, y: torch.Tensor, ctx: Context) -> torch.Tensor:
        # Must call profile() first to set impl and output buffer.
        assert self.impl is not None, "Elementwise_Binary.execute called before profile(): impl is None"
        assert self.output_buffer is not None, "Elementwise_Binary.execute called before profile(): output_buffer is None"

        # Run the selected implementation. Keep the original argument order and return type.
        self.impl(
            x, y, self.output_buffer, self.op_type, self.alpha, self.beta, ctx
        )

        # execute returns a plain torch.Tensor by design.
        return self.output_buffer


class Maximum(Elementwise_Binary):
    
    def __init__(self, alpha = 1, beta = 1):
        super().__init__(alpha, beta)
        self.op_type = "maximum"


class Minimum(Elementwise_Binary):
    
    def __init__(self, alpha = 1, beta = 1):
        super().__init__(alpha, beta)
        self.op_type = "minimum"
        
        
class Add(Elementwise_Binary):
    
    def __init__(self, alpha = 1, beta = 1):
        super().__init__(alpha, beta)
        self.op_type = "add"
        

class Multiply(Elementwise_Binary):
    
    def __init__(self, alpha = 1, beta = 1):
        super().__init__(alpha, beta)
        self.op_type = "mul"



