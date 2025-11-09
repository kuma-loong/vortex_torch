import torch
from typing import Tuple, Dict, Callable
from .context import Context
from ..abs import vTensor, as_vtensor, FORMAT, vOp
from .triton_kernels.elementwise_impl import elementwise_rr

class Elementwise(vOp):
    
    # Implementation dispatch table: keyed only by x_format.
    _impl_map: Dict[Tuple[FORMAT, FORMAT], Tuple[Callable, FORMAT]] = {
        FORMAT.RAGGED: (elementwise_rr, FORMAT.RAGGED),
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
    def profile(self, x: vTensor, ctx: Context) -> vTensor:
        # Inputs must be vTensor per the system rule (no mixing with plain tensors).
        assert isinstance(x, vTensor), f"Elementwise.profile expects x to be vTensor, got {type(x)}"
        
        
        assert len(x.shape), (
            f"Expected 3D inputs (S, C, D). Got x.ndim={x.dim()},"
        )
        
        x_fmt = x._format
        assert x_fmt in self._impl_map, (
            f"No implementation for (x_fmt={x_fmt}). "
            f"Available keys: {list(self._impl_map.keys())}"
        )

        self.impl, self.output_format = self._impl_map[x_fmt]

        # Allocate output buffer on x.device, with x.dtype (as per your original logic).
        self.output_buffer = torch.empty(
            (ctx.max_num_pages, x.shape[1], x.shape[2]),
            device=x.device,
            dtype=x.dtype,
        )

        # Account auxiliary memory.
        ctx.add_aux_memory(self.output_buffer)

        # Return vTensor view carrying the dispatched output format.
        return as_vtensor(self.output_buffer, self.output_format)

    # ---------------- execute ----------------
    def execute(self, x: torch.Tensor, ctx: Context) -> torch.Tensor:
        # Must call profile() first to set impl and output buffer.
        assert self.impl is not None, "Elementwise.execute called before profile(): impl is None"
        assert self.output_buffer is not None, "Elementwise.execute called before profile(): output_buffer is None"

        # Run the selected implementation. Keep the original argument order and return type.
        self.impl(
            x, self.output_buffer, self.op_type, self.alpha, self.beta, ctx
        )

        # execute returns a plain torch.Tensor by design.
        return self.output_buffer
    

class Relu(Elementwise):
    
    def __init__(self, alpha = 0.0, beta = 0.0):
        super().__init__(alpha, beta)
        self.op_type = "relu"

class Silu(Elementwise):
    
    def __init__(self, alpha = 0.0, beta = 0.0):
        super().__init__(alpha, beta)
        self.op_type = "silu"
        

class Sigmoid(Elementwise):
    
    def __init__(self, alpha = 0.0, beta = 0.0):
        super().__init__(alpha, beta)
        self.op_type = "sigmoid"
        

class Add_Mul(Elementwise):
    
    def __init__(self, alpha = 0.0, beta = 1.0):
        super().__init__(alpha, beta)
        self.op_type = "add_mul"


class Abs(Elementwise):
    
    def __init__(self, alpha = 0.0, beta = 1.0):
        super().__init__(alpha, beta)
        self.op_type = "abs"








