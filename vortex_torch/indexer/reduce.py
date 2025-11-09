import torch
from typing import Tuple, Dict, Callable
from .context import Context
from ..abs import vTensor, as_vtensor, FORMAT, vOp
from .triton_kernels.reduce_impl import reduce_rr


class Reduce(vOp):
    
    # Implementation dispatch table: keyed only by x_format.
    _impl_map: Dict[FORMAT, Tuple[Callable, FORMAT]] = {
        FORMAT.RAGGED: (reduce_rr, FORMAT.RAGGED),
        # Add more (x_fmt,) -> (impl, out_fmt) pairs as needed.
    }

    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim
        self.reduce_type: str = None
        self.impl: Callable = None
        self.output_format: FORMAT | None = None
        self.output_buffer: torch.Tensor | None = None

        assert self.dim in (1, 2), f"Reduce.__init__: dim must be 1 or 2, got dim={self.dim}"
        
    # ---------------- profile ----------------
    def profile(self, x: vTensor, ctx: Context) -> vTensor:
        # Inputs must be vTensor per the system rule (no mixing with plain tensors).
        assert isinstance(x, vTensor), f"Reduce.profile expects x to be vTensor, got {type(x)}"
        

        assert len(x.shape) == 3, (
            f"Expected 3D inputs (B, D0, D1). Got x.ndim={x.dim()}"
        )
        
        # Dispatch by (format_x,).
        x_fmt = x._format
        assert x_fmt in self._impl_map, (
            f"No implementation for (x_fmt={x_fmt}). "
            f"Available keys: {list(self._impl_map.keys())}"
        )

        self.impl, self.output_format = self._impl_map[x_fmt]
        
        # Allocate output buffer on x.device, with x.dtype (as per your original logic).
        self.output_buffer = torch.empty(
            (ctx.max_num_pages, 
                1 if self.dim == 1 else x.shape[1], 
                1 if self.dim == 2 else x.shape[2],
            ),
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
        assert self.impl is not None, "Reduce.execute called before profile(): impl is None"
        assert self.output_buffer is not None, "Reduce.execute called before profile(): output_buffer is None"

        # Run the selected implementation. Keep the original argument order and return type.
        self.impl(
            x, self.output_buffer, self.dim, self.reduce_type, ctx
        )

        # execute returns a plain torch.Tensor by design.
        return self.output_buffer
    

class Max(Reduce):
    
    def __init__(self, dim = 1):
        super().__init__(dim)
        self.reduce_type = "max"


class Min(Reduce):
    
    def __init__(self, dim = 1):
        super().__init__(dim)
        self.reduce_type = "min"


class Mean(Reduce):
    
    def __init__(self, dim = 1):
        super().__init__(dim)
        self.reduce_type = "mean"


class L2Norm(Reduce):
    
    def __init__(self, dim = 1):
        super().__init__(dim)
        self.reduce_type = "l2norm"
