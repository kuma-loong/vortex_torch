import torch
from typing import Tuple, Dict, Callable
from .context import Context
from ..abs import vTensor, as_vtensor, FORMAT, vOp
from .triton_kernels.transpose_impl import transpose_rr

class Transpose(vOp):
    
    # Implementation dispatch table: keyed only by x_format.
    _impl_map: Dict[FORMAT, Tuple[Callable, FORMAT]] = {
        FORMAT.RAGGED: (transpose_rr, FORMAT.RAGGED),
        # Add more (x_fmt,) -> (impl, out_fmt) pairs as needed.
    }

    def __init__(self):
        super().__init__()
        
        self.impl: Callable = None
        self.output_format: FORMAT | None = None
        self.output_buffer: torch.Tensor | None = None
        
        
    # ---------------- profile ----------------
    def profile(self, x: vTensor, ctx: Context):
        # Inputs must be vTensor per the system rule (no mixing with plain tensors).
        assert isinstance(x, vTensor), f"Transpose.profile expects x to be vTensor, got {type(x)}"
        

        assert len(x.shape) == 3, (
            f"Expected 3D inputs (S, D0, D1). Got x.ndim={x.dim()}"
        )
        
        # Dispatch by (format_x,).
        x_fmt = x._format
        assert x_fmt in self._impl_map, (
            f"No implementation for (x_fmt={x_fmt}). "
            f"Available keys: {list(self._impl_map.keys())}"
        )

        self.impl, self.output_format = self._impl_map[x_fmt]
        
        self.output_buffer = torch.empty(
            (ctx.max_num_pages, x.shape[2], x.shape[1]),
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
        assert self.impl is not None, "Transpose.execute called before profile(): impl is None"

        # Run the selected implementation. Keep the original argument order and return type.
        self.impl(x, self.output_buffer, ctx)
