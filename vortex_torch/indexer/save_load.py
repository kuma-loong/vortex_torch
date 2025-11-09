import torch
from typing import Tuple, Dict, Callable
from .context import Context
from ..abs import vTensor, as_vtensor, FORMAT, vOp
from .triton_kernels.save_load_impl import save_rp, load_pr

class Save(vOp):
    
    # Implementation dispatch table: keyed only by x_format.
    _impl_map: Dict[FORMAT, Tuple[Callable, FORMAT]] = {
        FORMAT.RAGGED: (save_rp, FORMAT.PAGED),
        # Add more (x_fmt,) -> (impl, out_fmt) pairs as needed.
    }

    def __init__(self):
        super().__init__()
        
        self.impl: Callable = None
        self.output_format: FORMAT | None = None
        self.output_buffer: torch.Tensor | None = None
        
        
    # ---------------- profile ----------------
    def profile(self, x: vTensor, o: vTensor, ctx: Context):
        # Inputs must be vTensor per the system rule (no mixing with plain tensors).
        assert isinstance(x, vTensor), f"Save.profile expects x to be vTensor, got {type(x)}"
        

        assert len(x.shape) == 3, (
            f"Expected 3D inputs (S, D0, D1). Got x.ndim={x.dim()}"
        )
        
        assert len(o.shape) == 3, f"Expected 3D output (S, D0, D1). Got o.ndim={o.dim()}"
        
        assert x.shape[1] == o.shape[1], (
            f"Expected matching D0 (second dim): x.shape[1]={x.shape[1]} "
            f"but got o.shape[1]={o.shape[1]}"
        )
        assert x.shape[2] == o.shape[2], (
            f"Expected matching D1 (third dim): x.shape[2]={x.shape[2]} "
            f"but got o.shape[2]={o.shape[2]}"
        )

        # Dispatch by (format_x,).
        x_fmt = x._format
        assert x_fmt in self._impl_map, (
            f"No implementation for (x_fmt={x_fmt}). "
            f"Available keys: {list(self._impl_map.keys())}"
        )

        self.impl, self.output_format = self._impl_map[x_fmt]
        
       
    # ---------------- execute ----------------
    def execute(self, x: torch.Tensor, o: torch.Tensor, ctx: Context) -> torch.Tensor:
        # Must call profile() first to set impl and output buffer.
        assert self.impl is not None, "Save.execute called before profile(): impl is None"

        # Run the selected implementation. Keep the original argument order and return type.
        self.impl(x, o, ctx)




class Load(vOp):
    
    # Implementation dispatch table: keyed only by x_format.
    _impl_map: Dict[FORMAT, Tuple[Callable, FORMAT]] = {
        FORMAT.PAGED: (load_pr, FORMAT.RAGGED),
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
        assert isinstance(x, vTensor), f"Load.profile expects x to be vTensor, got {type(x)}"
        

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
        assert self.impl is not None, "Load.execute called before profile(): impl is None"

        # Run the selected implementation. Keep the original argument order and return type.
        self.impl(x, self.output_buffer, ctx)
        
        return self.output_buffer