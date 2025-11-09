import torch
from typing import Tuple, Dict, Callable
from .context import Context
from ..abs import vTensor, FORMAT, vOp
from .triton_kernels.softmax_impl import softmax_inplace_r
from .triton_kernels.normalize_impl import normalize_inplace_r

class Softmax(vOp):
    
    # Implementation dispatch table: keyed only by x_format.
    _impl_map: Dict[FORMAT, Tuple[Callable, FORMAT]] = {
        FORMAT.RAGGED: (softmax_inplace_r, FORMAT.RAGGED),
        # Add more (x_fmt,) -> (impl, out_fmt) pairs as needed.
    }

    def __init__(self, dim: int = 0, scale: float = 1.0):
        super().__init__()
        self.dim = dim
        self.impl: Callable = None
        self.scale = scale
        assert self.dim in (0,), f"Softmax.__init__: dim must be 0, got dim={self.dim}"
        
    # ---------------- profile ----------------
    def profile(self, x: vTensor, ctx: Context):
        # Inputs must be vTensor per the system rule (no mixing with plain tensors).
        assert isinstance(x, vTensor), f"Softmax.profile expects x to be vTensor, got {type(x)}"
        

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
        
        
        
    # ---------------- execute ----------------
    def execute(self, x: torch.Tensor, ctx: Context) -> torch.Tensor:
        # Must call profile() first to set impl and output buffer.
        assert self.impl is not None, "Softmax.execute called before profile(): impl is None"

        # Run the selected implementation. Keep the original argument order and return type.
        self.impl(
            x, self.dim, self.scale, ctx
        )




class Normalize(vOp):
    
    # Implementation dispatch table: keyed only by x_format.
    _impl_map: Dict[FORMAT, Tuple[Callable, FORMAT]] = {
        FORMAT.RAGGED: (normalize_inplace_r, FORMAT.RAGGED),
        # Add more (x_fmt,) -> (impl, out_fmt) pairs as needed.
    }

    def __init__(self, dim: int = 0):
        super().__init__()
        self.dim = dim
        self.impl: Callable = None
        assert self.dim in (0,), f"Normalize.__init__: dim must be 0, got dim={self.dim}"
        
    # ---------------- profile ----------------
    def profile(self, x: vTensor, ctx: Context):
        # Inputs must be vTensor per the system rule (no mixing with plain tensors).
        assert isinstance(x, vTensor), f"Normalize.profile expects x to be vTensor, got {type(x)}"
        

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
        
        
        
    # ---------------- execute ----------------
    def execute(self, x: torch.Tensor, ctx: Context) -> torch.Tensor:
        # Must call profile() first to set impl and output buffer.
        assert self.impl is not None, "Normalize.execute called before profile(): impl is None"

        # Run the selected implementation. Keep the original argument order and return type.
        self.impl(
            x, self.dim, ctx
        )



