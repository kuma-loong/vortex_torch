import torch
from typing import Tuple, Dict, Callable, Optional
from .context import Context
from ..abs import vTensor, as_vtensor, FORMAT, vOp
from .triton_kernels.reduce_impl import reduce_rr
from ..utils import ReduceType

class Reduce(vOp):
    """
    Reduction dispatcher for rank-3 logical tensors [B, D0, D1].
    - Dispatch is keyed only by the input format (x._format).
    - Output shape depends on `dim`:
        dim == 1 -> reduce rows (over D0)  -> output [B, 1,  D1]
        dim == 2 -> reduce cols (over D1)  -> output [B, D0, 1]
    - The actual reduction type (Mean/Max/Min/L2Norm/...) is carried by `reduce_type`.
    """

    # Implementation dispatch table: keyed only by x_format.
    # Value: (callable_impl, resolved_output_format)
    _impl_map: Dict[FORMAT, Tuple[Callable, FORMAT]] = {
        FORMAT.RAGGED: (reduce_rr, FORMAT.RAGGED),
        # Add more entries if you support other formats, e.g.:
        # FORMAT.PAGED: (reduce_pp, FORMAT.PAGED),
    }

    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim
        self.reduce_type: Optional[ReduceType] = None
        self.impl: Optional[Callable] = None
        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[torch.Tensor] = None

        # Validate reduction dimension at construction
        prefix = self._prefix()
        assert self.dim in (1, 2), f"{prefix}__init__: dim must be 1 or 2, got dim={self.dim}"

    # ---------------- profile ----------------
    def profile(self, x: vTensor, ctx: Context) -> vTensor:
        """
        Validate input, select implementation by x._format,
        allocate output buffer with the resolved format, and return a vTensor view.
        """
        prefix = self._prefix()

        # Type & rank checks
        assert isinstance(x, vTensor), f"{prefix}profile expects x to be vTensor, got {type(x)}"
        assert x.dim() == 3, f"{prefix}expected 3D input [B, D0, D1], got ndim={x.dim()} shape={tuple(x.shape)}"

        # Dispatch by input format
        x_fmt = x._format
        assert x_fmt in self._impl_map, (
            f"{prefix}no implementation for x_fmt={x_fmt}. "
            f"Available keys: {list(self._impl_map.keys())}"
        )
        self.impl, self.output_format = self._impl_map[x_fmt]

        # Compute output logical shape according to `dim`
        B = ctx.max_num_pages                      # batch/time axis per your runtime
        D0, D1 = x.shape[1], x.shape[2]
        out_D0 = 1 if self.dim == 1 else D0        # dim-1 collapsed when reducing rows
        out_D1 = 1 if self.dim == 2 else D1        # dim-2 collapsed when reducing cols

        # Allocate output buffer on x.device with x.dtype
        self.output_buffer = torch.empty(
            (B, out_D0, out_D1),
            device=x.device,
            dtype=x.dtype,
        )

        # Account auxiliary memory
        ctx.add_aux_memory(self.output_buffer)

        # Return vTensor view carrying the dispatched output format
        return as_vtensor(self.output_buffer, self.output_format)

    # ---------------- execute ----------------
    def execute(self, x: torch.Tensor, ctx: Context) -> torch.Tensor:
        """
        Run the selected reduction implementation into the internal buffer and return it.
        """
        prefix = self._prefix()
        assert self.impl is not None, f"{prefix}execute called before profile() (impl is None)"
        assert self.output_buffer is not None, f"{prefix}output buffer is None; did profile() run?"

        # Expected signature: impl(x, output, dim, reduce_type, ctx)
        self.impl(x, self.output_buffer, self.dim, self.reduce_type, ctx)
        return self.output_buffer
    

class Max(Reduce):
    
    def __init__(self, dim = 1):
        super().__init__(dim)
        self.reduce_type = ReduceType.Max


class Min(Reduce):
    
    def __init__(self, dim = 1):
        super().__init__(dim)
        self.reduce_type = ReduceType.Min


class Mean(Reduce):
    
    def __init__(self, dim = 1):
        super().__init__(dim)
        self.reduce_type = ReduceType.Mean


class L2Norm(Reduce):
    
    def __init__(self, dim = 1):
        super().__init__(dim)
        self.reduce_type = ReduceType.L2Norm


class Sum(Reduce):
    
    def __init__(self, dim = 1):
        super().__init__(dim)
        self.reduce_type = ReduceType.Sum