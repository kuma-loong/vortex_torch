import torch
from typing import Tuple, Dict, Callable, Optional
from .context import Context
from ..abs import vTensor, as_vtensor, FORMAT, vOp
from .triton_kernels.transpose_impl import transpose_rr

class Transpose(vOp):
    """
    Transpose dispatcher for rank-3 logical tensors [S, D0, D1] -> [S, D1, D0].
    Dispatch is keyed by the input vTensor's format only.
    """

    # Implementation dispatch table: keyed only by x_format.
    # Value: (callable_impl, resolved_output_format)
    _impl_map: Dict[FORMAT, Tuple[Callable, FORMAT]] = {
        FORMAT.RAGGED: (transpose_rr, FORMAT.RAGGED),
        # Add more entries if you support other formats, e.g.:
        # FORMAT.PAGED: (transpose_pp, FORMAT.PAGED),
    }

    def __init__(self):
        super().__init__()
        self.impl: Optional[Callable] = None
        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[torch.Tensor] = None

    # ---------------- profile ----------------
    def profile(self, x: vTensor, ctx: Context) -> vTensor:
        """
        Validate inputs, choose implementation by x._format, allocate output buffer,
        and return an as_vtensor view with the resolved format.
        """
        prefix = self._prefix()

        # Type & rank checks
        assert isinstance(x, vTensor), f"{prefix}profile expects x to be vTensor, got {type(x)}"
        assert x.dim() == 3, f"{prefix}expected 3D input [S, D0, D1], got ndim={x.dim()} shape={tuple(x.shape)}"

        # Dispatch by input format
        x_fmt = x._format
        assert x_fmt in self._impl_map, (
            f"{prefix}no implementation for x_fmt={x_fmt}. "
            f"Available keys: {list(self._impl_map.keys())}"
        )
        self.impl, self.output_format = self._impl_map[x_fmt]

        # Allocate output buffer: [S, D1, D0]
        # S is derived from runtime context (number of pages/tokens in your pipeline)
        S = ctx.max_num_pages
        D0, D1 = x.shape[1], x.shape[2]
        self.output_buffer = torch.empty(
            (S, D1, D0),
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
        Run the selected implementation using the internal output buffer.
        """
        prefix = self._prefix()
        assert self.impl is not None, f"{prefix}execute called before profile() (impl is None)"
        assert self.output_buffer is not None, f"{prefix}internal output buffer is None; did profile() run?"

        # Run selected implementation; keep original call signature.
        # Expected signature: impl(x, output, ctx)
        self.impl(x, self.output_buffer, ctx)
        return self.output_buffer