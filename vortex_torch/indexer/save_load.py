import torch
from typing import Tuple, Dict, Callable, Optional
from .context import Context
from ..abs import vTensor, as_vtensor, FORMAT, vOp
from .triton_kernels.save_load_impl import save_rp, load_pr

class Save(vOp):
    """
    Save dispatcher.
    - Copy/convert from x (input vTensor) to o (preallocated output vTensor).
    - Dispatch is keyed by x's format only.
    - No buffer allocation happens here; `o` is validated and used directly.
    """

    # Implementation dispatch table: keyed only by x_format.
    # Value: (callable_impl, resolved_output_format)
    _impl_map: Dict[FORMAT, Tuple[Callable, FORMAT]] = {
        FORMAT.RAGGED: (save_rp, FORMAT.PAGED),
        # Add more entries if you support other formats.
    }

    def __init__(self):
        super().__init__()
        self.impl: Optional[Callable] = None
        self.output_format: Optional[FORMAT] = None

    # ---------------- profile ----------------
    def profile(self, x: vTensor, o: vTensor, ctx: Context) -> vTensor:
        """
        Validate inputs, select implementation, and return `o` as the vTensor view.
        """
        prefix = self._prefix()

        # Type & rank checks
        assert isinstance(x, vTensor), f"{prefix}profile expects x to be vTensor, got {type(x)}"
        assert isinstance(o, vTensor), f"{prefix}profile expects o to be vTensor, got {type(o)}"
        assert x.dim() == 3, f"{prefix}expected 3D x [S, D0, D1], got {tuple(x.shape)}"
        assert o.dim() == 3, f"{prefix}expected 3D o [S, D0, D1], got {tuple(o.shape)}"

        # Shape checks: D0/D1 must match (S may differ by layout; your impl handles it)
        assert x.shape[1] == o.shape[1], (
            f"{prefix}expected matching D0: x.shape[1]={x.shape[1]} vs o.shape[1]={o.shape[1]}"
        )
        assert x.shape[2] == o.shape[2], (
            f"{prefix}expected matching D1: x.shape[2]={x.shape[2]} vs o.shape[2]={o.shape[2]}"
        )

        # Dispatch by x format
        x_fmt = x._format
        assert x_fmt in self._impl_map, (
            f"{prefix}no implementation for x_fmt={x_fmt}. "
            f"Available keys: {list(self._impl_map.keys())}"
        )
        self.impl, self.output_format = self._impl_map[x_fmt]

        # Output format must match the resolved format from dispatch
        assert o._format == self.output_format, (
            f"{prefix}output format mismatch. Expected {self.output_format}, got {o._format}"
        )

        # Device consistency
        assert x.device == o.device, (
            f"{prefix}x and o must be on the same device "
            f"(x.device={x.device}, o.device={o.device})"
        )

        # Save is out-of-place but writes into `o`; return `o` view
        return o

    # ---------------- execute ----------------
    def execute(self, x: torch.Tensor, o: torch.Tensor, ctx: Context) -> torch.Tensor:
        """
        Run the selected implementation that copies/converts from x to o.
        """
        prefix = self._prefix()
        assert self.impl is not None, f"{prefix}execute called before profile() (impl is None)"

        # Expected signature: impl(x, o, ctx)
        self.impl(x, o, ctx)
        return o


class Load(vOp):
    """
    Load dispatcher.
    - Copy/convert from x (input vTensor) to an internally allocated output buffer.
    - Dispatch is keyed by x's format only.
    - Buffer is allocated in profile(); execute() fills it and returns it.
    """

    # Implementation dispatch table: keyed only by x_format.
    # Value: (callable_impl, resolved_output_format)
    _impl_map: Dict[FORMAT, Tuple[Callable, FORMAT]] = {
        FORMAT.PAGED: (load_pr, FORMAT.RAGGED),
        # Add more entries if you support other formats.
    }

    def __init__(self):
        super().__init__()
        self.impl: Optional[Callable] = None
        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[torch.Tensor] = None

    # ---------------- profile ----------------
    def profile(self, x: vTensor, ctx: Context) -> vTensor:
        """
        Validate input, select implementation, allocate output buffer,
        and return an as_vtensor view with the resolved format.
        """
        prefix = self._prefix()

        # Type & rank checks
        assert isinstance(x, vTensor), f"{prefix}profile expects x to be vTensor, got {type(x)}"
        assert x.dim() == 3, f"{prefix}expected 3D x [S, D0, D1], got {tuple(x.shape)}"

        # Dispatch by x format
        x_fmt = x._format
        assert x_fmt in self._impl_map, (
            f"{prefix}no implementation for x_fmt={x_fmt}. "
            f"Available keys: {list(self._impl_map.keys())}"
        )
        self.impl, self.output_format = self._impl_map[x_fmt]

        # Allocate output buffer [S_out, D0, D1].
        # Here S_out comes from runtime context (e.g., number of tokens/pages after transform).
        S_out = ctx.max_num_pages
        D0, D1 = x.shape[1], x.shape[2]
        self.output_buffer = torch.empty(
            (S_out, D0, D1),
            device=x.device,
            dtype=x.dtype,
        )
        ctx.add_aux_memory(self.output_buffer)

        # Return vTensor view with dispatched output format
        return as_vtensor(self.output_buffer, self.output_format)

    # ---------------- execute ----------------
    def execute(self, x: torch.Tensor, ctx: Context) -> torch.Tensor:
        """
        Run the selected implementation into the internally allocated buffer.
        """
        prefix = self._prefix()
        assert self.impl is not None, f"{prefix}execute called before profile() (impl is None)"
        assert self.output_buffer is not None, f"{prefix}internal output buffer is None; did profile() run?"

        # Expected signature: impl(x, output, ctx)
        self.impl(x, self.output_buffer, ctx)
        return self.output_buffer