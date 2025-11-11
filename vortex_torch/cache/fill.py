import torch
from ..abs import vOp
from .context import Context
from .triton_kernels.fill_impl import fill_p
from ..abs import vTensor, FORMAT
from typing import Tuple, Dict, Callable, Optional

class Fill(vOp):
    """
    In-place page-wise fill dispatcher.
    - Fills page tiles in `x` with a scalar `alpha` when the end-of-page token is reached.
    - Dispatch is keyed only by the input vTensor's format.
    - No output buffer is allocated; the operation is in-place on `x`.
    """

    # Implementation registry keyed by x_format.
    # Expected impl signature: impl(x: torch.Tensor, loc: torch.Tensor, ctx: Context, alpha: float) -> None
    _impl_map: Dict[FORMAT, Callable] = {
        FORMAT.PAGED: fill_p,  # e.g., wraps `fill_p_kernel` (page-major in-place fill)
        # Add more entries if you support other layouts, e.g.:
        # FORMAT.RAGGED: fill_r,
    }

    def __init__(self, alpha: float = 0.0):
        """
        Parameters
        ----------
        alpha : scalar fill value.
        """
        super().__init__()
        self.alpha = alpha
        self.impl: Optional[Callable] = None

    # ---------------- profile ----------------
    def profile(self, x: vTensor, loc: torch.Tensor, ctx: Context) -> vTensor:
        """
        Validate inputs and select implementation. Since the op is in-place,
        no output buffer is allocated; returns `x` as a vTensor view.
        """
        prefix = self._prefix()

        # Type & rank checks
        assert isinstance(x, vTensor), f"{prefix}profile expects x to be vTensor, got {type(x)}"
        assert isinstance(loc, torch.Tensor), f"{prefix}profile expects loc to be torch.Tensor, got {type(loc)}"
        assert x.dim() == 3, f"{prefix}expected 3D input [S, D0, D1], got ndim={x.dim()} shape={tuple(x.shape)}"

        # Dispatch by input format
        x_fmt = x._format
        assert x_fmt in self._impl_map, (
            f"{prefix}no implementation for x_fmt={x_fmt}. "
            f"Available keys: {list(self._impl_map.keys())}"
        )
        self.impl = self._impl_map[x_fmt]

        # In-place: return the same vTensor view
        return x

    # ---------------- execute ----------------
    def execute(self, x: torch.Tensor, loc: torch.Tensor, ctx: Context) -> torch.Tensor:
        """
        Run the selected in-place implementation and return `x`.
        """
        prefix = self._prefix()
        assert self.impl is not None, f"{prefix}execute called before profile() (impl is None)"

        # Expected signature: impl(x, loc, ctx, alpha)
        self.impl(x, loc, ctx, self.alpha)
        return x