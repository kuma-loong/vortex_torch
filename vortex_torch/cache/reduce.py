import torch
from ..op import vOp
from .context import Context
from .triton_kernels import mean_launcher
from ..tensor import vTensor, as_vtensor, FORMAT
from typing import Tuple, Dict, Callable

class Mean(vOp):
    """
    Mean reduction op with format-dispatch.
    Contract preserved:
      - profile(x: vTensor, output: vTensor, loc: torch.Tensor, ctx) -> vTensor
      - execute(x: torch.Tensor, output: torch.Tensor, loc: torch.Tensor, ctx) -> torch.Tensor
    NOTE: output=None is NOT supported at the moment.
    """

    # Dispatch table: (x_format, out_format) -> (impl, final_out_format)
    _impl_map: Dict[Tuple[FORMAT, FORMAT], Tuple[Callable, FORMAT]] = {
        (FORMAT.PAGED, FORMAT.PAGED): (mean_launcher, FORMAT.RAGGED),
        # Add more (x_fmt, o_fmt) -> (impl, out_fmt) pairs as needed.
    }

    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim
        self.impl: Callable | None = None
        self.output_format: FORMAT | None = None

        # Validate reduction dimension at construction time.
        assert self.dim in (1, 2), f"Mean.__init__: dim must be 1 or 2, got dim={self.dim}"

    # --------------------------------------------------------------------- #
    # profile: validate, pick impl/format, and return the provided vTensor
    # --------------------------------------------------------------------- #
    def profile(self, x: vTensor, output: vTensor, loc: torch.Tensor, ctx: Context) -> vTensor:
        # Type checks (keep consistent with your system rule that profile takes vTensor)
        assert isinstance(x, vTensor), f"Mean.profile: x must be vTensor, got {type(x)}"
        assert isinstance(output, vTensor), f"Mean.profile: output must be vTensor, got {type(output)}"
        assert isinstance(loc, torch.Tensor), f"Mean.profile: loc must be torch.Tensor, got {type(loc)}"

        # Explicitly disallow output=None for now
        assert output is not None, "Mean.profile: output=None is not supported at the moment"

        # Basic rank checks (3D tensors like [B, 1, D] or [B, N, 1] are expected in your codebase)
        assert x.dim() == 3, f"Mean.profile: x must be 3D, got x.ndim={x.dim()} with shape={tuple(x.shape)}"
        assert output.dim() == 3, f"Mean.profile: output must be 3D, got output.ndim={output.dim()} with shape={tuple(output.shape)}"

        # Format-based dispatch
        x_fmt, o_fmt = x._format, output._format
        assert (x_fmt, o_fmt) in self._impl_map, (
            f"Mean.profile: no implementation for (x_fmt={x_fmt}, o_fmt={o_fmt}). "
            f"Available keys: {list(self._impl_map.keys())}"
        )
        self.impl, self.output_format = self._impl_map[(x_fmt, o_fmt)]

        # Shape compatibility checks depending on reduction dim
        if self.dim == 1:
            # Reducing along dim=1: output should keep dim-2 equal to x.dim-2, and dim-1 == 1
            assert output.shape[1] == 1, (
                f"Mean.profile(dim=1): expected output.shape[1] == 1, got output.shape={tuple(output.shape)}"
            )
            assert output.shape[2] == x.shape[2], (
                f"Mean.profile(dim=1): expected output.shape[2] == x.shape[2], "
                f"got output.shape[2]={output.shape[2]}, x.shape[2]={x.shape[2]}"
            )
        else:  # self.dim == 2
            # Reducing along dim=2: output should keep dim-1 equal to x.dim-1, and dim-2 == 1
            assert output.shape[2] == 1, (
                f"Mean.profile(dim=2): expected output.shape[2] == 1, got output.shape={tuple(output.shape)}"
            )
            assert output.shape[1] == x.shape[1], (
                f"Mean.profile(dim=2): expected output.shape[1] == x.shape[1], "
                f"got output.shape[1]={output.shape[1]}, x.shape[1]={x.shape[1]}"
            )

        # Optional: device consistency at profile-time (helps catch host/device mismatches early)
        assert x.device == output.device, (
            f"Mean.profile: x and output must be on the same device, "
            f"got x.device={x.device}, output.device={output.device}"
        )

        # Return the same vTensor (contract)
        return output

    # --------------------------------------------------------------------- #
    # execute: run impl and return the (plain) output tensor
    # --------------------------------------------------------------------- #
    def execute(self, x: torch.Tensor, output: torch.Tensor, loc: torch.Tensor, ctx: Context) -> torch.Tensor:
        # Must have selected impl in profile()
        assert self.impl is not None, "Mean.execute: called before profile() or profile() failed (impl is None)"

        # Launch the kernel/implementation (keep your original call signature)
        self.impl(x, output, loc, ctx.head_num, ctx.page_size, self.dim)

        # execute returns a plain torch.Tensor by design
        return output