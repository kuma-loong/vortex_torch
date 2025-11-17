import torch
from ..abs import vOp
from .context import Context
from .triton_kernels.reduce_impl import reduce_pp, reduce_rp, reduce_pr, reduce_rr
from ..abs import vTensor, FORMAT, as_vtensor
from ..utils import ReduceType
from typing import Tuple, Dict, Callable, Optional


class Reduce(vOp):
    """
    Reduction dispatcher (dim ∈ {1,2}, type = Mean/Max/Min/L2, etc. decided in impl).
    Dispatch key: (x_format, o_format) -> (impl, resolved_output_format)

    Policy:
      - If `output` is None, prefer an implementation with o_format == FORMAT.RAGGED.
      - Otherwise, require an exact (x_fmt, o_fmt) match.
      - Input tensors are rank-3: [B, N, D]; the reduced axis collapses to 1.
    """

    # Consistent 2-tuple dispatch table
    _impl_map: Dict[Tuple[FORMAT, FORMAT], Tuple[Callable, FORMAT]] = {
        (FORMAT.PAGED,  FORMAT.PAGED):  (reduce_pp, FORMAT.PAGED),
        (FORMAT.RAGGED, FORMAT.PAGED):  (reduce_rp, FORMAT.PAGED),
        (FORMAT.PAGED,  FORMAT.RAGGED): (reduce_pr, FORMAT.RAGGED),
        (FORMAT.RAGGED, FORMAT.RAGGED): (reduce_rr, FORMAT.RAGGED),
    }

    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim
        self.reduce_type: Optional[ReduceType] = None
        self.impl: Optional[Callable] = None
        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[torch.Tensor] = None
        # Validate reduction dimension at construction time.
        cls = self.__class__.__name__
        assert self.dim in (1, 2), f"{cls}.__init__: dim must be 1 or 2, got dim={self.dim}"

    # ------------------------------ helpers ------------------------------ #
    def _infer_impl_ragged(self, x_fmt: FORMAT) -> Tuple[Callable, FORMAT]:
        """
        Default inference when output is None:
        choose (x_fmt, FORMAT.RAGGED). Raise if missing.
        """
        key = (x_fmt, FORMAT.RAGGED)
        assert key in self._impl_map, (
            f"{self._prefix()}no RAGGED-output implementation for x_fmt={x_fmt}. "
            f"Available keys: {list(self._impl_map.keys())}"
        )
        return self._impl_map[key]  # -> (impl, out_fmt)

    # --------------------------------------------------------------------- #
    # profile: validate, pick impl/format, and return the provided vTensor
    # --------------------------------------------------------------------- #
    def profile(
        self, x: vTensor, output: Optional[vTensor], loc: torch.Tensor, ctx: Context
    ) -> vTensor:
        prefix = self._prefix()

        # --- type & rank checks ---
        assert isinstance(x, vTensor), f"{prefix}x must be vTensor, got {type(x)}"
        assert isinstance(loc, torch.Tensor), f"{prefix}loc must be torch.Tensor, got {type(loc)}"
        assert x.dim() == 3, f"{prefix}x must be 3D, got ndim={x.dim()} shape={tuple(x.shape)}"

        x_fmt = x._format

        # Compute expected output (N, D) given reduction dim
        # dim==1 -> reduce rows: keep D, set N=1
        # dim==2 -> reduce cols: keep N, set D=1
        exp_N = 1 if self.dim == 1 else x.shape[1]
        exp_D = 1 if self.dim == 2 else x.shape[2]

        # Case A: output not provided -> infer impl (RAGGED) and allocate buffer
        if output is None:
            self.impl, self.output_format = self._infer_impl_ragged(x_fmt)

            B = ctx.max_new_tokens_per_batch * ctx.head_num
            self.output_buffer = torch.empty(
                (B, exp_N, exp_D),
                device=x.device,
                dtype=x.dtype,
            )
            ctx.add_aux_memory(self.output_buffer)
            return as_vtensor(self.output_buffer, self.output_format)

        # Case B: output provided -> validate and pick exact impl by (x_fmt, o_fmt)
        assert isinstance(output, vTensor), f"{prefix}output must be vTensor, got {type(output)}"
        assert output.dim() == 3, f"{prefix}output must be 3D, got ndim={output.dim()} shape={tuple(output.shape)}"

        o_fmt = output._format
        key = (x_fmt, o_fmt)
        assert key in self._impl_map, (
            f"{prefix}no implementation for (x_fmt={x_fmt}, o_fmt={o_fmt}). "
            f"Available keys: {list(self._impl_map.keys())}"
        )
        self.impl, self.output_format = self._impl_map[key]

        # Shape checks per reduction dim
        if self.dim == 1:
            # Expect (*, 1, x.D)
            assert output.shape[1] == 1, (
                f"{prefix}profile(dim=1): expected output.shape[1] == 1, got {tuple(output.shape)}"
            )
            assert output.shape[2] == x.shape[2], (
                f"{prefix}profile(dim=1): expected output.shape[2] == x.shape[2], "
                f"got {output.shape[2]} vs {x.shape[2]}"
            )
        else:  # self.dim == 2
            # Expect (*, x.N, 1)
            assert output.shape[2] == 1, (
                f"{prefix}profile(dim=2): expected output.shape[2] == 1, got {tuple(output.shape)}"
            )
            assert output.shape[1] == x.shape[1], (
                f"{prefix}profile(dim=2): expected output.shape[1] == x.shape[1], "
                f"got {output.shape[1]} vs {x.shape[1]}"
            )

        # Device consistency
        assert x.device == output.device, (
            f"{prefix}x and output must be on the same device "
            f"(x.device={x.device}, output.device={output.device})"
        )

        return output

    # --------------------------------------------------------------------- #
    # execute: run impl and return the (plain) output tensor
    # --------------------------------------------------------------------- #
    def execute(
        self, x: torch.Tensor, output: Optional[torch.Tensor], loc: torch.Tensor, ctx: Context
    ) -> torch.Tensor:
        prefix = self._prefix()

        # Must have selected impl in profile()
        assert self.impl is not None, f"{prefix}called before profile() (impl is None)"

        if output is None:
            assert self.output_buffer is not None, f"{prefix}internal output buffer is None; did profile() run?"
            output = self.output_buffer

        # Launch the kernel/implementation: impl(x, output, loc, ctx, dim, reduce_type)
        self.impl(x, output, loc, ctx, self.dim, self.reduce_type)
        return output

    

class Mean(Reduce):
    def __init__(self, dim = 1):
        super().__init__(dim)
        self.reduce_type = ReduceType.Mean
    


class Max(Reduce):
    def __init__(self, dim = 1):
        super().__init__(dim)
        self.reduce_type = ReduceType.Max
        


class Min(Reduce):
    def __init__(self, dim = 1):
        super().__init__(dim)
        self.reduce_type = ReduceType.Min



class L2Norm(Reduce):
    def __init__(self, dim = 1):
        super().__init__(dim)
        self.reduce_type = ReduceType.L2Norm

class Sum(Reduce):
    
    def __init__(self, dim = 1):
        super().__init__(dim)
        self.reduce_type = ReduceType.Sum