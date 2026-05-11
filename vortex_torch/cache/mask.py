import torch
from typing import Optional
from .context import Context
from ..abs import vTensor, FORMAT, vOp
from ..utils import Schedule


class MaskSlice(vOp):
    r"""
    Position-dependent slice mask over one inner axis of a rank-3 tensor.

    For an input tensor

    .. math::

        X \in \mathbb{R}^{B \times D_0 \times D_1}

    and a target axis :attr:`dim` (``1`` for :math:`D_0`, ``2`` for
    :math:`D_1`), ``MaskSlice`` writes

    .. math::

        Y[\ldots, i, \ldots] =
        \begin{cases}
            \alpha, & \text{if } \text{start} \le i < \text{end}, \\
            \beta,  & \text{otherwise},
        \end{cases}

    where :math:`i` is the index along the chosen axis. The other axes
    are broadcast unchanged. The output shape matches :attr:`x.shape`.

    Parameters
    ----------
    start : int
        Inclusive lower bound along ``dim``.
    end : int
        Exclusive upper bound along ``dim``.
    dim : int
        Axis to slice (1 = :math:`D_0`, 2 = :math:`D_1`).
    alpha : float
        Value written for positions inside ``[start, end)``.
    beta : float
        Value written for positions outside.
    """

    def __init__(
        self,
        start: int,
        end: int,
        dim: int,
        alpha: float = 1.0,
        beta: float = 0.0,
    ):
        super().__init__()
        self.start = int(start)
        self.end = int(end)
        self.dim = int(dim)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[vTensor] = None
        # Cache MaskSlice fuses into the per-block kernel.
        self.schedule = Schedule.W

        prefix = self._prefix()
        assert self.dim in (1, 2), (
            f"{prefix}__init__: dim must be 1 or 2, got dim={self.dim}"
        )
        assert self.start <= self.end, (
            f"{prefix}__init__: require start <= end, got "
            f"start={self.start}, end={self.end}"
        )

    # ---------------- profile ----------------
    def profile(
        self,
        x: vTensor,
        output: Optional[vTensor],
        loc: torch.Tensor,
        ctx: Context,
    ) -> vTensor:
        prefix = self._prefix()

        assert isinstance(x, vTensor), f"{prefix}x must be vTensor, got {type(x)}"
        assert isinstance(loc, torch.Tensor), (
            f"{prefix}loc must be torch.Tensor, got {type(loc)}"
        )
        assert x.dim() == 3, f"{prefix}x must be 3D, got {tuple(x.shape)}"

        B, N, D = x.shape

        # Bounds check against the chosen axis.
        dim_size = x.shape[self.dim]
        assert 0 <= self.start <= self.end <= dim_size, (
            f"{prefix}[start, end) = [{self.start}, {self.end}) out of "
            f"bounds for dim={self.dim} (size={dim_size})"
        )

        # Case A: output not provided -> allocate a RAGGED metadata buffer.
        if output is None:
            self.output_format = FORMAT.RAGGED

            eff_B = ctx.max_new_tokens_per_batch * ctx.head_num
            self.output_buffer = vTensor(
                shape=(eff_B, N, D),
                dtype=ctx.vortex_dtype,
                device=x.device,
                _format=self.output_format,
                tensor_id=len(ctx.tensor_list),
            )
            ctx.tensor_list.append(self.output_buffer)
            ctx.output_tensor_to_op_list.append(len(ctx.op_list))
            ctx.op_list.append(self)
            ctx.op_to_input_tensor_list.append([x.tensor_id])
            ctx.op_to_output_tensor_list.append([self.output_buffer.tensor_id])
            return self.output_buffer

        # Case B: caller-provided output -> output_format follows output._format.
        assert isinstance(output, vTensor), (
            f"{prefix}output must be vTensor, got {type(output)}"
        )
        assert output.dim() == 3, (
            f"{prefix}output must be 3D, got {tuple(output.shape)}"
        )
        assert output._format in (FORMAT.PAGED, FORMAT.RAGGED), (
            f"{prefix}output._format must be PAGED or RAGGED, got {output._format}"
        )
        self.output_format = output._format
        # Shape-preserving op; check only the inner dims since the leading
        # axis may differ between B-major and S-major views of the cache.
        assert output.shape[1] == N and output.shape[2] == D, (
            f"{prefix}output shape mismatch. Expected (*,{N},{D}), "
            f"got {tuple(output.shape)}"
        )

        # Register in the cache graph — claim the caller's output tensor.
        ctx.output_tensor_to_op_list[output.tensor_id] = len(ctx.op_list)
        ctx.op_list.append(self)
        ctx.op_to_input_tensor_list.append([x.tensor_id])
        ctx.op_to_output_tensor_list.append([output.tensor_id])

        return output
