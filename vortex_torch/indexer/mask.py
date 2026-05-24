import torch
from typing import Optional
from .context import Context
from ..abs import vTensor, FORMAT, vOp
from ..utils import Schedule


class MaskSlice(vOp):
    r"""
    Position-dependent slice mask over one inner axis.

    :Math:
        For target axis ``dim`` (``1`` → :math:`D_0`, ``2`` → :math:`D_1`) and
        index :math:`i` along it (other axes broadcast unchanged):

        .. math::

            Y_{\dots,i,\dots} = \begin{cases} \alpha, & \text{start} \le i < \text{end}, \\ \beta, & \text{otherwise}. \end{cases}
    :__init__: ``MaskSlice(start, end, dim, alpha=1.0, beta=0.0)`` — write
        :math:`\alpha` on ``[start, end)`` of axis ``dim`` (1 or 2) and
        :math:`\beta` elsewhere.
    :__call__: ``y = op(x, ctx=ctx)`` — ``x`` ``[S, D_0, D_1]`` → same shape. A
        pure **position** writer (``x`` values are unused); output is
        ``BATCHED`` iff ``x`` is.
    :Note: only ``dim ∈ {1, 2}`` (the packed ``S`` axis is structural).
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
    def profile(self, x: vTensor, ctx: Context) -> vTensor:
        prefix = self._prefix()
        assert isinstance(x, vTensor), (
            f"{prefix}profile expects x to be vTensor, got {type(x)}"
        )
        assert x.dim() == 3, (
            f"{prefix}expected 3D input [S, D0, D1], got shape={tuple(x.shape)}"
        )

        # Output is BATCHED iff the input is BATCHED; otherwise RAGGED.
        self.output_format = (
            FORMAT.BATCHED if x._format == FORMAT.BATCHED else FORMAT.RAGGED
        )

        dim_size = x.shape[self.dim]
        assert 0 <= self.start <= self.end <= dim_size, (
            f"{prefix}[start, end) = [{self.start}, {self.end}) out of "
            f"bounds for dim={self.dim} (size={dim_size})"
        )

        self.output_buffer = vTensor(
            shape=(0, x.shape[1], x.shape[2]),
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
