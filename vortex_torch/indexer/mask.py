import torch
from typing import Dict, Optional
from .context import Context
from ..abs import vTensor, FORMAT, vOp
from ..utils import Schedule


class MaskSlice(vOp):
    r"""
    Position-dependent slice mask over one inner axis of a rank-3 tensor.

    For an input tensor

    .. math::

        X \in \mathbb{R}^{S \times D_0 \times D_1}

    and a target axis :attr:`dim` (``1`` for :math:`D_0`, ``2`` for
    :math:`D_1`), ``MaskSlice`` writes

    .. math::

        Y[\ldots, i, \ldots] =
        \begin{cases}
            \alpha, & \text{if } \text{start} \le i < \text{end}, \\
            \beta,  & \text{otherwise},
        \end{cases}

    where :math:`i` is the index along the chosen axis. The other axes
    are broadcast unchanged. The output shape exactly matches
    :attr:`x.shape`.

    Notes
    -----
    - Only ``dim in {1, 2}`` is supported. The packed ``S`` axis is
      structural (its global index is only known after unpacking
      workload metadata), so a position-dependent mask along it does
      not fit cleanly into the inline W-kernel template; model that
      in user code via an explicit index op instead.
    - The input ``x`` is read by the surrounding workload kernel but its
      values are not used by the output — ``MaskSlice`` is a pure
      position-based writer. The dependency edge is preserved so the
      compiler still fuses it with its neighbours.

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

    _impl_map: Dict[FORMAT, FORMAT] = {
        FORMAT.RAGGED: FORMAT.RAGGED,
        FORMAT.PAGED: FORMAT.RAGGED,
        FORMAT.BATCHED: FORMAT.BATCHED,
    }

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

        x_fmt = x._format
        assert x_fmt in self._impl_map, (
            f"{prefix}no implementation for x_fmt={x_fmt}. "
            f"Available: {list(self._impl_map.keys())}"
        )
        self.output_format = self._impl_map[x_fmt]

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
