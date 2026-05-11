import torch
from typing import Optional
from .context import Context
from ..abs import vTensor, FORMAT, vOp
from ..utils import Schedule


class Reshape(vOp):
    r"""
    Same-numel reshape of the inner two axes of a rank-3 indexer tensor.

    Given an input

    .. math::

        X \in \mathbb{R}^{S \times x_1 \times y_1},

    this op produces

    .. math::

        Y \in \mathbb{R}^{S \times x_2 \times y_2},
        \qquad \text{with} \qquad x_2 \cdot y_2 \;=\; x_1 \cdot y_1,

    by interpreting the flat ``x_1 * y_1`` elements row-major into the
    new ``(x_2, y_2)`` layout, independently for each of the ``S``
    slices. The leading axis is preserved and must be passed as
    ``-1`` to the constructor as a placeholder, mirroring
    ``torch.reshape``'s "infer this dim" convention.

    No data movement happens beyond the existing per-workload load /
    store — the loaded ``(W, x_1, y_1)`` block is reshaped at the
    Triton-tile level via :func:`tl.reshape` to ``(W, x_2, y_2)``.
    The constraint ``x_1 * y_1 == x_2 * y_2`` is enforced at
    :meth:`profile` time (and again by ``tl.reshape`` at compile
    time as a numel check).

    Output format rule: ``BATCHED`` iff the input is ``BATCHED``,
    otherwise ``RAGGED``. Format compatibility is enforced by the
    compiler's per-workload kernel.

    Constructor
    -----------
    ``Reshape(-1, x2, y2)`` — three integers; the leading must be
    ``-1`` (the leading ``S`` axis is preserved automatically).

    Attributes
    ----------
    x2, y2 : int
        Target inner-axis sizes.
    output_format : Optional[FORMAT]
        Resolved output format from :meth:`profile`.
    output_buffer : Optional[vTensor]
        Pure-metadata vTensor descriptor for the output (graph node).
    """

    def __init__(self, batch_dim: int, x2: int, y2: int):
        super().__init__()
        cls = self.__class__.__name__
        assert batch_dim == -1, (
            f"{cls}.__init__: leading dim must be -1 (auto-infer; the "
            f"leading S axis is preserved), got {batch_dim}"
        )
        assert isinstance(x2, int) and x2 >= 1, (
            f"{cls}.__init__: x2 must be a positive int, got {x2!r}"
        )
        assert isinstance(y2, int) and y2 >= 1, (
            f"{cls}.__init__: y2 must be a positive int, got {y2!r}"
        )
        self.x2 = x2
        self.y2 = y2

        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[vTensor] = None
        # Fused into the per-workload kernel.
        self.schedule = Schedule.W

    # ---------------- profile ----------------
    def profile(self, x: vTensor, ctx: Context) -> vTensor:
        r"""
        Validate, resolve format, and register this reshape in the
        indexer graph.

        Checks ``x.dim() == 3`` and the same-numel constraint
        ``x.shape[1] * x.shape[2] == self.x2 * self.y2``. The leading
        axis is preserved (the metadata vTensor is allocated with a
        placeholder leading dim — the runtime knows the actual one
        from the pipeline).
        """
        prefix = self._prefix()

        assert isinstance(x, vTensor), (
            f"{prefix}profile expects x to be vTensor, got {type(x)}"
        )
        assert x.dim() == 3, (
            f"{prefix}expected 3D input [S, x1, y1], "
            f"got ndim={x.dim()} shape={tuple(x.shape)}"
        )

        x1, y1 = x.shape[1], x.shape[2]
        in_numel = x1 * y1
        out_numel = self.x2 * self.y2
        assert in_numel == out_numel, (
            f"{prefix}same-numel reshape required: "
            f"x.shape[1]*x.shape[2] = {x1}*{y1} = {in_numel}  vs  "
            f"target x2*y2 = {self.x2}*{self.y2} = {out_numel}"
        )

        # Output is BATCHED iff the input is BATCHED; otherwise RAGGED.
        self.output_format = (
            FORMAT.BATCHED if x._format == FORMAT.BATCHED else FORMAT.RAGGED
        )

        # Pure-metadata vTensor — leading dim is a placeholder; the
        # runtime knows the actual ``S`` from the pipeline.
        self.output_buffer = vTensor(
            shape=(0, self.x2, self.y2),
            dtype=ctx.vortex_dtype,
            device=x.device,
            _format=self.output_format,
            tensor_id=len(ctx.tensor_list),
        )

        # Register in the indexer graph.
        ctx.tensor_list.append(self.output_buffer)
        ctx.output_tensor_to_op_list.append(len(ctx.op_list))
        ctx.op_list.append(self)
        ctx.op_to_input_tensor_list.append([x.tensor_id])
        ctx.op_to_output_tensor_list.append([self.output_buffer.tensor_id])

        return self.output_buffer
