import torch
from typing import Optional
from .context import Context
from ..abs import vTensor, FORMAT, vOp
from ..utils import Schedule

class Transpose(vOp):
    r"""
    Swap the two inner axes of each slice along the leading axis.

    :Math:
        .. math::

            Y_{s,d_1,d_0} = X_{s,d_0,d_1},

        applied independently per leading index :math:`s`.
    :__init__: ``Transpose()`` — no arguments.
    :__call__: ``y = op(x, ctx=ctx)`` — ``x`` ``[S, D_0, D_1]`` →
        ``[S, D_1, D_0]``. ``BATCHED`` iff the input is ``BATCHED``, else
        ``RAGGED``.
    """

    def __init__(self):
        super().__init__()
        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[torch.Tensor] = None
        self.schedule = Schedule.W

    # ---------------- profile ----------------
    def profile(self, x: vTensor, ctx: Context) -> vTensor:
        r"""Trace-time: validate ``x`` ``[S, D_0, D_1]``, resolve the output
        format, and return a ``vTensor`` view of the ``[S, D_1, D_0]``
        transpose."""
        prefix = self._prefix()

        # Type & rank checks
        assert isinstance(x, vTensor), f"{prefix}profile expects x to be vTensor, got {type(x)}"
        assert x.dim() == 3, (
            f"{prefix}expected 3D input [S, D0, D1], "
            f"got ndim={x.dim()} shape={tuple(x.shape)}"
        )

        # Output is BATCHED iff the input is BATCHED; otherwise RAGGED.
        self.output_format = (
            FORMAT.BATCHED if x._format == FORMAT.BATCHED else FORMAT.RAGGED
        )

        # Allocate output buffer: [S, D1, D0]
        # S is derived from runtime context (number of pages/tokens in the pipeline)
        S = ctx.max_num_pages
        D0, D1 = x.shape[1], x.shape[2]
        # Pure-metadata vTensor — no real allocation. The compiled code
        # supplies storage; we only need shape/dtype/device for codegen.
        self.output_buffer = vTensor(
            shape=(S, D1, D0),
            dtype=ctx.vortex_dtype,
            device=x.device,
            _format=self.output_format,
            tensor_id=-1,  # set by caller / graph registration if needed
        )

        for t in [x]:
            if t._format == FORMAT.PAGED:
                ctx.add_aux_flops(
                    t.shape[1] * t.shape[2]
                )

        return self.output_buffer
