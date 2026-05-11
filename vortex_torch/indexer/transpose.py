import torch
from typing import Optional
from .context import Context
from ..abs import vTensor, FORMAT, vOp
from ..utils import Schedule

class Transpose(vOp):
    r"""
    Transpose op for rank-3 logical tensors.

    This operator transposes the last two dimensions of a rank-3 tensor
    while keeping the leading axis unchanged. The input is treated as

    .. math::

        X \in \mathbb{R}^{S \times D_0 \times D_1},

    and the output has logical shape

    .. math::

        Y \in \mathbb{R}^{S \times D_1 \times D_0},

    with

    .. math::

        Y[s, d_1, d_0] = X[s, d_0, d_1].

    The leading dimension :math:`S` may represent a true sequence axis or
    a packed axis (e.g. :math:`S_{\text{pack}} = \sum_b S_b`); the transpose
    is applied independently for each slice along that axis.

    Output format rule: ``BATCHED`` iff the input is ``BATCHED``,
    otherwise ``RAGGED``. Format compatibility is enforced by the
    compiler's per-workload kernel.

    Attributes
    ----------
    output_format : Optional[FORMAT]
        The output tensor format as determined in :meth:`profile`.
    output_buffer : Optional[torch.Tensor]
        Preallocated output tensor buffer with logical shape
        ``[S, D_1, D_0]``.
    """

    def __init__(self):
        super().__init__()
        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[torch.Tensor] = None
        self.schedule = Schedule.W

    # ---------------- profile ----------------
    def profile(self, x: vTensor, ctx: Context) -> vTensor:
        r"""
        Validate the input, allocate the output buffer, and return a
        :class:`vTensor` view.

        The input tensor is expected to have logical shape
        ``[S_in, D_0, D_1]``. The output buffer is allocated with shape

        .. math::

            [S, D_1, D_0],

        where :math:`S` is taken from ``ctx.max_num_pages`` to match the
        runtime configuration for the leading dimension.

        Parameters
        ----------
        x : vTensor
            Input tensor to be transposed, with logical shape
            ``[S_in, D_0, D_1]``.

        ctx : Context
            Execution context providing ``ctx.max_num_pages`` for the leading
            dimension and tracking auxiliary memory usage.

        Returns
        -------
        vTensor
            A ``vTensor`` view wrapping the internally allocated output
            buffer.

        Raises
        ------
        AssertionError
            If ``x`` is not a :class:`vTensor` or its rank is not 3.
        """
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
