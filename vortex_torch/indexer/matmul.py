import torch
from typing import Optional
from .context import Context
from ..abs import vTensor, FORMAT, vOp
from ..utils import Schedule


class GeMV(vOp):
    r"""
    General matrix-vector multiplication (GEMV) dispatcher.

    This operator computes a *piecewise* batched matrix-vector product.
    Let

    .. math::

        X \in \mathbb{R}^{B \times 1 \times D}, \qquad
        Y \in \mathbb{R}^{S_{\text{pack}} \times 1 \times D},

    where the ``S``-axis of :math:`Y` is a concatenation of batch-wise
    segments

    .. math::

        S_{\text{pack}} = \sum_{i=0}^{B-1} S_i, \qquad
        Y =
        \begin{bmatrix}
            Y_0 \\
            Y_1 \\
            \vdots \\
            Y_{B-1}
        \end{bmatrix},

    with

    .. math::

        Y_i \in \mathbb{R}^{S_i \times 1 \times D}, \qquad
        X_i = X[i, 0, :] \in \mathbb{R}^{1 \times D}.

    For each batch index :math:`i \in \{0,\dots,B-1\}`, we define

    .. math::

        O_i = Y_i X_i^{\mathsf{T}} \in \mathbb{R}^{S_i \times 1 \times 1},

    and the overall output is the concatenation

    .. math::

        O =
        \begin{bmatrix}
            O_0 \\
            O_1 \\
            \vdots \\
            O_{B-1}
        \end{bmatrix}
        \in \mathbb{R}^{S_{\text{pack}} \times 1 \times 1}.

    In the runtime, :math:`S_{\text{pack}}` is given by
    ``ctx.max_num_pages``. Output format rule: ``BATCHED`` iff both
    inputs are ``BATCHED`` (both have their ``S`` axis already collapsed
    to 1), otherwise ``RAGGED``. Format compatibility is enforced by
    the compiler's per-workload kernel.

    Attributes
    ----------
    output_format : Optional[FORMAT]
        The output tensor format as determined in :meth:`profile`.

    output_buffer : Optional[torch.Tensor]
        Preallocated output tensor buffer of shape ``[S_pack, 1, 1]``.
    """

    def __init__(self):
        super().__init__()
        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[torch.Tensor] = None
        self.schedule = Schedule.W
    # ---------------- profile ----------------
    def profile(self, x: vTensor, y: vTensor, ctx: Context) -> vTensor:
        r"""
        Validate inputs, allocate the output buffer, and return a
        :class:`vTensor` view.

        The method enforces the logical shapes

        - ``x``: ``[B, 1, D]``
        - ``y``: ``[S_pack, 1, D]``

        and checks that the last dimensions match. The output buffer is
        allocated with shape ``[S_pack, 1, 1]``, where ``S_pack`` is taken
        from the runtime context as ``ctx.max_num_pages``.
        """
        prefix = self._prefix()

        # Type checks
        assert isinstance(x, vTensor), f"{prefix}profile expects x to be vTensor, got {type(x)}"
        assert isinstance(y, vTensor), f"{prefix}profile expects y to be vTensor, got {type(y)}"

        # Rank/shape checks
        assert x.dim() == 3 and y.dim() == 3, (
            f"{prefix}expected 3D inputs; got x.ndim={x.dim()}, y.ndim={y.dim()}"
        )
        assert x.shape[1] == 1, f"{prefix}expected x.shape[1] == 1, got {tuple(x.shape)}"
        assert y.shape[1] == 1, f"{prefix}expected y.shape[1] == 1, got {tuple(y.shape)}"
        assert x.shape[2] == y.shape[2], (
            f"{prefix}last dimension mismatch: x.shape[2]={x.shape[2]} vs y.shape[2]={y.shape[2]}"
        )

        # Output is BATCHED iff both inputs are BATCHED; otherwise RAGGED.
        self.output_format = (
            FORMAT.BATCHED
            if (x._format == FORMAT.BATCHED and y._format == FORMAT.BATCHED)
            else FORMAT.RAGGED
        )
        # Pure-metadata vTensor — no torch.empty allocation needed.
        self.output_buffer = vTensor(
            shape=(0, 1, 1),
            dtype=ctx.vortex_dtype,
            device=x.device,
            _format=self.output_format,
            tensor_id=len(ctx.tensor_list),
        )

        # Track auxiliary memory and graph structure in the context
        ctx.tensor_list.append(self.output_buffer)  # Track the output buffer in the context
        ctx.output_tensor_to_op_list.append(len(ctx.op_list))  # Map the output tensor to this operation
        ctx.op_list.append(self)  # Track this operation in the context
        ctx.op_to_input_tensor_list.append([x.tensor_id, y.tensor_id])  # Map this op to its input tensors
        ctx.op_to_output_tensor_list.append([self.output_buffer.tensor_id])  # Map this op to its output tensor

        return self.output_buffer



# ------------------------------ GeMM ------------------------------ #
class GeMM(vOp):
    r"""
    General matrix-matrix multiplication (GeMM) dispatcher.

    Logically this computes, for each logical ``S``-slice, a matrix-matrix
    product

    .. math::

        O[s] = Y[s] X[s]^{\mathsf{T}}, \quad s = 0, \dots, S-1,

    with slice-wise shapes

    .. math::

        X[s] \in \mathbb{R}^{N_x \times K}, \quad
        Y[s] \in \mathbb{R}^{N_y \times K}, \quad
        O[s] \in \mathbb{R}^{N_y \times N_x}.

    In the packed 3D representation used by this dispatcher:

    - ``Y`` has logical shape ``[S, N_y, K]``.
    - ``X`` has logical shape ``[L_x, N_x, K]``, where the leading
      dimension :math:`L_x` can represent **either**:

      * a batch axis :math:`B` (when ``x_format == FORMAT.BATCHED``), or
      * the same ``S`` axis as ``Y`` (when ``x_format`` is ragged/paged and
        already laid out per-page).

      This is why the code comments use ``X: [B/S, N_x, K]``: the first
      dimension is interpreted as either a batch size :math:`B` or an
      ``S``-like logical page index, depending on the format.

    - The output tensor ``O`` has logical shape ``[S, N_y, N_x]``.

    At runtime, the logical ``S`` is taken from ``ctx.max_num_pages``.
    Output format rule: ``BATCHED`` iff both inputs are ``BATCHED``,
    otherwise ``RAGGED``. Format compatibility is enforced by the
    compiler's per-workload kernel.

    Attributes
    ----------
    output_format : Optional[FORMAT]
        The output tensor format as determined in :meth:`profile`.

    output_buffer : Optional[torch.Tensor]
        Preallocated output tensor buffer of shape ``[S, N_y, N_x]``.
    """

    def __init__(self):
        super().__init__()
        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[torch.Tensor] = None
        self.schedule = Schedule.W

    # ---------------- profile ----------------
    def profile(self, x: vTensor, y: vTensor, ctx: Context) -> vTensor:
        r"""
        Validate inputs, allocate the output buffer, and return a
        :class:`vTensor` view.

        The method enforces that both inputs are rank-3 tensors and that the
        inner dimension :math:`K` matches:

        - ``x``: ``[B_or_S, N_x, K]``

          *When* ``x_format == FORMAT.BATCHED``, the leading dimension is a
          batch size :math:`B`. For ragged/paged formats, it may conceptually
          coincide with :math:`S`.

        - ``y``: ``[S, N_y, K]``

        The output buffer is allocated with shape ``[S, N_y, N_x]``, where
        ``S`` is taken from the runtime context as ``ctx.max_num_pages``.

        Parameters
        ----------
        x : vTensor
            Right-hand operand (transposed in the mathematical view), with
            shape ``[B_or_S, N_x, K]``.

        y : vTensor
            Left-hand operand with shape ``[S, N_y, K]``.

        ctx : Context
            Execution context providing ``ctx.max_num_pages`` for the logical
            ``S`` dimension and tracking auxiliary memory.

        Returns
        -------
        vTensor
            A ``vTensor`` view wrapping the allocated output buffer.

        Raises
        ------
        AssertionError
            If types are not ``vTensor``, ranks are not 3, or the inner
            dimensions :math:`K` do not match.
        """
        prefix = self._prefix()

        # Type checks
        assert isinstance(x, vTensor), f"{prefix}profile expects x to be vTensor, got {type(x)}"
        assert isinstance(y, vTensor), f"{prefix}profile expects y to be vTensor, got {type(y)}"

        # Rank/shape checks
        assert x.dim() == 3 and y.dim() == 3, (
            f"{prefix}expected 3D inputs; got x.ndim={x.dim()}, y.ndim={y.dim()}"
        )
        # K must match
        assert x.shape[2] == y.shape[2], (
            f"{prefix}last dimension mismatch: x.shape[2]={x.shape[2]} vs y.shape[2]={y.shape[2]}"
        )

        # Output is BATCHED iff both inputs are BATCHED; otherwise RAGGED.
        self.output_format = (
            FORMAT.BATCHED
            if (x._format == FORMAT.BATCHED and y._format == FORMAT.BATCHED)
            else FORMAT.RAGGED
        )

        # Output logical sizes: Ny x Nx
        Ny, Nx = y.shape[1], x.shape[1]

        # Pure-metadata vTensor — no torch.empty allocation needed.
        self.output_buffer = vTensor(
            shape=(0, Ny, Nx),
            dtype=ctx.vortex_dtype,
            device=x.device,
            _format=self.output_format,
            tensor_id=len(ctx.tensor_list),
        )

        # Track auxiliary memory and graph structure in the context
        ctx.tensor_list.append(self.output_buffer)  # Track the output buffer in the context
        ctx.output_tensor_to_op_list.append(len(ctx.op_list))  # Map the output tensor to this operation
        ctx.op_list.append(self)  # Track this operation in the context
        ctx.op_to_input_tensor_list.append([x.tensor_id, y.tensor_id])  # Map this op to its input tensors
        ctx.op_to_output_tensor_list.append([self.output_buffer.tensor_id])  # Map this op to its output tensor

        return self.output_buffer