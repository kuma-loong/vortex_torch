import torch
from typing import Dict, Optional
from .context import Context
from ..abs import vTensor, FORMAT, vOp
from ..utils import Schedule

class Softmax(vOp):
    r"""
    In-place softmax dispatcher over a packed leading axis.

    The input is treated as a rank-3 tensor

    .. math::

        X \in \mathbb{R}^{S_{\text{pack}} \times D_0 \times D_1},

    where the leading dimension :math:`S_{\text{pack}}` is a packed
    concatenation of :math:`B` segments:

    .. math::

        S_{\text{pack}}
        = \sum_{b=0}^{B-1} S_b, \qquad
        X =
        \begin{bmatrix}
            X_0 \\
            X_1 \\
            \vdots \\
            X_{B-1}
        \end{bmatrix},

    with

    .. math::

        X_b \in \mathbb{R}^{S_b \times D_0 \times D_1}.
    
    For each segment :math:`b` and each fixed pair
    :math:`(d_0, d_1)`, this operator applies a scaled softmax along the
    packed axis within that segment:

    .. math::

        \text{out}[s, d_0, d_1]
        =
        \frac{\exp\bigl(\text{scale} \cdot X[s, d_0, d_1]\bigr)}
             {\sum_{s' \in \mathcal{I}_b}
                \exp\bigl(\text{scale} \cdot X[s', d_0, d_1]\bigr)},
        \quad s \in \mathcal{I}_b,

    where :math:`\mathcal{I}_b` denotes the index range in
    :math:`[0, S_{\text{pack}})` corresponding to the :math:`b`-th
    segment of length :math:`S_b`.

    In other words, ``dim == 0`` is a **packed S axis**, and softmax is
    applied independently within each segment of that axis.

    Key properties
    --------------
    - Only ``dim == 0`` is supported.
    - Dispatch is keyed by the input tensor format ``x._format``.
    - No output buffer is allocated; the operation is performed in-place
      on ``x`` and the same tensor is returned.

    Attributes
    ----------
    _impl_map : Dict[FORMAT, FORMAT]
        Dispatch table keyed by ``x_format``. Each entry maps to the
        resolved output format.
    dim : int
        Softmax axis. Must be ``0`` and corresponds to the packed
        :math:`S_{\text{pack}}` dimension.
    scale : float
        Multiplicative factor applied to ``x`` before the softmax
        (i.e. computes softmax of ``x * scale``).
    output_format : Optional[FORMAT]
        The output tensor format as determined in :meth:`profile`.
    """

    # Dispatch table keyed by x_format -> resolved output format.
    _impl_map: Dict[FORMAT, FORMAT] = {
        FORMAT.RAGGED: FORMAT.RAGGED,
        # Extend with other formats if you add more kernels:
        # FORMAT.PAGED: FORMAT.PAGED,
    }

    def __init__(self, dim: int = 0, scale: float = 1.0):
        super().__init__()
        self.dim = dim
        self.scale = scale
        self.output_format: Optional[FORMAT] = None
        self.schedule = Schedule.S
        # Validate dim at construction
        prefix = self._prefix()
        assert self.dim in (0,), f"{prefix}__init__: dim must be 0, got dim={self.dim}"

    # ---------------- profile ----------------
    def profile(self, x: vTensor, ctx: Context) -> vTensor:
        r"""
        Validate the input and select an implementation.

        Since the operation is in-place, no output buffer is allocated and
        this method simply returns the input :class:`vTensor` unmodified.

        The input is expected to have logical shape
        ``[S_pack, D_0, D_1]``, where ``S_pack`` is understood as a packed
        concatenation of :math:`B` segments of lengths :math:`S_b`. The
        softmax is applied along ``dim == 0`` **within** each such segment.

        Parameters
        ----------
        x : vTensor
            Input tensor to be modified in-place, with logical shape
            ``[S_pack, D_0, D_1]``.

        ctx : Context
            Execution context (included for API symmetry; not used for
            buffer allocation here).

        Returns
        -------
        vTensor
            The same object as ``x``, returned as the output view.

        Raises
        ------
        AssertionError
            If ``x`` is not a :class:`vTensor`, if its rank is not 3, or if
            no implementation is registered for ``x._format``.
        """
        prefix = self._prefix()

        # Type & rank checks
        assert isinstance(x, vTensor), f"{prefix}profile expects x to be vTensor, got {type(x)}"
        assert x.dim() == 3, (
            f"{prefix}expected 3D input [S_pack, D0, D1], "
            f"got ndim={x.dim()} shape={tuple(x.shape)}"
        )

        # Dispatch by input format
        x_fmt = x._format
        assert x_fmt in self._impl_map, (
            f"{prefix}no implementation for x_fmt={x_fmt}. "
            f"Available keys: {list(self._impl_map.keys())}"
        )
        self.output_format = self._impl_map[x_fmt]

        self.output_buffer = vTensor(
            shape=(0, x.shape[1], x.shape[2]),
            dtype=ctx.vortex_dtype,
            device=x.device,
            _format=self.output_format,
            tensor_id=len(ctx.tensor_list),
        )
        # Track auxiliary memory and graph structure in the context
        ctx.tensor_list.append(self.output_buffer)  # Track the output buffer in the context
        ctx.output_tensor_to_op_list.append(len(ctx.op_list))  # Map the output tensor to this operation
        ctx.op_list.append(self)  # Track this operation in the context
        ctx.op_to_input_tensor_list.append([x.tensor_id])  # Map this op to its input tensors
        ctx.op_to_output_tensor_list.append([self.output_buffer.tensor_id])  # Map this op to its output tensor

        return self.output_buffer


class Normalize(vOp):
    r"""
    In-place normalization dispatcher over a packed leading axis.

    The input is treated as a rank-3 tensor

    .. math::

        X \in \mathbb{R}^{S_{\text{pack}} \times D_0 \times D_1},

    where the leading dimension :math:`S_{\text{pack}}` is a packed
    concatenation of :math:`B` segments:

    .. math::

        S_{\text{pack}}
        = \sum_{b=0}^{B-1} S_b, \qquad
        X =
        \begin{bmatrix}
            X_0 \\
            X_1 \\
            \vdots \\
            X_{B-1}
        \end{bmatrix},

    with

    .. math::

        X_b \in \mathbb{R}^{S_b \times D_0 \times D_1}.
    
    For each segment :math:`b` and each fixed pair :math:`(d_0, d_1)`,
    this operator normalizes the values along the packed axis within that
    segment. A common interpretation is L2 normalization:

    .. math::

        \hat{X}[s, d_0, d_1]
        =
        \frac{X[s, d_0, d_1]}
             {\sqrt{\sum_{s' \in \mathcal{I}_b}
                     X[s', d_0, d_1]^2}},
        \quad s \in \mathcal{I}_b,

    where :math:`\mathcal{I}_b` denotes the index range in
    :math:`[0, S_{\text{pack}})` corresponding to the :math:`b`-th
    segment of length :math:`S_b`.

    In other words, ``dim == 0`` is a **packed S axis**, and normalization
    is applied independently within each segment of that axis.

    Key properties
    --------------
    - Only ``dim == 0`` is supported.
    - Dispatch is keyed by the input tensor format ``x._format``.
    - No output buffer is allocated; the operation is performed in-place
      on ``x`` and the same tensor is returned.

    Attributes
    ----------
    _impl_map : Dict[FORMAT, FORMAT]
        Dispatch table keyed by ``x_format``. Each entry maps to the
        resolved output format.
    dim : int
        Normalization axis. Must be ``0`` and corresponds to the packed
        :math:`S_{\text{pack}}` dimension.
    output_format : Optional[FORMAT]
        The output tensor format as determined in :meth:`profile`.
    """

    # Dispatch table keyed by x_format -> resolved output format.
    _impl_map: Dict[FORMAT, FORMAT] = {
        FORMAT.RAGGED: FORMAT.RAGGED,
        # Extend with other formats if you add more kernels:
        # FORMAT.PAGED: FORMAT.PAGED,
    }

    def __init__(self, dim: int = 0):
        super().__init__()
        self.dim = dim
        self.output_format: Optional[FORMAT] = None
        self.schedule = Schedule.S

        # Validate dim at construction
        prefix = self._prefix()
        assert self.dim in (0,), f"{prefix}__init__: dim must be 0, got dim={self.dim}"

    # ---------------- profile ----------------
    def profile(self, x: vTensor, ctx: Context) -> vTensor:
        r"""
        Validate the input and select an implementation.

        Since the operation is in-place, no output buffer is allocated and
        this method simply returns the input :class:`vTensor` unmodified.

        The input is expected to have logical shape
        ``[S_pack, D_0, D_1]``, where ``S_pack`` is understood as a packed
        concatenation of :math:`B` segments of lengths :math:`S_b`. The
        normalization is applied along ``dim == 0`` **within** each such
        segment.

        Parameters
        ----------
        x : vTensor
            Input tensor to be normalized in-place, with logical shape
            ``[S_pack, D_0, D_1]``.

        ctx : Context
            Execution context (included for API symmetry; not used for
            buffer allocation here).

        Returns
        -------
        vTensor
            The same object as ``x``, returned as the output view.

        Raises
        ------
        AssertionError
            If ``x`` is not a :class:`vTensor`, if its rank is not 3, or if
            no implementation is registered for ``x._format``.
        """
        prefix = self._prefix()

        # Type & rank checks
        assert isinstance(x, vTensor), f"{prefix}profile expects x to be vTensor, got {type(x)}"
        assert x.dim() == 3, (
            f"{prefix}expected 3D input [S_pack, D0, D1], "
            f"got ndim={x.dim()} shape={tuple(x.shape)}"
        )

        # Dispatch by input format
        x_fmt = x._format
        assert x_fmt in self._impl_map, (
            f"{prefix}no implementation for x_fmt={x_fmt}. "
            f"Available keys: {list(self._impl_map.keys())}"
        )
        self.output_format = self._impl_map[x_fmt]

        self.output_buffer = vTensor(
            shape=(0, x.shape[1], x.shape[2]),
            dtype=ctx.vortex_dtype,
            device=x.device,
            _format=self.output_format,
            tensor_id=len(ctx.tensor_list),
        )
        # Track auxiliary memory and graph structure in the context
        ctx.tensor_list.append(self.output_buffer)  # Track the output buffer in the context
        ctx.output_tensor_to_op_list.append(len(ctx.op_list))  # Map the output tensor to this operation
        ctx.op_list.append(self)  # Track this operation in the context
        ctx.op_to_input_tensor_list.append([x.tensor_id])  # Map this op to its input tensors
        ctx.op_to_output_tensor_list.append([self.output_buffer.tensor_id])  # Map this op to its output tensor

        return self.output_buffer


class Conv1d(vOp):
    r"""
    Causal 1D convolution dispatcher over a packed leading axis.

    The input is treated as a rank-3 tensor

    .. math::

        X \in \mathbb{R}^{S_{\text{pack}} \times D_0 \times D_1},

    where the leading dimension :math:`S_{\text{pack}}` is a packed
    concatenation of :math:`B` segments of lengths :math:`S_b`. For each
    segment :math:`b` and each fixed :math:`(d_0, d_1)`, this operator
    applies a depth-wise causal 1D convolution of kernel size :math:`K`
    along the packed axis **within** the segment, restricted to the range
    that excludes the reserved BOS and EOS pages:

    .. math::

        \mathcal{I}_b^{\text{mid}}
        = [\,b_{\text{bos}},\; S_b - b_{\text{eos}}\,),

    where :math:`b_{\text{bos}}` and :math:`b_{\text{eos}}` are
    ``ctx.block_reserved_bos`` and ``ctx.block_reserved_eos``.

    For :math:`s \in \mathcal{I}_b^{\text{mid}}` (local index within the
    mid-range),

    .. math::

        Y[s, d_0, d_1]
        = \sum_{k=0}^{K-1} W[k, d_0, d_1] \cdot X[s - k, d_0, d_1],

    where positions :math:`s - k < 0` (within the mid-range) are treated
    as zero. The BOS/EOS reserved regions are neither read nor written.

    Key properties
    --------------
    - Only ``dim == 0`` is supported.
    - Dispatch is keyed by the input tensor format ``x._format``.
    - The convolution is depth-wise: each :math:`(d_0, d_1)` channel has
      its own :math:`K`-tap filter.
    - Out-of-place: a fresh output buffer is allocated.

    Attributes
    ----------
    _impl_map : Dict[FORMAT, FORMAT]
        Dispatch table keyed by ``x_format``. Each entry maps to the
        resolved output format.
    dim : int
        Convolution axis. Must be ``0`` and corresponds to the packed
        :math:`S_{\text{pack}}` dimension.
    weight : torch.Tensor
        Convolution weight with shape ``[K, D_0, D_1]``, materialized
        from the nested Python list passed at construction.
    kernel_size : int
        Kernel length :math:`K`, derived from ``weight.shape[0]``.
    """

    _impl_map: Dict[FORMAT, FORMAT] = {
        FORMAT.RAGGED: FORMAT.RAGGED,
    }

    def __init__(
        self,
        weight: list,
        dim: int = 0,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        assert isinstance(weight, list), (
            f"Conv1d: weight must be a Python list, got {type(weight)}"
        )
        weight_tensor = torch.tensor(weight, dtype=dtype, device=device)
        assert weight_tensor.dim() == 3, (
            f"Conv1d: weight must be a 3D nested list [K, D0, D1], "
            f"got shape {tuple(weight_tensor.shape)}"
        )
        self.dim = dim
        self.weight = weight_tensor
        self.kernel_size = weight_tensor.shape[0]
        self.output_format: Optional[FORMAT] = None
        self.schedule = Schedule.S

        prefix = self._prefix()
        assert self.dim in (0,), f"{prefix}__init__: dim must be 0, got dim={self.dim}"

    # ---------------- profile ----------------
    def profile(self, x: vTensor, ctx: Context) -> vTensor:
        r"""
        Validate the input, select an implementation, allocate the output
        buffer, and return a :class:`vTensor` view with the resolved format.

        The input is expected to have logical shape ``[S_pack, D_0, D_1]``,
        and the weight stored on ``self`` must have shape
        ``[K, D_0, D_1]`` with matching inner dimensions.

        Parameters
        ----------
        x : vTensor
            Input tensor with logical shape ``[S_pack, D_0, D_1]``.

        ctx : Context
            Execution context providing ``ctx.max_num_pages`` for the
            output buffer leading dimension.

        Returns
        -------
        vTensor
            A ``vTensor`` view wrapping the internally allocated output
            buffer with the resolved output format.
        """
        prefix = self._prefix()

        assert isinstance(x, vTensor), f"{prefix}profile expects x to be vTensor, got {type(x)}"
        assert x.dim() == 3, (
            f"{prefix}expected 3D input [S_pack, D0, D1], "
            f"got ndim={x.dim()} shape={tuple(x.shape)}"
        )
        assert self.weight.shape[1] == x.shape[1] and self.weight.shape[2] == x.shape[2], (
            f"{prefix}weight inner dims {tuple(self.weight.shape[1:])} "
            f"must match input inner dims {tuple(x.shape[1:])}"
        )

        x_fmt = x._format
        assert x_fmt in self._impl_map, (
            f"{prefix}no implementation for x_fmt={x_fmt}. "
            f"Available keys: {list(self._impl_map.keys())}"
        )
        self.output_format = self._impl_map[x_fmt]

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
