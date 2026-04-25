import torch
from typing import Dict, Optional
from .context import Context
from ..abs import vTensor, FORMAT, vOp
from ..utils import ReduceType, Schedule

class Reduce(vOp):
    r"""
    Generic reduction dispatcher for rank-3 logical tensors ``[N, D_0, D_1]``.

    This operator performs a 1D reduction over either the ``D_0`` or ``D_1``
    axis of a 3D tensor. The leading dimension ``N`` is generic and may
    represent a batch axis (``B``) or a sequence/page axis (``S``); the
    reduction is applied independently for each of the ``N`` slices.

    Given an input tensor

    .. math::

        X \in \mathbb{R}^{N \times D_0 \times D_1},

    the output logical shape depends on the configured reduction dimension
    ``dim``:

    - ``dim == 1`` (reduce over :math:`D_0`):

      .. math::

         \text{out} \in \mathbb{R}^{N \times 1 \times D_1}.

    - ``dim == 2`` (reduce over :math:`D_1`):

      .. math::

         \text{out} \in \mathbb{R}^{N \times D_0 \times 1}.

    The specific reduction operation (e.g. mean, max, min, L2-norm, sum)
    is selected via :attr:`reduce_type`.

    Dispatch is keyed only by the input format ``x._format``.

    Attributes
    ----------
    _impl_map : Dict[FORMAT, FORMAT]
        Dispatch table keyed by ``x_format``. Each entry maps to the
        resolved output format.

    dim : int
        Reduction dimension in the logical 3D tensor: must be either

        - ``1`` for reduction over the :math:`D_0` axis, or
        - ``2`` for reduction over the :math:`D_1` axis.

    reduce_type : Optional[ReduceType]
        The type of reduction to perform (e.g. mean, max, min, L2-norm, sum).

    output_format : Optional[FORMAT]
        The output tensor format as determined in :meth:`profile`.

    output_buffer : Optional[torch.Tensor]
        Preallocated output tensor buffer with logical shape
        ``[N, out_D0, out_D1]``, where ``out_D0`` and ``out_D1`` depend on
        ``dim`` as described above.
    """

    # Dispatch table for dim in {1, 2} (fused, Schedule.W). dim==0 is a
    # cross-row reduction (Schedule.S, RAGGED → BATCHED) handled below.
    _impl_map: Dict[FORMAT, FORMAT] = {
        FORMAT.RAGGED: FORMAT.RAGGED,
        FORMAT.BATCHED: FORMAT.BATCHED,
        # Add more entries if you support other formats:
        # FORMAT.PAGED: FORMAT.PAGED,
    }

    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim
        self.reduce_type: Optional[ReduceType] = None
        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[torch.Tensor] = None
        # dim==0 reduces across the packed leading axis; the result is one
        # summary per (batch, kv_head), so it can't fuse into a per-block
        # workload kernel — schedule it standalone.
        self.schedule = Schedule.S if dim == 0 else Schedule.W
        prefix = self._prefix()
        assert self.dim in (0, 1, 2), (
            f"{prefix}__init__: dim must be 0, 1, or 2, got dim={self.dim}"
        )

    # ---------------- profile ----------------
    def profile(self, x: vTensor, ctx: Context) -> vTensor:
        r"""
        Validate the input, select an implementation based on ``x._format``,
        allocate the output buffer, and return a ``vTensor`` view.

        The input tensor is expected to have logical shape ``[N, D_0, D_1]``,
        where the leading dimension ``N`` may represent either a batch size or
        a sequence/page count. The runtime uses ``ctx.max_num_pages`` to define
        the leading dimension of the output, in line with other operators that
        treat the first axis as the logical ``N`` axis.

        According to :attr:`dim`, the output logical shape is:

        - ``dim == 1`` → ``[N, 1, D_1]``
        - ``dim == 2`` → ``[N, D_0, 1]``

        Parameters
        ----------
        x : vTensor
            Input tensor with logical shape ``[N, D_0, D_1]``.

        ctx : Context
            Execution context providing ``ctx.max_num_pages`` for the leading
            dimension and tracking auxiliary memory usage.

        Returns
        -------
        vTensor
            A ``vTensor`` view wrapping the allocated output buffer with the
            resolved output format.

        Raises
        ------
        AssertionError
            If ``x`` is not a :class:`vTensor`, if its rank is not 3, or if no
            implementation is registered for ``x._format``.
        """
        prefix = self._prefix()

        # Type & rank checks
        assert isinstance(x, vTensor), f"{prefix}profile expects x to be vTensor, got {type(x)}"
        assert x.dim() == 3, (
            f"{prefix}expected 3D input [N, D0, D1], "
            f"got ndim={x.dim()} shape={tuple(x.shape)}"
        )

        x_fmt = x._format
        D0, D1 = x.shape[1], x.shape[2]

        if self.dim == 0:
            # Cross-row reduction: collapse the packed leading axis into one
            # summary per (batch, kv_head). Input must be RAGGED (per-page or
            # per-token); the compiler allocates a BATCHED buffer with leading
            # dim ``ctx.max_bs * ctx.num_kv_heads`` (see indexer interface).
            assert x_fmt == FORMAT.RAGGED, (
                f"{prefix}dim=0 reduce requires RAGGED input, got {x_fmt}"
            )
            self.output_format = FORMAT.BATCHED
            out_D0, out_D1 = D0, D1
        else:
            assert x_fmt in self._impl_map, (
                f"{prefix}no implementation for x_fmt={x_fmt}. "
                f"Available keys: {list(self._impl_map.keys())}"
            )
            self.output_format = self._impl_map[x_fmt]
            out_D0 = 1 if self.dim == 1 else D0
            out_D1 = 1 if self.dim == 2 else D1

        # Pure-metadata vTensor — no torch.empty allocation needed.
        self.output_buffer = vTensor(
            shape=(0, out_D0, out_D1),
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
        
        # Return vTensor view carrying the dispatched output format
        return self.output_buffer


class Max(Reduce):
    r"""
    Maximum reduction over a single logical axis.

    Given an input tensor

    .. math::

        X \in \mathbb{R}^{N \times D_0 \times D_1},

    this operator computes, depending on ``dim``:

    - ``dim == 1`` (reduce over :math:`D_0`):

      .. math::

         \text{out}[n, 0, d_1]
         = \max_{0 \le d_0 < D_0} X[n, d_0, d_1],

      with shape :math:`[N, 1, D_1]`.

    - ``dim == 2`` (reduce over :math:`D_1`):

      .. math::

         \text{out}[n, d_0, 0]
         = \max_{0 \le d_1 < D_1} X[n, d_0, d_1],

      with shape :math:`[N, D_0, 1]`.

    The leading dimension :math:`N` may represent either a batch axis
    (``B``) or a sequence/page axis (``S``); the reduction is applied
    independently for each slice along this dimension.

    Parameters
    ----------
    dim : int, optional
        Reduction dimension in the logical 3D tensor (``1`` for :math:`D_0`,
        ``2`` for :math:`D_1`). Default is ``1``.
    """
    def __init__(self, dim: int = 1):
        super().__init__(dim)
        self.reduce_type = ReduceType.Max


class Min(Reduce):
    r"""
    Minimum reduction over a single logical axis.

    Given an input tensor

    .. math::

        X \in \mathbb{R}^{N \times D_0 \times D_1},

    this operator computes, depending on ``dim``:

    - ``dim == 1`` (reduce over :math:`D_0`):

      .. math::

         \text{out}[n, 0, d_1]
         = \min_{0 \le d_0 < D_0} X[n, d_0, d_1],

      with shape :math:`[N, 1, D_1]`.

    - ``dim == 2`` (reduce over :math:`D_1`):

      .. math::

         \text{out}[n, d_0, 0]
         = \min_{0 \le d_1 < D_1} X[n, d_0, d_1],

      with shape :math:`[N, D_0, 1]`.

    The leading dimension :math:`N` may represent either a batch axis
    (``B``) or a sequence/page axis (``S``); the reduction is applied
    independently for each slice along this dimension.

    Parameters
    ----------
    dim : int, optional
        Reduction dimension in the logical 3D tensor (``1`` for :math:`D_0`,
        ``2`` for :math:`D_1`). Default is ``1``.
    """
    def __init__(self, dim: int = 1):
        super().__init__(dim)
        self.reduce_type = ReduceType.Min


class Mean(Reduce):
    r"""
    Mean reduction over a single logical axis.

    Given an input tensor

    .. math::

        X \in \mathbb{R}^{N \times D_0 \times D_1},

    this operator computes, depending on ``dim``:

    - ``dim == 1`` (reduce over :math:`D_0`):

      .. math::

         \text{out}[n, 0, d_1]
         = \frac{1}{D_0} \sum_{d_0=0}^{D_0-1} X[n, d_0, d_1],

      with shape :math:`[N, 1, D_1]`.

    - ``dim == 2`` (reduce over :math:`D_1`):

      .. math::

         \text{out}[n, d_0, 0]
         = \frac{1}{D_1} \sum_{d_1=0}^{D_1-1} X[n, d_0, d_1],

      with shape :math:`[N, D_0, 1]`.

    The leading dimension :math:`N` may represent either a batch axis
    (``B``) or a sequence/page axis (``S``); the reduction is applied
    independently for each slice along this dimension.

    Parameters
    ----------
    dim : int, optional
        Reduction dimension in the logical 3D tensor (``1`` for :math:`D_0`,
        ``2`` for :math:`D_1`). Default is ``1``.
    """
    def __init__(self, dim: int = 1):
        super().__init__(dim)
        self.reduce_type = ReduceType.Mean


class L2Norm(Reduce):
    r"""
    L2-norm reduction over a single logical axis.

    Given an input tensor

    .. math::

        X \in \mathbb{R}^{N \times D_0 \times D_1},

    this operator computes, depending on ``dim``:

    - ``dim == 1`` (reduce over :math:`D_0`):

      .. math::

         \text{out}[n, 0, d_1]
         = \sqrt{\sum_{d_0=0}^{D_0-1} X[n, d_0, d_1]^2},

      with shape :math:`[N, 1, D_1]`.

    - ``dim == 2`` (reduce over :math:`D_1`):

      .. math::

         \text{out}[n, d_0, 0]
         = \sqrt{\sum_{d_1=0}^{D_1-1} X[n, d_0, d_1]^2},

      with shape :math:`[N, D_0, 1]`.

    The leading dimension :math:`N` may represent either a batch axis
    (``B``) or a sequence/page axis (``S``); the reduction is applied
    independently for each slice along this dimension.

    Parameters
    ----------
    dim : int, optional
        Reduction dimension in the logical 3D tensor (``1`` for :math:`D_0`,
        ``2`` for :math:`D_1`). Default is ``1``.
    """
    def __init__(self, dim: int = 1):
        super().__init__(dim)
        self.reduce_type = ReduceType.L2Norm


class Sum(Reduce):
    r"""
    Sum reduction over a single logical axis.

    Given an input tensor

    .. math::

        X \in \mathbb{R}^{N \times D_0 \times D_1},

    this operator computes, depending on ``dim``:

    - ``dim == 1`` (reduce over :math:`D_0`):

      .. math::

         \text{out}[n, 0, d_1]
         = \sum_{d_0=0}^{D_0-1} X[n, d_0, d_1],

      with shape :math:`[N, 1, D_1]`.

    - ``dim == 2`` (reduce over :math:`D_1`):

      .. math::

         \text{out}[n, d_0, 0]
         = \sum_{d_1=0}^{D_1-1} X[n, d_0, d_1],

      with shape :math:`[N, D_0, 1]`.

    The leading dimension :math:`N` may represent either a batch axis
    (``B``) or a sequence/page axis (``S``); the reduction is applied
    independently for each slice along this dimension.

    Parameters
    ----------
    dim : int, optional
        Reduction dimension in the logical 3D tensor (``1`` for :math:`D_0`,
        ``2`` for :math:`D_1`). Default is ``1``.
    """
    def __init__(self, dim: int = 1):
        super().__init__(dim)
        self.reduce_type = ReduceType.Sum
