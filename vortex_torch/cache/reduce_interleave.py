import torch
from ..abs import vOp, vTensor, FORMAT
from .context import Context
from ..utils import ReduceType, QuantizationType, Schedule
from typing import Tuple, Dict, Optional


class ReduceInterleave(vOp):
    r"""
    Generic *interleaved* reduction dispatcher over the last two logical
    axes.

    This dispatcher covers a family of reductions (mean/max/min/L2-norm/sum,
    etc.) on rank-3 tensors that fold consecutive groups of :attr:`k`
    elements along the reduced axis into a single output element. The
    input is treated as

    .. math::

        X \in \mathbb{R}^{B \times N \times D},

    where:

    - :math:`B` is a leading batch-like axis (typically derived from the
      runtime, e.g. ``max_new_tokens_per_batch * head_num``),
    - :math:`N` is a sequence or position dimension, and
    - :math:`D` is a feature/channel dimension.

    The reduction dimension is chosen by :attr:`dim`. The size of the
    reduced axis must be divisible by :attr:`k`, and consecutive groups of
    :attr:`k` adjacent elements collapse into one output element:

    - ``dim == 1`` (interleaved row-wise reduction over :math:`N`,
      requires ``N % k == 0``):

      .. math::

         \text{out} \in \mathbb{R}^{B \times (N/k) \times D}, \qquad
         \text{out}[b, m, d]
         = \mathop{\mathrm{reduce}}_{0 \le j < k}
           X[b, m \cdot k + j, d].

    - ``dim == 2`` (interleaved column-wise reduction over :math:`D`,
      requires ``D % k == 0``):

      .. math::

         \text{out} \in \mathbb{R}^{B \times N \times (D/k)}, \qquad
         \text{out}[b, n, e]
         = \mathop{\mathrm{reduce}}_{0 \le j < k}
           X[b, n, e \cdot k + j].

    Setting ``k == 1`` makes this op an identity (each group has a single
    element). Setting ``k`` equal to the full reduced-axis length recovers
    the plain :class:`Reduce` op.

    The exact reduction operation (mean, max, min, L2-norm, sum, etc.) is
    encoded in :attr:`reduce_type` and interpreted by the implementation.

    Dispatch is based on the pair of tensor formats
    ``(x_format, o_format)`` and a registry mapping:

    .. code-block:: text

        (x_format, o_format) -> (impl, resolved_output_format)

    Policy
    ------
    - If ``output`` is ``None``:

      - :meth:`profile` selects an implementation for
        ``(x_fmt, FORMAT.RAGGED)`` (i.e. with RAGGED output).
      - An internal buffer is allocated with logical shape
        ``[B, exp_N, exp_D]``, where:

        - for ``dim == 1``: ``exp_N = N / k``, ``exp_D = D``,
        - for ``dim == 2``: ``exp_N = N``, ``exp_D = D / k``.

    - If ``output`` is provided:

      - :meth:`profile` requires an exact implementation key for
        ``(x_fmt, o_fmt)``.
      - The shape of ``output`` must match the expected
        ``(exp_N, exp_D)`` given :attr:`dim` and :attr:`k`.
      - Device consistency is enforced between ``x`` and ``output``.

    Attributes
    ----------
    _impl_map : Dict[Tuple[FORMAT, FORMAT], FORMAT]
        Dispatch table keyed by ``(x_format, o_format)``. Each entry
        maps to the resolved output format.
    dim : int
        Reduction dimension in the logical 3D tensor. Must be either:

        - ``1`` for row-wise reduction over :math:`N`, or
        - ``2`` for column-wise reduction over :math:`D`.
    k : int
        Group size. Consecutive ``k`` elements along the reduced axis are
        folded into one output element. Must be a positive integer that
        divides the size of the reduced axis.
    reduce_type : Optional[ReduceType]
        Enum describing which reduction to perform.
    output_format : Optional[FORMAT]
        The output tensor format as determined in :meth:`profile`.
    output_buffer : Optional[vTensor]
        Pure-metadata vTensor descriptor for the output (graph node).
    """

    _impl_map: Dict[Tuple[FORMAT, FORMAT], FORMAT] = {
        (FORMAT.PAGED,  FORMAT.PAGED):  FORMAT.PAGED,
        (FORMAT.RAGGED, FORMAT.PAGED):  FORMAT.PAGED,
        (FORMAT.PAGED,  FORMAT.RAGGED): FORMAT.RAGGED,
        (FORMAT.RAGGED, FORMAT.RAGGED): FORMAT.RAGGED,
    }

    def __init__(self, dim: int = 1, k: int = 2):
        super().__init__()
        self.dim = dim
        self.k = k
        self.reduce_type: Optional[ReduceType] = None
        self.quantization_type: Optional[QuantizationType] = None
        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[vTensor] = None
        # Cache reductions fuse into the per-block kernel — see
        # ``cache.compiler.triton_impl.kernel_gen``.
        self.schedule = Schedule.W
        # Validate reduction dimension and group size at construction time.
        # Cache reduce always runs inside the per-block fused kernel, so
        # dim==0 (which would need to span blocks/tokens) is not allowed —
        # use an indexer ``Reduce(dim=0)`` if you need a cross-row summary.
        cls = self.__class__.__name__
        assert self.dim in (1, 2), (
            f"{cls}.__init__: dim must be 1 or 2 (cache reduce cannot operate on dim=0), "
            f"got dim={self.dim}"
        )
        assert isinstance(self.k, int) and self.k >= 1, (
            f"{cls}.__init__: k must be a positive int, got k={self.k!r}"
        )

    # ------------------------------ helpers ------------------------------ #
    def _resolve_quantization(self, x: vTensor) -> QuantizationType:
        r"""Map ``x.dtype`` to the matching :class:`QuantizationType`.

        Centralized so both branches of :meth:`profile` agree on FP8/BF16
        dispatch and we don't silently leave ``quantization_type`` as ``None``.
        """
        prefix = self._prefix()
        if x.dtype == torch.bfloat16:
            return QuantizationType.BF16
        if x.dtype == torch.float8_e5m2:
            return QuantizationType.FP8_E5M2
        if x.dtype == torch.float8_e4m3fn:
            return QuantizationType.FP8_E4M3
        raise ValueError(f"{prefix}unsupported dtype {x.dtype} for reduction")

    def _infer_output_format_ragged(self, x_fmt: FORMAT) -> FORMAT:
        """Pick the RAGGED-output dispatch entry for ``x_fmt``."""
        key = (x_fmt, FORMAT.RAGGED)
        assert key in self._impl_map, (
            f"{self._prefix()}no RAGGED-output implementation for x_fmt={x_fmt}. "
            f"Available keys: {list(self._impl_map.keys())}"
        )
        return self._impl_map[key]

    # --------------------------------------------------------------------- #
    # profile: validate, pick impl/format, and return the provided vTensor
    # --------------------------------------------------------------------- #
    def profile(
        self, x: vTensor, output: Optional[vTensor], loc: torch.Tensor, ctx: Context
    ) -> vTensor:
        r"""
        Validate inputs, resolve the interleaved reduction implementation
        and output format, and optionally allocate an internal output
        buffer.

        The input tensor ``x`` is expected to have logical shape
        ``[B, N, D]``. According to :attr:`dim` and :attr:`k`, the
        expected output logical shape is:

        - ``dim == 1``: ``[B, N // k, D]`` (requires ``N % k == 0``)
        - ``dim == 2``: ``[B, N, D // k]`` (requires ``D % k == 0``)

        The auxiliary tensor ``loc`` carries per-position metadata used
        by the implementation; its shape and semantics are
        implementation-defined.

        Parameters
        ----------
        x : vTensor
            Input tensor with logical shape ``[B, N, D]``.

        output : Optional[vTensor]
            Optional preallocated output tensor. If ``None``, an internal
            buffer with shape ``[B, exp_N, exp_D]`` is allocated using
            ``ctx.max_new_tokens_per_batch * ctx.head_num`` for ``B`` and a
            RAGGED-output implementation is selected. If not ``None``,
            this tensor must have rank 3 and shape compatible with
            :attr:`dim` and :attr:`k` as described above.

        loc : torch.Tensor
            Auxiliary tensor carrying metadata required by the reduction
            implementation.

        ctx : Context
            Execution context that provides the runtime value of ``B`` and
            is used for auxiliary memory accounting.

        Returns
        -------
        vTensor
            A :class:`vTensor` view representing the resolved output:
            either the provided ``output`` or an internally allocated
            buffer.

        Raises
        ------
        AssertionError
            If types, ranks, formats, shapes, or devices are incompatible,
            if the reduced axis is not divisible by :attr:`k`, or if no
            implementation is found in :attr:`_impl_map`.
        """
        prefix = self._prefix()

        # --- type & rank checks ---
        assert isinstance(x, vTensor), f"{prefix}x must be vTensor, got {type(x)}"
        assert isinstance(loc, torch.Tensor), f"{prefix}loc must be torch.Tensor, got {type(loc)}"
        assert x.dim() == 3, f"{prefix}x must be 3D, got ndim={x.dim()} shape={tuple(x.shape)}"

        x_fmt = x._format
        N, D = x.shape[1], x.shape[2]

        # Divisibility check on the reduced axis.
        if self.dim == 1:
            assert N % self.k == 0, (
                f"{prefix}profile(dim=1): expected x.shape[1] (N={N}) divisible by k={self.k}"
            )
            exp_N, exp_D = N // self.k, D
        else:  # self.dim == 2
            assert D % self.k == 0, (
                f"{prefix}profile(dim=2): expected x.shape[2] (D={D}) divisible by k={self.k}"
            )
            exp_N, exp_D = N, D // self.k

        # Case A: output not provided -> pick RAGGED output and build a
        # pure-metadata vTensor in ``ctx.vortex_dtype`` (the intermediate
        # dtype used by the cache pipeline, default bf16).
        if output is None:
            self.output_format = self._infer_output_format_ragged(x_fmt)
            self.quantization_type = self._resolve_quantization(x)

            B = ctx.max_new_tokens_per_batch * ctx.head_num
            self.output_buffer = vTensor(
                shape=(B, exp_N, exp_D),
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

        # Case B: output provided -> validate and pick exact impl by (x_fmt, o_fmt)
        assert isinstance(output, vTensor), f"{prefix}output must be vTensor, got {type(output)}"
        assert output.dim() == 3, (
            f"{prefix}output must be 3D, got ndim={output.dim()} shape={tuple(output.shape)}"
        )

        o_fmt = output._format
        key = (x_fmt, o_fmt)
        assert key in self._impl_map, (
            f"{prefix}no implementation for (x_fmt={x_fmt}, o_fmt={o_fmt}). "
            f"Available keys: {list(self._impl_map.keys())}"
        )
        self.output_format = self._impl_map[key]

        # Shape checks per reduction dim and group size.
        assert output.shape[1] == exp_N, (
            f"{prefix}profile(dim={self.dim}, k={self.k}): expected output.shape[1] == {exp_N}, "
            f"got {tuple(output.shape)}"
        )
        assert output.shape[2] == exp_D, (
            f"{prefix}profile(dim={self.dim}, k={self.k}): expected output.shape[2] == {exp_D}, "
            f"got {tuple(output.shape)}"
        )

        # Device consistency
        assert x.device == output.device, (
            f"{prefix}x and output must be on the same device "
            f"(x.device={x.device}, output.device={output.device})"
        )

        self.quantization_type = self._resolve_quantization(x)

        # Register in the cache graph. Caller-provided ``output`` must already
        # have a valid ``tensor_id`` in ``ctx.tensor_list``; we claim it as
        # this op's produced tensor (override producer slot).
        ctx.output_tensor_to_op_list[output.tensor_id] = len(ctx.op_list)
        ctx.op_list.append(self)
        ctx.op_to_input_tensor_list.append([x.tensor_id])
        ctx.op_to_output_tensor_list.append([output.tensor_id])

        return output


class MeanInterleave(ReduceInterleave):
    r"""
    Interleaved mean reduction over a single logical axis.

    This operator reduces a rank-3 tensor

    .. math::

        X \in \mathbb{R}^{B \times N \times D}

    by taking the arithmetic mean over consecutive groups of :attr:`k`
    elements along one of the inner dimensions, as configured by
    :attr:`dim`:

    - ``dim == 1`` (requires ``N % k == 0``): grouped row-wise mean over
      :math:`N`, producing

      .. math::

          Y \in \mathbb{R}^{B \times (N/k) \times D}, \qquad
          Y[b, m, d]
          = \frac{1}{k} \sum_{j=0}^{k-1} X[b, m \cdot k + j, d].

    - ``dim == 2`` (requires ``D % k == 0``): grouped column-wise mean
      over :math:`D`, producing

      .. math::

          Y \in \mathbb{R}^{B \times N \times (D/k)}, \qquad
          Y[b, n, e]
          = \frac{1}{k} \sum_{j=0}^{k-1} X[b, n, e \cdot k + j].

    Parameters
    ----------
    dim : int, optional
        Logical reduction dimension. Must be ``1`` (reduce over
        :math:`N`) or ``2`` (reduce over :math:`D`). Default is ``1``.
    k : int, optional
        Group size. Default is ``2``.
    """
    def __init__(self, dim: int = 1, k: int = 2):
        super().__init__(dim, k)
        self.reduce_type = ReduceType.Mean


class MaxInterleave(ReduceInterleave):
    r"""
    Interleaved max reduction over a single logical axis.

    This operator reduces a rank-3 tensor

    .. math::

        X \in \mathbb{R}^{B \times N \times D}

    by taking the maximum over consecutive groups of :attr:`k` elements
    along one of the inner dimensions, as configured by :attr:`dim`:

    - ``dim == 1`` (requires ``N % k == 0``): grouped row-wise max over
      :math:`N`, producing

      .. math::

          Y \in \mathbb{R}^{B \times (N/k) \times D}, \qquad
          Y[b, m, d]
          = \max_{0 \le j < k} X[b, m \cdot k + j, d].

    - ``dim == 2`` (requires ``D % k == 0``): grouped column-wise max
      over :math:`D`, producing

      .. math::

          Y \in \mathbb{R}^{B \times N \times (D/k)}, \qquad
          Y[b, n, e]
          = \max_{0 \le j < k} X[b, n, e \cdot k + j].

    Parameters
    ----------
    dim : int, optional
        Logical reduction dimension. Must be ``1`` (reduce over
        :math:`N`) or ``2`` (reduce over :math:`D`). Default is ``1``.
    k : int, optional
        Group size. Default is ``2``.
    """
    def __init__(self, dim: int = 1, k: int = 2):
        super().__init__(dim, k)
        self.reduce_type = ReduceType.Max


class MinInterleave(ReduceInterleave):
    r"""
    Interleaved min reduction over a single logical axis.

    This operator reduces a rank-3 tensor

    .. math::

        X \in \mathbb{R}^{B \times N \times D}

    by taking the minimum over consecutive groups of :attr:`k` elements
    along one of the inner dimensions, as configured by :attr:`dim`:

    - ``dim == 1`` (requires ``N % k == 0``): grouped row-wise min over
      :math:`N`, producing

      .. math::

          Y \in \mathbb{R}^{B \times (N/k) \times D}, \qquad
          Y[b, m, d]
          = \min_{0 \le j < k} X[b, m \cdot k + j, d].

    - ``dim == 2`` (requires ``D % k == 0``): grouped column-wise min
      over :math:`D`, producing

      .. math::

          Y \in \mathbb{R}^{B \times N \times (D/k)}, \qquad
          Y[b, n, e]
          = \min_{0 \le j < k} X[b, n, e \cdot k + j].

    Parameters
    ----------
    dim : int, optional
        Logical reduction dimension. Must be ``1`` (reduce over
        :math:`N`) or ``2`` (reduce over :math:`D`). Default is ``1``.
    k : int, optional
        Group size. Default is ``2``.
    """
    def __init__(self, dim: int = 1, k: int = 2):
        super().__init__(dim, k)
        self.reduce_type = ReduceType.Min


class L2NormInterleave(ReduceInterleave):
    r"""
    Interleaved L2-norm reduction (not RMS) over a single logical axis.

    This operator reduces a rank-3 tensor

    .. math::

        X \in \mathbb{R}^{B \times N \times D}

    by computing an L2 norm over consecutive groups of :attr:`k` elements
    along one of the inner dimensions, as configured by :attr:`dim`. The
    reduction is *not* normalized by the number of elements (it is an L2
    norm, not an RMS):

    - ``dim == 1`` (requires ``N % k == 0``): grouped row-wise L2 norm
      over :math:`N`, producing

      .. math::

          Y \in \mathbb{R}^{B \times (N/k) \times D}, \qquad
          Y[b, m, d]
          = \sqrt{\sum_{j=0}^{k-1} X[b, m \cdot k + j, d]^2}.

    - ``dim == 2`` (requires ``D % k == 0``): grouped column-wise L2 norm
      over :math:`D`, producing

      .. math::

          Y \in \mathbb{R}^{B \times N \times (D/k)}, \qquad
          Y[b, n, e]
          = \sqrt{\sum_{j=0}^{k-1} X[b, n, e \cdot k + j]^2}.

    Notes
    -----
    This is a pure L2 norm over each group, with no division by the
    group size. It should not be confused with an RMS over the group.

    Parameters
    ----------
    dim : int, optional
        Logical reduction dimension. Must be ``1`` (reduce over
        :math:`N`) or ``2`` (reduce over :math:`D`). Default is ``1``.
    k : int, optional
        Group size. Default is ``2``.
    """
    def __init__(self, dim: int = 1, k: int = 2):
        super().__init__(dim, k)
        self.reduce_type = ReduceType.L2Norm
