import torch
from ..abs import vOp, vTensor, FORMAT
from .context import Context
from ..utils import ReduceType, QuantizationType, Schedule
from typing import Tuple, Dict, Optional


class Reduce(vOp):
    r"""
    Generic reduction dispatcher over the last two logical axes.

    This dispatcher covers a family of reductions (mean/max/min/L2-norm/sum,
    etc.) on rank-3 tensors. The input is treated as

    .. math::

        X \in \mathbb{R}^{B \times N \times D},

    where:

    - :math:`B` is a leading batch-like axis (typically derived from the
      runtime, e.g. ``max_new_tokens_per_batch * head_num``),
    - :math:`N` is a sequence or position dimension, and
    - :math:`D` is a feature/channel dimension.

    The reduction dimension is chosen by :attr:`dim`:

    - ``dim == 1`` (row-wise reduction over :math:`N`):

      .. math::

         \text{out} \in \mathbb{R}^{B \times 1 \times D},

    - ``dim == 2`` (column-wise reduction over :math:`D`):

      .. math::

         \text{out} \in \mathbb{R}^{B \times N \times 1}.

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

        - for ``dim == 1``: ``exp_N = 1``, ``exp_D = D``,
        - for ``dim == 2``: ``exp_N = N``, ``exp_D = 1``.

    - If ``output`` is provided:

      - :meth:`profile` requires an exact implementation key for
        ``(x_fmt, o_fmt)``.
      - The shape of ``output`` must match the expected
        ``(exp_N, exp_D)`` given :attr:`dim`.
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

    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim
        self.reduce_type: Optional[ReduceType] = None
        self.quantization_type: Optional[QuantizationType] = None
        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[vTensor] = None
        # Cache reductions fuse into the per-block kernel — see
        # ``cache.compiler.triton_impl.kernel_gen``.
        self.schedule = Schedule.W
        # Validate reduction dimension at construction time. Cache reduce
        # always runs inside the per-block fused kernel, so dim==0 (which
        # would need to span blocks/tokens) is not allowed — use an indexer
        # ``Reduce(dim=0)`` if you need a cross-row summary.
        cls = self.__class__.__name__
        assert self.dim in (1, 2), (
            f"{cls}.__init__: dim must be 1 or 2 (cache reduce cannot operate on dim=0), "
            f"got dim={self.dim}"
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
        Validate inputs, resolve the reduction implementation and output
        format, and optionally allocate an internal output buffer.

        The input tensor ``x`` is expected to have logical shape
        ``[B, N, D]``. According to :attr:`dim`, the expected output
        logical shape is:

        - ``dim == 1``: ``[B, 1, D]``
        - ``dim == 2``: ``[B, N, 1]``

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
            :attr:`dim` as described above.

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
            or if no implementation is found in :attr:`_impl_map`.
        """
        prefix = self._prefix()

        # --- type & rank checks ---
        assert isinstance(x, vTensor), f"{prefix}x must be vTensor, got {type(x)}"
        assert isinstance(loc, torch.Tensor), f"{prefix}loc must be torch.Tensor, got {type(loc)}"
        assert x.dim() == 3, f"{prefix}x must be 3D, got ndim={x.dim()} shape={tuple(x.shape)}"

        x_fmt = x._format

        # Compute expected output (N, D) given reduction dim
        # dim==1 -> reduce rows: keep D, set N=1
        # dim==2 -> reduce cols: keep N, set D=1
        exp_N = 1 if self.dim == 1 else x.shape[1]
        exp_D = 1 if self.dim == 2 else x.shape[2]

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

        # Shape checks per reduction dim
        if self.dim == 1:
            # Expect (*, 1, x.D)
            assert output.shape[1] == 1, (
                f"{prefix}profile(dim=1): expected output.shape[1] == 1, "
                f"got {tuple(output.shape)}"
            )
            assert output.shape[2] == x.shape[2], (
                f"{prefix}profile(dim=1): expected output.shape[2] == x.shape[2], "
                f"got {output.shape[2]} vs {x.shape[2]}"
            )
        else:  # self.dim == 2
            # Expect (*, x.N, 1)
            assert output.shape[2] == 1, (
                f"{prefix}profile(dim=2): expected output.shape[2] == 1, "
                f"got {tuple(output.shape)}"
            )
            assert output.shape[1] == x.shape[1], (
                f"{prefix}profile(dim=2): expected output.shape[1] == x.shape[1], "
                f"got {output.shape[1]} vs {x.shape[1]}"
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


class Mean(Reduce):
    r"""
    Mean reduction over a single logical axis.

    This operator reduces a rank-3 tensor

    .. math::

        X \in \mathbb{R}^{B \times N \times D}

    by taking the arithmetic mean along one of the inner dimensions, as
    configured by :attr:`dim`:

    - ``dim == 1``: row-wise mean over :math:`N`, producing

      .. math::

          Y \in \mathbb{R}^{B \times 1 \times D}, \qquad
          Y[b, 0, d]
          = \frac{1}{N} \sum_{n=0}^{N-1} X[b, n, d].

    - ``dim == 2``: column-wise mean over :math:`D`, producing

      .. math::

          Y \in \mathbb{R}^{B \times N \times 1}, \qquad
          Y[b, n, 0]
          = \frac{1}{D} \sum_{d=0}^{D-1} X[b, n, d].

    Parameters
    ----------
    dim : int, optional
        Logical reduction dimension. Must be ``1`` (reduce over
        :math:`N`) or ``2`` (reduce over :math:`D`). Default is ``1``.
    """
    def __init__(self, dim: int = 1):
        super().__init__(dim)
        self.reduce_type = ReduceType.Mean
    


class Max(Reduce):
    r"""
    Max reduction over a single logical axis.

    This operator reduces a rank-3 tensor

    .. math::

        X \in \mathbb{R}^{B \times N \times D}

    by taking the maximum along one of the inner dimensions, as
    configured by :attr:`dim`:

    - ``dim == 1``: row-wise maximum over :math:`N`, producing

      .. math::

          Y \in \mathbb{R}^{B \times 1 \times D}, \qquad
          Y[b, 0, d]
          = \max_{0 \le n < N} X[b, n, d].

    - ``dim == 2``: column-wise maximum over :math:`D`, producing

      .. math::

          Y \in \mathbb{R}^{B \times N \times 1}, \qquad
          Y[b, n, 0]
          = \max_{0 \le d < D} X[b, n, d].

    Parameters
    ----------
    dim : int, optional
        Logical reduction dimension. Must be ``1`` (reduce over
        :math:`N`) or ``2`` (reduce over :math:`D`). Default is ``1``.
    """
    def __init__(self, dim: int = 1):
        super().__init__(dim)
        self.reduce_type = ReduceType.Max
        


class Min(Reduce):
    r"""
    Min reduction over a single logical axis.

    This operator reduces a rank-3 tensor

    .. math::

        X \in \mathbb{R}^{B \times N \times D}

    by taking the minimum along one of the inner dimensions, as
    configured by :attr:`dim`:

    - ``dim == 1``: row-wise minimum over :math:`N`, producing

      .. math::

          Y \in \mathbb{R}^{B \times 1 \times D}, \qquad
          Y[b, 0, d]
          = \min_{0 \le n < N} X[b, n, d].

    - ``dim == 2``: column-wise minimum over :math:`D`, producing

      .. math::

          Y \in \mathbb{R}^{B \times N \times 1}, \qquad
          Y[b, n, 0]
          = \min_{0 \le d < D} X[b, n, d].

    Parameters
    ----------
    dim : int, optional
        Logical reduction dimension. Must be ``1`` (reduce over
        :math:`N`) or ``2`` (reduce over :math:`D`). Default is ``1``.
    """
    def __init__(self, dim: int = 1):
        super().__init__(dim)
        self.reduce_type = ReduceType.Min



class L2Norm(Reduce):
    r"""
    L2-norm reduction (not RMS) over a single logical axis.

    This operator reduces a rank-3 tensor

    .. math::

        X \in \mathbb{R}^{B \times N \times D}

    by computing an L2 norm along one of the inner dimensions, as
    configured by :attr:`dim`. The reduction is *not* normalized by the
    number of elements (it is an L2 norm, not an RMS):

    - ``dim == 1``: row-wise L2 norm over :math:`N`, producing

      .. math::

          Y \in \mathbb{R}^{B \times 1 \times D}, \qquad
          Y[b, 0, d]
          = \sqrt{\sum_{n=0}^{N-1} X[b, n, d]^2}.

    - ``dim == 2``: column-wise L2 norm over :math:`D`, producing

      .. math::

          Y \in \mathbb{R}^{B \times N \times 1}, \qquad
          Y[b, n, 0]
          = \sqrt{\sum_{d=0}^{D-1} X[b, n, d]^2}.

    Notes
    -----
    This is a pure L2 norm over the reduced axis:

    .. math::

        \|v\|_2 = \sqrt{\sum_i v_i^2},

    with no division by the number of elements. It should not be
    confused with RMS (root mean square).

    Parameters
    ----------
    dim : int, optional
        Logical reduction dimension. Must be ``1`` (reduce over
        :math:`N`) or ``2`` (reduce over :math:`D`). Default is ``1``.
    """
    def __init__(self, dim: int = 1):
        super().__init__(dim)
        self.reduce_type = ReduceType.L2Norm