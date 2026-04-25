import torch
from ..abs import vOp, vTensor, FORMAT
from .context import Context
from ..utils import ElementwiseBinaryOpType, Schedule
from typing import Tuple, Dict, Optional


class Elementwise_Binary(vOp):
    r"""
    Binary elementwise operator dispatcher
    (e.g. Maximum / Minimum / AXPBY / Mul).

    This class dispatches a family of binary elementwise operations on
    rank-3 tensors. The inputs are treated as

    .. math::

        X, Y \in \mathbb{R}^{B \times N \times D},

    where:

    - :math:`B` is a leading batch-like axis (typically derived from the
      runtime context, e.g. ``max_new_tokens_per_batch * head_num``),
    - :math:`N` is a sequence or position dimension, and
    - :math:`D` is a feature/channel dimension.

    Broadcasting is supported on the last two dimensions:

    - :math:`N` is broadcastable if ``x.shape[1] == y.shape[1]``,
      or one of them equals ``1``.
    - :math:`D` is broadcastable if ``x.shape[2] == y.shape[2]``,
      or one of them equals ``1``.

    For a given operation type :attr:`op_type`, the dispatcher applies a
    scalar function

    .. math::

        f(x, y; \alpha, \beta, \text{op_type})

    pointwise to produce

    .. math::

        Z[b, n, d]
        = f\bigl(X[b, n', d'], Y[b, n'', d'']; \alpha, \beta, \text{op_type}\bigr),

    where :math:`(n', d')` and :math:`(n'', d'')` are the broadcasted
    indices derived from :math:`(n, d)`.

    Dispatch is based on the triplet of tensor formats
    ``(x_format, y_format, o_format)`` and a registry mapping:

    .. code-block:: text

        (x_format, y_format, o_format) -> (impl, resolved_output_format)

    Policy
    ------
    - If ``output`` is ``None``:

      - :meth:`profile` selects an implementation with
        ``o_format == FORMAT.RAGGED``, i.e. a key
        ``(x_fmt, y_fmt, FORMAT.RAGGED)`` in :attr:`_impl_map`.
      - An internal buffer of shape ``[B, N_out, D_out]`` is allocated,
        where

        .. math::

            N_{\text{out}} = \max(N_x, N_y), \quad
            D_{\text{out}} = \max(D_x, D_y),

        and :math:`B` is derived from the runtime context.

    - If ``output`` is provided:

      - :meth:`profile` requires an exact implementation key for
        ``(x_fmt, y_fmt, o_fmt)``.
      - The shape of ``output`` must match the broadcasted
        ``(N_out, D_out)``.
      - Device consistency is enforced for ``x``, ``y`` and ``output``.

    Attributes
    ----------
    _impl_map : Dict[Tuple[FORMAT, FORMAT, FORMAT], FORMAT]
        Dispatch table keyed by ``(x_format, y_format, o_format)``.
        Each entry maps to the resolved output format.

    alpha : float
        Scalar parameter used by certain binary ops.
    beta : float
        Scalar parameter used by certain binary ops.
    op_type : Optional[ElementwiseBinaryOpType]
        Enum value describing the specific binary operation
        (e.g. maximum, minimum, AXPBY, multiply).
    output_format : Optional[FORMAT]
        The output tensor format as determined in :meth:`profile`.
    output_buffer : Optional[vTensor]
        Pure-metadata vTensor descriptor for the output (graph node).
    """

    _impl_map: Dict[Tuple[FORMAT, FORMAT, FORMAT], FORMAT] = {
        (FORMAT.PAGED,  FORMAT.PAGED,  FORMAT.PAGED):  FORMAT.PAGED,
        (FORMAT.PAGED,  FORMAT.PAGED,  FORMAT.RAGGED): FORMAT.RAGGED,
        (FORMAT.PAGED,  FORMAT.RAGGED, FORMAT.PAGED):  FORMAT.PAGED,
        (FORMAT.PAGED,  FORMAT.RAGGED, FORMAT.RAGGED): FORMAT.RAGGED,
        (FORMAT.RAGGED, FORMAT.PAGED,  FORMAT.PAGED):  FORMAT.PAGED,
        (FORMAT.RAGGED, FORMAT.PAGED,  FORMAT.RAGGED): FORMAT.RAGGED,
        (FORMAT.RAGGED, FORMAT.RAGGED, FORMAT.PAGED):  FORMAT.PAGED,
        (FORMAT.RAGGED, FORMAT.RAGGED, FORMAT.RAGGED): FORMAT.RAGGED,
    }

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.op_type: Optional[ElementwiseBinaryOpType] = None
        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[vTensor] = None
        # Cache binary elementwise ops fuse into the per-block kernel —
        # see ``cache.compiler.triton_impl.kernel_gen``.
        self.schedule = Schedule.W

    # ------------------------------ helpers ------------------------------ #
    def _infer_output_format_ragged(
        self, x_fmt: FORMAT, y_fmt: FORMAT
    ) -> FORMAT:
        r"""
        Infer an implementation assuming a RAGGED output format.

        This helper is used when :meth:`profile` is called with
        ``output is None``. It selects an implementation for the key
        ``(x_fmt, y_fmt, FORMAT.RAGGED)`` in :attr:`_impl_map`.

        Parameters
        ----------
        x_fmt : FORMAT
            Format of the left-hand operand ``x``.

        y_fmt : FORMAT
            Format of the right-hand operand ``y``.

        Returns
        -------
        (Callable, FORMAT)
            The implementation callable and the resolved output format.

        Raises
        ------
        AssertionError
            If there is no entry for
            ``(x_fmt, y_fmt, FORMAT.RAGGED)`` in :attr:`_impl_map`.
        """
        key = (x_fmt, y_fmt, FORMAT.RAGGED)
        assert key in self._impl_map, (
            f"{self._prefix()}no RAGGED-output implementation for "
            f"(x_fmt={x_fmt}, y_fmt={y_fmt}). Available keys: {list(self._impl_map.keys())}"
        )
        return self._impl_map[key]

    # ------------------------------------------------------------------ #
    def profile(
        self, x: vTensor, y: vTensor, output: Optional[vTensor], loc: torch.Tensor, ctx: Context
    ) -> vTensor:
        r"""
        Validate inputs, resolve the implementation and output format,
        and optionally allocate an internal output buffer.

        The input tensors ``x`` and ``y`` are expected to have logical
        shape ``[B, N_x, D_x]`` and ``[B, N_y, D_y]`` respectively,
        with broadcasting allowed on ``N`` and ``D``:

        .. math::

            N_{\text{out}} = \max(N_x, N_y), \quad
            D_{\text{out}} = \max(D_x, D_y).

        The auxiliary tensor ``loc`` carries per-position metadata used
        by the implementation (for example, indices or segment offsets);
        its shape and semantics are kernel-defined.

        Parameters
        ----------
        x : vTensor
            Left-hand operand with logical shape ``[B, N_x, D_x]``.

        y : vTensor
            Right-hand operand with logical shape ``[B, N_y, D_y]``.

        output : Optional[vTensor]
            Optional preallocated output tensor. If ``None``, an internal
            buffer with shape ``[B, N_out, D_out]`` is allocated using
            ``ctx.max_new_tokens_per_batch * ctx.head_num`` for ``B`` and a
            RAGGED-output implementation is selected. If not ``None``,
            this tensor must have rank 3, broadcasted shape
            ``[B_out, N_out, D_out]`` and a format compatible with
            :attr:`_impl_map`.

        loc : torch.Tensor
            Auxiliary tensor carrying per-position metadata used by the
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
            If types, ranks, broadcast conditions, formats, shapes, or
            devices are incompatible, or if no implementation is found in
            :attr:`_impl_map`.
        """
        prefix = self._prefix()

        # --- type checks ---
        assert isinstance(x, vTensor), f"{prefix}x must be vTensor, got {type(x)}"
        assert isinstance(y, vTensor), f"{prefix}y must be vTensor, got {type(y)}"
        assert isinstance(loc, torch.Tensor), f"{prefix}loc must be torch.Tensor, got {type(loc)}"

        assert x.dim() == 3, f"{prefix}x must be 3D, got ndim={x.dim()} shape={tuple(x.shape)}"
        assert y.dim() == 3, f"{prefix}y must be 3D, got ndim={y.dim()} shape={tuple(y.shape)}"

        # --- broadcastability checks ---
        assert (
            x.shape[1] == y.shape[1] or x.shape[1] == 1 or y.shape[1] == 1
        ), f"{prefix}dim-1 not broadcastable: x={x.shape}, y={y.shape}"
        assert (
            x.shape[2] == y.shape[2] or x.shape[2] == 1 or y.shape[2] == 1
        ), f"{prefix}dim-2 not broadcastable: x={x.shape}, y={y.shape}"

        x_fmt, y_fmt = x._format, y._format
        exp_N, exp_D = max(x.shape[1], y.shape[1]), max(x.shape[2], y.shape[2])

        # Case A: output None → pick RAGGED impl, construct a metadata
        # vTensor, and register it in the cache graph.
        if output is None:
            self.output_format = self._infer_output_format_ragged(x_fmt, y_fmt)
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
            ctx.op_to_input_tensor_list.append([x.tensor_id, y.tensor_id])
            ctx.op_to_output_tensor_list.append([self.output_buffer.tensor_id])
            return self.output_buffer

        # Case B: output provided → exact match
        assert isinstance(output, vTensor), f"{prefix}output must be vTensor, got {type(output)}"
        assert output.dim() == 3, (
            f"{prefix}output must be 3D, got ndim={output.dim()} shape={tuple(output.shape)}"
        )

        o_fmt = output._format
        key = (x_fmt, y_fmt, o_fmt)
        assert key in self._impl_map, (
            f"{prefix}no implementation for (x_fmt={x_fmt}, y_fmt={y_fmt}, o_fmt={o_fmt}). "
            f"Available keys: {list(self._impl_map.keys())}"
        )
        self.output_format = self._impl_map[key]

        assert output.shape[1] == exp_N and output.shape[2] == exp_D, (
            f"{prefix}output shape mismatch. Expected (*,{exp_N},{exp_D}), got {tuple(output.shape)}"
        )
        assert x.device == y.device == output.device, (
            f"{prefix}x, y, and output must be on the same device "
            f"(x.device={x.device}, y.device={y.device}, output.device={output.device})"
        )

        # Register in the cache graph (caller-provided output path).
        ctx.output_tensor_to_op_list[output.tensor_id] = len(ctx.op_list)
        ctx.op_list.append(self)
        ctx.op_to_input_tensor_list.append([x.tensor_id, y.tensor_id])
        ctx.op_to_output_tensor_list.append([output.tensor_id])

        return output


class Maximum(Elementwise_Binary):
    r"""
    Elementwise maximum of two tensors.

    This operator applies, pointwise, the scalar function

    .. math::

        f(x, y) = \max(x, y).

    Given two input tensors

    .. math::

        X, Y \in \mathbb{R}^{B \times N \times D},

    with broadcasting allowed on the ``N`` and ``D`` dimensions, the
    output tensor :math:`Z` is defined by

    .. math::

        Z[b, n, d] = \max\bigl(X[b, n', d'], Y[b, n'', d'']\bigr),

    where :math:`(n', d')` and :math:`(n'', d'')` are the broadcasted
    indices corresponding to :math:`(n, d)`.

    Parameters
    ----------
    alpha : float, optional
        Unused for this operation. Present only to match the common
        :class:`Elementwise_Binary` interface. Default is ``1``.

    beta : float, optional
        Unused for this operation. Present only to match the common
        :class:`Elementwise_Binary` interface. Default is ``1``.
    """
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseBinaryOpType.Maximum


class Minimum(Elementwise_Binary):
    r"""
    Elementwise minimum of two tensors.

    This operator applies, pointwise, the scalar function

    .. math::

        f(x, y) = \min(x, y).

    Given two input tensors

    .. math::

        X, Y \in \mathbb{R}^{B \times N \times D},

    with broadcasting allowed on the ``N`` and ``D`` dimensions, the
    output tensor :math:`Z` is defined by

    .. math::

        Z[b, n, d] = \min\bigl(X[b, n', d'], Y[b, n'', d'']\bigr),

    where :math:`(n', d')` and :math:`(n'', d'')` are the broadcasted
    indices corresponding to :math:`(n, d)`.

    Parameters
    ----------
    alpha : float, optional
        Unused for this operation. Present only to match the common
        :class:`Elementwise_Binary` interface. Default is ``1``.

    beta : float, optional
        Unused for this operation. Present only to match the common
        :class:`Elementwise_Binary` interface. Default is ``1``.
    """
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseBinaryOpType.Minimum
        

class Add(Elementwise_Binary):
    r"""
    Weighted sum (AXPBY-style) of two tensors.

    This operator applies, pointwise, the scalar function

    .. math::

        f(x, y; \alpha, \beta) = \alpha x + \beta y.

    Given two input tensors

    .. math::

        X, Y \in \mathbb{R}^{B \times N \times D},

    with broadcasting allowed on the ``N`` and ``D`` dimensions, the
    output tensor :math:`Z` is defined by

    .. math::

        Z[b, n, d]
        = \alpha \, X[b, n', d'] + \beta \, Y[b, n'', d''],

    where :math:`(n', d')` and :math:`(n'', d'')` are the broadcasted
    indices corresponding to :math:`(n, d)`.

    Parameters
    ----------
    alpha : float, optional
        Coefficient :math:`\alpha` applied to the first input tensor.
        Default is ``1``.

    beta : float, optional
        Coefficient :math:`\beta` applied to the second input tensor.
        Default is ``1``.
    """
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseBinaryOpType.Add


class Multiply(Elementwise_Binary):
    r"""
    Elementwise product of two tensors.

    This operator applies, pointwise, the scalar function

    .. math::

        f(x, y) = x \cdot y.

    Given two input tensors

    .. math::

        X, Y \in \mathbb{R}^{B \times N \times D},

    with broadcasting allowed on the ``N`` and ``D`` dimensions, the
    output tensor :math:`Z` is defined by

    .. math::

        Z[b, n, d]
        = X[b, n', d'] \cdot Y[b, n'', d''],

    where :math:`(n', d')` and :math:`(n'', d'')` are the broadcasted
    indices corresponding to :math:`(n, d)`.

    Parameters
    ----------
    alpha : float, optional
        Unused for this operation. Present only to match the common
        :class:`Elementwise_Binary` interface. Default is ``1``.

    beta : float, optional
        Unused for this operation. Present only to match the common
        :class:`Elementwise_Binary` interface. Default is ``1``.
    """
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseBinaryOpType.Mul


class WhereEqual(Elementwise_Binary):
    r"""
    Comparison mask encoded as ``0`` or ``-inf``.

    This operator compares the two inputs elementwise and emits:

    .. math::

        f(x, y) =
        \begin{cases}
            0, & x = y, \\
            -\infty, & x \ne y.
        \end{cases}

    This is primarily useful for building additive masks in score-space.
    """
    def __init__(self):
        super().__init__()
        self.op_type = ElementwiseBinaryOpType.WhereEqual


class WhereNotEqual(Elementwise_Binary):
    r"""Emit ``0`` where ``x != y`` and ``-inf`` otherwise."""
    def __init__(self):
        super().__init__()
        self.op_type = ElementwiseBinaryOpType.WhereNotEqual


class WhereGreater(Elementwise_Binary):
    r"""Emit ``0`` where ``x > y`` and ``-inf`` otherwise."""
    def __init__(self):
        super().__init__()
        self.op_type = ElementwiseBinaryOpType.WhereGreater


class WhereGreaterEqual(Elementwise_Binary):
    r"""Emit ``0`` where ``x >= y`` and ``-inf`` otherwise."""
    def __init__(self):
        super().__init__()
        self.op_type = ElementwiseBinaryOpType.WhereGreaterEqual


class WhereLess(Elementwise_Binary):
    r"""Emit ``0`` where ``x < y`` and ``-inf`` otherwise."""
    def __init__(self):
        super().__init__()
        self.op_type = ElementwiseBinaryOpType.WhereLess


class WhereLessEqual(Elementwise_Binary):
    r"""Emit ``0`` where ``x <= y`` and ``-inf`` otherwise."""
    def __init__(self):
        super().__init__()
        self.op_type = ElementwiseBinaryOpType.WhereLessEqual
