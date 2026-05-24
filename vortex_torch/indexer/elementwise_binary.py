import torch
from typing import Optional
from .context import Context
from ..abs import vTensor, FORMAT, vOp
from ..utils import ElementwiseBinaryOpType, Schedule

class Elementwise_Binary(vOp):
    r"""
    Binary elementwise op — combines two tensors pointwise (with broadcasting).

    :Math:
        .. math::

            Z_{s,c,d} = g(X_{s,c,d},\, Y_{s,c,d};\, \alpha, \beta),

        where :math:`g` is fixed by the subclass (max / min / affine-sum /
        product / comparison) and the inner ``(C, D)`` axes broadcast.
    :__init__: ``Elementwise_Binary(alpha=1.0, beta=1.0)`` — scalar parameters
        used by some ops (e.g. the affine sum :math:`\alpha x + \beta y`).
    :__call__: ``z = op(x, y, ctx=ctx)`` — ``x`` / ``y`` ``[S, C, D]`` (broadcast
        on ``C, D``); output ``[S, max(C_x,C_y), max(D_x,D_y)]``. ``BATCHED``
        iff both inputs are, else ``RAGGED``.
    :Note: use a concrete subclass — :class:`Maximum`, :class:`Minimum`,
        :class:`Add`, :class:`Multiply`, or a comparison mask
        (:class:`WhereGreater`, :class:`WhereEqual`, …).
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__()
        self.op_type: Optional[ElementwiseBinaryOpType] = None
        self.alpha = alpha
        self.beta = beta
        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[torch.Tensor] = None
        self.schedule = Schedule.W
    # ---------------- profile ----------------
    def profile(self, x: vTensor, y: vTensor, ctx: Context) -> vTensor:
        r"""Trace-time: validate ``x`` / ``y`` (rank-3, broadcastable on ``C`` /
        ``D``), register the op, and return a ``vTensor`` view of the broadcast
        output (see the class docstring for shapes)."""
        prefix = self._prefix()

        # Type checks
        assert isinstance(x, vTensor), f"{prefix}profile expects x to be vTensor, got {type(x)}"
        assert isinstance(y, vTensor), f"{prefix}profile expects y to be vTensor, got {type(y)}"

        # Rank & basic shape checks
        assert x.dim() == 3 and y.dim() == 3, (
            f"{prefix}expected 3D inputs [S, C, D]; got x.ndim={x.dim()}, y.ndim={y.dim()}"
        )

        # Broadcastability on C/D
        assert (x.shape[1] == y.shape[1] or x.shape[1] == 1 or y.shape[1] == 1), (
            f"{prefix}dim-1 not broadcastable: x.shape={tuple(x.shape)}, y.shape={tuple(y.shape)}"
        )
        assert (x.shape[2] == y.shape[2] or x.shape[2] == 1 or y.shape[2] == 1), (
            f"{prefix}dim-2 not broadcastable: x.shape={tuple(x.shape)}, y.shape={tuple(y.shape)}"
        )

        # Output preserves BATCHED iff both inputs are BATCHED, else RAGGED.
        self.output_format = (
            FORMAT.BATCHED
            if (x._format == FORMAT.BATCHED and y._format == FORMAT.BATCHED)
            else FORMAT.RAGGED
        )

        # Device consistency
        assert x.device == y.device, (
            f"{prefix}x and y must be on the same device "
            f"(x.device={x.device}, y.device={y.device})"
        )

        # Broadcasted output (C, D)
        C_out = max(x.shape[1], y.shape[1])
        D_out = max(x.shape[2], y.shape[2])

        # Pure-metadata vTensor — no torch.empty allocation needed.
        self.output_buffer = vTensor(
            shape=(0, C_out, D_out),
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


class Maximum(Elementwise_Binary):
    r"""
    Elementwise maximum (an :class:`Elementwise_Binary`).

    :Math:
        .. math::

            Z_{s,c,d} = \max(X_{s,c,d},\, Y_{s,c,d}).
    :__init__: ``Maximum(alpha=1.0, beta=1.0)`` — ``alpha`` / ``beta`` unused.
    """
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseBinaryOpType.Maximum


class Minimum(Elementwise_Binary):
    r"""
    Elementwise minimum (an :class:`Elementwise_Binary`).

    :Math:
        .. math::

            Z_{s,c,d} = \min(X_{s,c,d},\, Y_{s,c,d}).
    :__init__: ``Minimum(alpha=1.0, beta=1.0)`` — ``alpha`` / ``beta`` unused.
    """
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseBinaryOpType.Minimum
        

class Add(Elementwise_Binary):
    r"""
    Affine combination :math:`\alpha x + \beta y` (an :class:`Elementwise_Binary`).

    :Math:
        .. math::

            Z_{s,c,d} = \alpha\,X_{s,c,d} + \beta\,Y_{s,c,d}.
    :__init__: ``Add(alpha=1.0, beta=1.0)`` — multipliers for :math:`x` and
        :math:`y` (defaults give :math:`x+y`).
    """
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseBinaryOpType.Add
        

class Multiply(Elementwise_Binary):
    r"""
    Elementwise product (an :class:`Elementwise_Binary`).

    :Math:
        .. math::

            Z_{s,c,d} = X_{s,c,d}\cdot Y_{s,c,d}.
    :__init__: ``Multiply(alpha=1.0, beta=1.0)`` — ``alpha`` / ``beta`` unused.
    """
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseBinaryOpType.Mul


class WhereEqual(Elementwise_Binary):
    r"""
    Comparison mask :math:`x = y` (an :class:`Elementwise_Binary`).

    :Math:
        .. math::

            Z_{s,c,d} = \begin{cases} 0, & X_{s,c,d} = Y_{s,c,d}, \\ -\infty, & \text{otherwise}. \end{cases}
    :__init__: ``WhereEqual()`` — no arguments.
    :Note: additive mask, intended to gate a score tensor before
        :class:`vortex_torch.indexer.Softmax` / :func:`vortex_torch.indexer.topK`.
    """
    def __init__(self):
        super().__init__()
        self.op_type = ElementwiseBinaryOpType.WhereEqual


class WhereNotEqual(Elementwise_Binary):
    r"""
    Comparison mask :math:`x \ne y` (an :class:`Elementwise_Binary`).

    :Math:
        .. math::

            Z_{s,c,d} = \begin{cases} 0, & X_{s,c,d} \ne Y_{s,c,d}, \\ -\infty, & \text{otherwise}. \end{cases}
    :__init__: ``WhereNotEqual()`` — no arguments.
    """
    def __init__(self):
        super().__init__()
        self.op_type = ElementwiseBinaryOpType.WhereNotEqual


class WhereGreater(Elementwise_Binary):
    r"""
    Comparison mask :math:`x > y` (an :class:`Elementwise_Binary`).

    :Math:
        .. math::

            Z_{s,c,d} = \begin{cases} 0, & X_{s,c,d} > Y_{s,c,d}, \\ -\infty, & \text{otherwise}. \end{cases}
    :__init__: ``WhereGreater()`` — no arguments.
    """
    def __init__(self):
        super().__init__()
        self.op_type = ElementwiseBinaryOpType.WhereGreater


class WhereGreaterEqual(Elementwise_Binary):
    r"""
    Comparison mask :math:`x \ge y` (an :class:`Elementwise_Binary`).

    :Math:
        .. math::

            Z_{s,c,d} = \begin{cases} 0, & X_{s,c,d} \ge Y_{s,c,d}, \\ -\infty, & \text{otherwise}. \end{cases}
    :__init__: ``WhereGreaterEqual()`` — no arguments.
    """
    def __init__(self):
        super().__init__()
        self.op_type = ElementwiseBinaryOpType.WhereGreaterEqual


class WhereLess(Elementwise_Binary):
    r"""
    Comparison mask :math:`x < y` (an :class:`Elementwise_Binary`).

    :Math:
        .. math::

            Z_{s,c,d} = \begin{cases} 0, & X_{s,c,d} < Y_{s,c,d}, \\ -\infty, & \text{otherwise}. \end{cases}
    :__init__: ``WhereLess()`` — no arguments.
    """
    def __init__(self):
        super().__init__()
        self.op_type = ElementwiseBinaryOpType.WhereLess


class WhereLessEqual(Elementwise_Binary):
    r"""
    Comparison mask :math:`x \le y` (an :class:`Elementwise_Binary`).

    :Math:
        .. math::

            Z_{s,c,d} = \begin{cases} 0, & X_{s,c,d} \le Y_{s,c,d}, \\ -\infty, & \text{otherwise}. \end{cases}
    :__init__: ``WhereLessEqual()`` — no arguments.
    """
    def __init__(self):
        super().__init__()
        self.op_type = ElementwiseBinaryOpType.WhereLessEqual
