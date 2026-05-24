import torch
from typing import Optional
from .context import Context
from ..abs import vTensor, FORMAT, vOp
from ..utils import ElementwiseOpType, Schedule

class Elementwise(vOp):
    r"""
    Unary elementwise op — applies a scalar function pointwise.

    :Math:
        .. math::

            Y_{s,c,d} = f(X_{s,c,d};\, \alpha, \beta),

        where :math:`f` is fixed by the subclass (ReLU / SiLU / Sigmoid /
        affine / abs / log / exp).
    :__init__: ``Elementwise(alpha=1.0, beta=1.0)`` — scalar parameters
        :math:`\alpha`, :math:`\beta` consumed by :math:`f`.
    :__call__: ``y = op(x, ctx=ctx)`` — ``x`` is ``[S, C, D]``; returns the same
        shape. Output is ``BATCHED`` iff the input is, else ``RAGGED``.
    :Note: use a concrete subclass — :class:`Relu`, :class:`Silu`,
        :class:`Sigmoid`, :class:`Add_Mul`, :class:`Abs`, :class:`Log`,
        :class:`Exp`.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__()
        self.op_type: Optional[ElementwiseOpType] = None
        self.alpha = alpha
        self.beta = beta
        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[torch.Tensor] = None
        self.schedule = Schedule.W

    def profile(self, x: vTensor, ctx: Context) -> vTensor:
        r"""Trace-time: validate ``x`` ``[S, C, D]``, register the op, and return
        a same-shape ``vTensor`` view (``BATCHED`` iff ``x`` is, else
        ``RAGGED``)."""
        prefix = self._prefix()

        # Type & rank checks
        assert isinstance(x, vTensor), f"{prefix}profile expects x to be vTensor, got {type(x)}"
        assert x.dim() == 3, (
            f"{prefix}expected 3D input [S, C, D]. Got ndim={x.dim()} shape={tuple(x.shape)}"
        )

        # Output is BATCHED iff the input is BATCHED (S already collapsed
        # to 1); otherwise RAGGED.
        self.output_format = (
            FORMAT.BATCHED if x._format == FORMAT.BATCHED else FORMAT.RAGGED
        )

        C, D = x.shape[1], x.shape[2]

        # Pure-metadata vTensor — no torch.empty allocation needed.
        self.output_buffer = vTensor(
            shape=(0, C, D),
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


class Relu(Elementwise):
    r"""
    ReLU-like activation with threshold/fallback (an :class:`Elementwise`).

    :Math:
        .. math::

            f(x;\alpha,\beta) = \begin{cases} x, & x \ge \alpha, \\ \beta, & x < \alpha. \end{cases}
    :__init__: ``Relu(alpha=0.0, beta=0.0)`` — threshold :math:`\alpha`,
        fallback value :math:`\beta` (used when :math:`x<\alpha`).
    """
    def __init__(self, alpha: float = 0.0, beta: float = 0.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseOpType.Relu



class Silu(Elementwise):
    r"""
    SiLU-like activation with affine pre-transform (an :class:`Elementwise`).

    :Math:
        .. math::

            f(x;\alpha,\beta) = \frac{x}{1 + \exp(\beta x + \alpha)}.
    :__init__: ``Silu(alpha=0.0, beta=0.0)`` — bias :math:`\alpha`, slope
        :math:`\beta` inside the exponential.
    """
    def __init__(self, alpha: float = 0.0, beta: float = 0.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseOpType.Silu
        

class Sigmoid(Elementwise):
    r"""
    Sigmoid activation with affine argument (an :class:`Elementwise`).

    :Math:
        .. math::

            f(x;\alpha,\beta) = \frac{1}{1 + \exp(\beta x + \alpha)}.
    :__init__: ``Sigmoid(alpha=0.0, beta=0.0)`` — bias :math:`\alpha`, slope
        :math:`\beta` inside the exponential.
    """
    def __init__(self, alpha: float = 0.0, beta: float = 0.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseOpType.Sigmoid
        

class Add_Mul(Elementwise):
    r"""
    Affine transform :math:`\beta x + \alpha` (an :class:`Elementwise`).

    :Math:
        .. math::

            f(x;\alpha,\beta) = \beta x + \alpha.
    :__init__: ``Add_Mul(alpha=0.0, beta=1.0)`` — additive :math:`\alpha`,
        multiplicative :math:`\beta` (defaults give the identity).
    """
    def __init__(self, alpha: float = 0.0, beta: float = 1.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseOpType.Add_Mul


class Abs(Elementwise):
    r"""
    Absolute value of an affine transform (an :class:`Elementwise`).

    :Math:
        .. math::

            f(x;\alpha,\beta) = \lvert \beta x + \alpha \rvert.
    :__init__: ``Abs(alpha=0.0, beta=1.0)`` — additive :math:`\alpha`,
        multiplicative :math:`\beta` inside the absolute value.
    """
    def __init__(self, alpha: float = 0.0, beta: float = 1.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseOpType.Abs


class Log(Elementwise):
    r"""
    Natural logarithm of an affine transform (an :class:`Elementwise`).

    :Math:
        .. math::

            f(x;\alpha,\beta) = \log(\beta x + \alpha).
    :__init__: ``Log(alpha=0.0, beta=1.0)`` — additive :math:`\alpha`,
        multiplicative :math:`\beta` inside the logarithm.
    """
    def __init__(self, alpha: float = 0.0, beta: float = 1.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseOpType.Log


class Exp(Elementwise):
    r"""
    Exponential of an affine transform (an :class:`Elementwise`).

    :Math:
        .. math::

            f(x;\alpha,\beta) = \exp(\beta x + \alpha).
    :__init__: ``Exp(alpha=0.0, beta=1.0)`` — additive :math:`\alpha`,
        multiplicative :math:`\beta` inside the exponential.
    """
    def __init__(self, alpha: float = 0.0, beta: float = 1.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseOpType.Exp
