import torch
from typing import Optional
from .context import Context
from ..abs import vTensor, FORMAT, vOp
from ..utils import ElementwiseOpType, Schedule

class Elementwise(vOp):
    """
    Unary elementwise op for rank-3 logical tensors ``[S, C, D]``.

    Output format rule: ``BATCHED`` iff the input is ``BATCHED`` (its
    ``S`` axis is already collapsed to 1), otherwise ``RAGGED``. Format
    compatibility is enforced by the compiler's per-workload kernel.

    Attributes
    ----------
    alpha : float
        Scalar parameter used by some ops. Default is ``1.0``.

    beta : float
        Scalar parameter used by some ops. Default is ``1.0``.

    op_type : Optional[ElementwiseOpType]
        The operator type used by the implementation.

    output_format : Optional[FORMAT]
        The output tensor format as determined in :meth:`profile`.

    output_buffer : Optional[torch.Tensor]
        Preallocated output tensor buffer.
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
        """
        Validate input, allocate the output buffer, and return a
        ``vTensor`` view. The output is ``BATCHED`` iff the input is
        ``BATCHED``, otherwise ``RAGGED``.

        Parameters
        ----------
        x : vTensor
            Input tensor. Must be rank-3 with shape ``[S, C, D]``.

        ctx : Context
            Execution context providing runtime ``S`` (``ctx.max_num_pages``)
            and memory tracking.

        Returns
        -------
        vTensor
            A ``vTensor`` view wrapping the allocated output buffer.

        Raises
        ------
        AssertionError
            If input tensor type or rank is invalid.
        """
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
    ReLU-style elementwise operator.

    This operator applies a thresholded linear function:

    .. math::

        \operatorname{out}(x) =
        \begin{cases}
            x, & x \ge \alpha \\
            \beta, & x < \alpha
        \end{cases}

    Parameters
    ----------
    alpha : float, optional
        Threshold value for activation. Default is ``0.0``.

    beta : float, optional
        Value used when :math:`x < \alpha`. Default is ``0.0``.

    """
    def __init__(self, alpha: float = 0.0, beta: float = 0.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseOpType.Relu



class Silu(Elementwise):
    r"""
    SiLU-style elementwise operator with affine pre-transform.

    This operator applies:

    .. math::

        \operatorname{SiLU}_{\alpha,\beta}(x)
        = \frac{x}{1 + \exp(\beta x + \alpha)}

    When :math:`\alpha = 0` and :math:`\beta = -1`, this reduces to the
    common SiLU/Swish-like form :math:`x \, \sigma(x)` (up to the chosen
    parameterization in the kernel).

    Parameters
    ----------
    alpha : float, optional
        Bias term inside the exponent, used in :math:`\beta x + \alpha`.
        Default is ``0.0``.

    beta : float, optional
        Scale term inside the exponent, used in :math:`\beta x + \alpha`.
        Default is ``0.0``.
    """
    def __init__(self, alpha: float = 0.0, beta: float = 0.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseOpType.Silu
        

class Sigmoid(Elementwise):
    r"""
    Sigmoid elementwise operator with affine argument.

    This operator applies:

    .. math::

        \sigma_{\alpha,\beta}(x)
        = \frac{1}{1 + \exp(\beta x + \alpha)}

    When :math:`\alpha = 0` and :math:`\beta = -1`, this is the standard
    logistic sigmoid :math:`\sigma(x) = 1 / (1 + e^{-x})`.

    Parameters
    ----------
    alpha : float, optional
        Bias term inside the exponent, used in :math:`\beta x + \alpha`.
        Default is ``0.0``.

    beta : float, optional
        Scale term inside the exponent, used in :math:`\beta x + \alpha`.
        Default is ``0.0``.
    """
    def __init__(self, alpha: float = 0.0, beta: float = 0.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseOpType.Sigmoid
        

class Add_Mul(Elementwise):
    r"""
    Affine elementwise transform.

    This operator applies a simple affine mapping:

    .. math::

        \operatorname{out}(x) = \beta x + \alpha

    With the defaults :math:`\alpha = 0` and :math:`\beta = 1`, this is
    the identity transform :math:`\operatorname{out}(x) = x`.

    Parameters
    ----------
    alpha : float, optional
        Additive bias term :math:`\alpha`. Default is ``0.0``.

    beta : float, optional
        Multiplicative scale term :math:`\beta`. Default is ``1.0``.
    """
    def __init__(self, alpha: float = 0.0, beta: float = 1.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseOpType.Add_Mul


class Abs(Elementwise):
    r"""
    Absolute value of an affine transform.

    This operator applies:

    .. math::

        \operatorname{out}(x) = \lvert \beta x + \alpha \rvert

    With the defaults :math:`\alpha = 0` and :math:`\beta = 1`, this
    reduces to the standard absolute value :math:`\lvert x \rvert`.

    Parameters
    ----------
    alpha : float, optional
        Additive bias term inside the absolute value, used in
        :math:`\beta x + \alpha`. Default is ``0.0``.

    beta : float, optional
        Multiplicative scale term inside the absolute value, used in
        :math:`\beta x + \alpha`. Default is ``1.0``.
    """
    def __init__(self, alpha: float = 0.0, beta: float = 1.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseOpType.Abs


class Log(Elementwise):
    r"""
    Natural logarithm of an affine transform.

    This operator applies:

    .. math::

        \operatorname{out}(x) = \log(\beta x + \alpha)

    With the defaults :math:`\alpha = 0` and :math:`\beta = 1`, this
    reduces to the standard natural logarithm :math:`\log(x)`.

    Parameters
    ----------
    alpha : float, optional
        Additive bias term inside the logarithm, used in
        :math:`\beta x + \alpha`. Default is ``0.0``.

    beta : float, optional
        Multiplicative scale term inside the logarithm, used in
        :math:`\beta x + \alpha`. Default is ``1.0``.
    """
    def __init__(self, alpha: float = 0.0, beta: float = 1.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseOpType.Log


class Exp(Elementwise):
    r"""
    Exponential of an affine transform.

    This operator applies:

    .. math::

        \operatorname{out}(x) = \exp(\beta x + \alpha)

    With the defaults :math:`\alpha = 0` and :math:`\beta = 1`, this
    reduces to the standard exponential :math:`\exp(x)`.

    Parameters
    ----------
    alpha : float, optional
        Additive bias term inside the exponential, used in
        :math:`\beta x + \alpha`. Default is ``0.0``.

    beta : float, optional
        Multiplicative scale term inside the exponential, used in
        :math:`\beta x + \alpha`. Default is ``1.0``.
    """
    def __init__(self, alpha: float = 0.0, beta: float = 1.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseOpType.Exp
