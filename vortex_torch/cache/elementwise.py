import torch
from ..abs import vOp, vTensor, FORMAT
from .context import Context
from ..utils import ElementwiseOpType, Schedule
from typing import Optional

class Elementwise(vOp):
    r"""
    Unary elementwise op (e.g. ReLU/Sigmoid/SiLU/Abs/Affine).

    Operates on rank-3 tensors

    .. math::

        X \in \mathbb{R}^{B \times N \times D},

    where:

    - :math:`B` is a leading batch-like axis (for example,
      ``max_new_tokens_per_batch * head_num`` coming from the runtime
      context),
    - :math:`N` is a sequence or position dimension, and
    - :math:`D` is a feature/channel dimension.

    The operation is applied pointwise:

    .. math::

        Y[b, n, d] = f(X[b, n, d]; \alpha, \beta, \text{op_type}),

    where the actual function :math:`f` is selected by :attr:`op_type`.

    Output format rule: if a caller-provided ``output`` is supplied with
    ``PAGED`` format, the output is ``PAGED``; in every other case
    (``output is None``, or ``output._format == RAGGED``) the output is
    ``RAGGED``. Format compatibility is enforced by the compiler's
    per-block kernel.

    Attributes
    ----------
    alpha : float
        Scalar parameter used by certain unary ops.
    beta : float
        Scalar parameter used by certain unary ops.
    op_type : Optional[ElementwiseOpType]
        Runtime-set enum/int describing the specific elementwise operation.
    output_format : Optional[FORMAT]
        The output tensor format as determined in :meth:`profile`.
    output_buffer : Optional[vTensor]
        Pure-metadata vTensor descriptor for the output (graph node).
    """

    def __init__(self, alpha: float = 0.0, beta: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.op_type: Optional[ElementwiseOpType] = None
        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[vTensor] = None
        # Cache elementwise ops fuse with neighbours into a single
        # token-driven kernel — see ``cache.compiler.triton_impl.kernel_gen``.
        self.schedule = Schedule.W

    # --------------------------------------------------------------------- #
    # profile: validate and optionally allocate output
    # --------------------------------------------------------------------- #
    def profile(
        self, x: vTensor, output: Optional[vTensor], loc: torch.Tensor, ctx: Context
    ) -> vTensor:
        r"""
        Validate inputs and optionally allocate an internal output buffer.

        The input tensor ``x`` is expected to have logical shape
        ``[B, N, D]``.

        Two modes:

        - **No output provided** (``output is None``):

          - Allocate an internal RAGGED buffer with shape ``[B, N, D]``,
            where

            .. math::

                B = \text{ctx.max_new_tokens_per_batch} \times \text{ctx.head_num}.

        - **Output provided** (``output is not None``):

          - Take the format directly from ``output._format`` (must be
            ``PAGED`` or ``RAGGED``).
          - Validate that ``output`` has rank 3 and preserves the
            ``(N, D)`` dimensions of ``x``.
          - Validate device consistency between ``x`` and ``output``.

        Parameters
        ----------
        x : vTensor
            Input tensor with logical shape ``[B, N, D]``.

        output : Optional[vTensor]
            Optional preallocated output tensor. If ``None``, an internal
            RAGGED buffer is allocated; otherwise, this tensor must have
            shape ``[B_out, N, D]`` for some ``B_out`` and a format in
            ``{PAGED, RAGGED}``.

        loc : torch.Tensor
            Auxiliary tensor carrying per-position metadata required by
            the implementation (e.g., location/segment indices).

        ctx : Context
            Execution context that provides the runtime value of ``B``
            (via ``ctx.max_new_tokens_per_batch`` and ``ctx.head_num``)
            and is used for auxiliary memory accounting.

        Returns
        -------
        vTensor
            A :class:`vTensor` view representing the resolved output:
            either the provided ``output`` or an internally allocated
            buffer.

        Raises
        ------
        AssertionError
            If types, ranks, shapes, or devices are incompatible.
        """
        prefix = self._prefix()

        # --- type & rank checks ---
        assert isinstance(x, vTensor), f"{prefix}x must be vTensor, got {type(x)}"
        assert isinstance(loc, torch.Tensor), f"{prefix}loc must be torch.Tensor, got {type(loc)}"
        assert x.dim() == 3, f"{prefix}x must be 3D, got ndim={x.dim()} shape={tuple(x.shape)}"

        N, D = x.shape[1], x.shape[2]

        # Case A: output not provided -> allocate a RAGGED metadata buffer.
        if output is None:
            self.output_format = FORMAT.RAGGED

            B = ctx.max_new_tokens_per_batch * ctx.head_num
            self.output_buffer = vTensor(
                shape=(B, N, D),
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

        # Case B: output provided -> output_format follows output._format
        # (PAGED iff caller supplied a PAGED tensor; otherwise RAGGED).
        assert isinstance(output, vTensor), f"{prefix}output must be vTensor, got {type(output)}"
        assert output.dim() == 3, (
            f"{prefix}output must be 3D, "
            f"got ndim={output.dim()} shape={tuple(output.shape)}"
        )
        assert output._format in (FORMAT.PAGED, FORMAT.RAGGED), (
            f"{prefix}output._format must be PAGED or RAGGED, got {output._format}"
        )
        self.output_format = output._format

        # Shape consistency: unary elementwise keeps (N,D)
        assert output.shape[1] == N and output.shape[2] == D, (
            f"{prefix}output shape mismatch. Expected (*,{N},{D}), got {tuple(output.shape)}"
        )

        # Device consistency check
        assert x.device == output.device, (
            f"{prefix}x and output must be on the same device "
            f"(x.device={x.device}, output.device={output.device})"
        )

        # Register in the cache graph. ``output`` is provided by the caller
        # and assumed to already have a valid ``tensor_id`` and to live in
        # ``ctx.tensor_list``. We claim it as this op's produced tensor
        # (override producer), mirroring ``indexer.save_load.Save``.
        ctx.output_tensor_to_op_list[output.tensor_id] = len(ctx.op_list)
        ctx.op_list.append(self)
        ctx.op_to_input_tensor_list.append([x.tensor_id])
        ctx.op_to_output_tensor_list.append([output.tensor_id])

        return output


class Relu(Elementwise):
    r"""
    Piecewise ReLU-like activation.

    This operator applies, elementwise, the scalar function

    .. math::

        f(x; \alpha, \beta) =
        \begin{cases}
            x,      & x \ge \alpha, \\
            \beta,  & x < \alpha.
        \end{cases}

    Given an input tensor :math:`X \in \mathbb{R}^{B \times N \times D}`,
    the output is defined by

    .. math::

        Y[b, n, d] = f\bigl(X[b, n, d]; \alpha, \beta\bigr).

    Parameters
    ----------
    alpha : float, optional
        Threshold value :math:`\alpha`. Inputs greater than or equal to
        this threshold are passed through unchanged. Default is ``0.0``.

    beta : float, optional
        Fallback value :math:`\beta` used when :math:`x < \alpha`.
        Default is ``0.0``.
    """
    def __init__(self, alpha: float = 0.0, beta: float = 0.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseOpType.Relu


class Silu(Elementwise):
    r"""
    SiLU-like activation with configurable shift and slope.

    This operator applies, elementwise, the scalar function

    .. math::

        \operatorname{SiLU}(x; \alpha, \beta)
        = \frac{x}{1 + \exp(\beta x + \alpha)}.

    Given an input tensor :math:`X \in \mathbb{R}^{B \times N \times D}`,
    the output is

    .. math::

        Y[b, n, d]
        = \operatorname{SiLU}\bigl(X[b, n, d]; \alpha, \beta\bigr).

    Parameters
    ----------
    alpha : float, optional
        Bias term :math:`\alpha` added inside the exponential. Default is
        ``0.0``.

    beta : float, optional
        Slope :math:`\beta` multiplying :math:`x` inside the exponential.
        Default is ``0.0``.
    """
    def __init__(self, alpha: float = 0.0, beta: float = 0.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseOpType.Silu


class Sigmoid(Elementwise):
    r"""
    Sigmoid activation with configurable shift and slope.

    This operator applies, elementwise, the scalar function

    .. math::

        \sigma(x; \alpha, \beta)
        = \frac{1}{1 + \exp(\beta x + \alpha)}.

    Given an input tensor :math:`X \in \mathbb{R}^{B \times N \times D}`,
    the output is

    .. math::

        Y[b, n, d]
        = \sigma\bigl(X[b, n, d]; \alpha, \beta\bigr).

    Parameters
    ----------
    alpha : float, optional
        Bias term :math:`\alpha` added inside the exponential. Default is
        ``0.0``.

    beta : float, optional
        Slope :math:`\beta` multiplying :math:`x` inside the exponential.
        Default is ``0.0``.
    """
    def __init__(self, alpha: float = 0.0, beta: float = 0.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseOpType.Sigmoid


class Add_Mul(Elementwise):
    r"""
    Affine transformation :math:`y = \beta x + \alpha`.

    This operator applies, elementwise, the scalar function

    .. math::

        f(x; \alpha, \beta) = \beta x + \alpha.

    For an input tensor :math:`X \in \mathbb{R}^{B \times N \times D}`,
    the output is

    .. math::

        Y[b, n, d]
        = \beta \, X[b, n, d] + \alpha.

    Parameters
    ----------
    alpha : float, optional
        Additive term :math:`\alpha` in the affine transform. Default is
        ``0.0``.

    beta : float, optional
        Multiplicative term :math:`\beta` in the affine transform.
        Default is ``1.0``.
    """
    def __init__(self, alpha: float = 0.0, beta: float = 1.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseOpType.Add_Mul


class Abs(Elementwise):
    r"""
    Absolute-value transform of an affine argument.

    This operator applies, elementwise, the scalar function

    .. math::

        f(x; \alpha, \beta) = \bigl|\beta x + \alpha\bigr|.

    For an input tensor :math:`X \in \mathbb{R}^{B \times N \times D}`,
    the output is

    .. math::

        Y[b, n, d]
        = \bigl|\beta \, X[b, n, d] + \alpha\bigr|.

    Parameters
    ----------
    alpha : float, optional
        Additive term :math:`\alpha` inside the absolute value. Default is
        ``0.0``.

    beta : float, optional
        Multiplicative term :math:`\beta` inside the absolute value.
        Default is ``1.0``.
    """
    def __init__(self, alpha: float = 0.0, beta: float = 1.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseOpType.Abs


class Log(Elementwise):
    r"""
    Natural logarithm of an affine transform.

    This operator applies, elementwise, the scalar function

    .. math::

        f(x; \alpha, \beta) = \log(\beta x + \alpha).

    Given an input tensor :math:`X \in \mathbb{R}^{B \times N \times D}`,
    the output is

    .. math::

        Y[b, n, d] = \log(\beta X[b, n, d] + \alpha).

    Parameters
    ----------
    alpha : float, optional
        Additive bias term inside the logarithm. Default is ``0.0``.

    beta : float, optional
        Multiplicative scale term inside the logarithm. Default is ``1.0``.
    """
    def __init__(self, alpha: float = 0.0, beta: float = 1.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseOpType.Log


class Exp(Elementwise):
    r"""
    Exponential of an affine transform.

    This operator applies, elementwise, the scalar function

    .. math::

        f(x; \alpha, \beta) = \exp(\beta x + \alpha).

    Given an input tensor :math:`X \in \mathbb{R}^{B \times N \times D}`,
    the output is

    .. math::

        Y[b, n, d] = \exp(\beta X[b, n, d] + \alpha).

    Parameters
    ----------
    alpha : float, optional
        Additive bias term inside the exponential. Default is ``0.0``.

    beta : float, optional
        Multiplicative scale term inside the exponential. Default is ``1.0``.
    """
    def __init__(self, alpha: float = 0.0, beta: float = 1.0):
        super().__init__(alpha, beta)
        self.op_type = ElementwiseOpType.Exp
