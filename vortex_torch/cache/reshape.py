import torch
from typing import Tuple, Dict, Optional
from .context import Context
from ..abs import vOp, vTensor, FORMAT
from ..utils import QuantizationType, Schedule


class Reshape(vOp):
    r"""
    Same-numel reshape of the inner two axes of a rank-3 cache block.

    Given a per-block tile

    .. math::

        X \in \mathbb{R}^{1 \times x_1 \times y_1},

    this op produces

    .. math::

        Y \in \mathbb{R}^{1 \times x_2 \times y_2},
        \qquad \text{with} \qquad x_2 \cdot y_2 \;=\; x_1 \cdot y_1,

    by interpreting the flat ``x_1 * y_1`` elements row-major into the
    new ``(x_2, y_2)`` layout. The leading (per-block batch) axis is
    fixed at size 1 on the cache side and must be passed as ``-1`` to
    the constructor as a placeholder, mirroring ``torch.reshape``'s
    "infer this dim" convention.

    No data movement happens beyond the existing cache load/store —
    the loaded ``(D0, D1)`` block is reshaped at the Triton-tile level
    via :func:`tl.reshape`. The kernel reads with the input's
    ``(x_1, y_1)`` shape and stores with the output's ``(x_2, y_2)``
    shape.

    Constructor
    -----------
    ``Reshape(-1, x2, y2)`` — three integers; the leading must be
    ``-1`` (the per-block batch dim is auto-inferred and always ``1``).
    The product ``x2 * y2`` is checked against the input's
    ``x1 * y1`` at :meth:`profile` time.

    Attributes
    ----------
    _impl_map : Dict[Tuple[FORMAT, FORMAT], FORMAT]
        Dispatch table keyed by ``(x_format, o_format)``.
    x2, y2 : int
        Target inner-axis sizes.
    output_format : Optional[FORMAT]
        Resolved output format from :meth:`profile`.
    output_buffer : Optional[vTensor]
        Pure-metadata vTensor descriptor for the output (graph node).
    """

    _impl_map: Dict[Tuple[FORMAT, FORMAT], FORMAT] = {
        (FORMAT.PAGED,  FORMAT.PAGED):  FORMAT.PAGED,
        (FORMAT.RAGGED, FORMAT.PAGED):  FORMAT.PAGED,
        (FORMAT.PAGED,  FORMAT.RAGGED): FORMAT.RAGGED,
        (FORMAT.RAGGED, FORMAT.RAGGED): FORMAT.RAGGED,
    }

    def __init__(self, batch_dim: int, x2: int, y2: int):
        super().__init__()
        cls = self.__class__.__name__
        assert batch_dim == -1, (
            f"{cls}.__init__: leading dim must be -1 (auto-infer; the "
            f"per-block batch axis is always 1 on the cache side), got "
            f"{batch_dim}"
        )
        assert isinstance(x2, int) and x2 >= 1, (
            f"{cls}.__init__: x2 must be a positive int, got {x2!r}"
        )
        assert isinstance(y2, int) and y2 >= 1, (
            f"{cls}.__init__: y2 must be a positive int, got {y2!r}"
        )
        self.x2 = x2
        self.y2 = y2

        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[vTensor] = None
        self.quantization_type: Optional[QuantizationType] = None
        # Cache reshape fuses into the per-block kernel — see
        # ``cache.compiler.triton_impl.kernel_gen``.
        self.schedule = Schedule.W

    # ------------------------------ helpers ------------------------------ #
    def _resolve_quantization(self, x: vTensor) -> QuantizationType:
        prefix = self._prefix()
        if x.dtype == torch.bfloat16:
            return QuantizationType.BF16
        if x.dtype == torch.float8_e5m2:
            return QuantizationType.FP8_E5M2
        if x.dtype == torch.float8_e4m3fn:
            return QuantizationType.FP8_E4M3
        raise ValueError(f"{prefix}unsupported dtype {x.dtype} for reshape")

    def _infer_output_format_ragged(self, x_fmt: FORMAT) -> FORMAT:
        key = (x_fmt, FORMAT.RAGGED)
        assert key in self._impl_map, (
            f"{self._prefix()}no RAGGED-output implementation for x_fmt={x_fmt}. "
            f"Available keys: {list(self._impl_map.keys())}"
        )
        return self._impl_map[key]

    # ---------------- profile ----------------
    def profile(
        self, x: vTensor, output: Optional[vTensor], loc: torch.Tensor, ctx: Context
    ) -> vTensor:
        r"""
        Validate, resolve formats, and register the reshape in the cache
        graph.

        Checks ``x.dim() == 3`` and the same-numel constraint
        ``x.shape[1] * x.shape[2] == self.x2 * self.y2``. If ``output``
        is supplied, also validates its shape matches
        ``[*, self.x2, self.y2]``.
        """
        prefix = self._prefix()

        assert isinstance(x, vTensor), f"{prefix}x must be vTensor, got {type(x)}"
        assert isinstance(loc, torch.Tensor), (
            f"{prefix}loc must be torch.Tensor, got {type(loc)}"
        )
        assert x.dim() == 3, (
            f"{prefix}x must be 3D [B, x1, y1], got ndim={x.dim()} shape={tuple(x.shape)}"
        )

        x1, y1 = x.shape[1], x.shape[2]
        in_numel = x1 * y1
        out_numel = self.x2 * self.y2
        assert in_numel == out_numel, (
            f"{prefix}same-numel reshape required: "
            f"x.shape[1]*x.shape[2] = {x1}*{y1} = {in_numel}  vs  "
            f"target x2*y2 = {self.x2}*{self.y2} = {out_numel}"
        )

        x_fmt = x._format

        # Case A: no preallocated output — allocate a RAGGED metadata buffer.
        if output is None:
            self.output_format = self._infer_output_format_ragged(x_fmt)
            self.quantization_type = self._resolve_quantization(x)

            B = ctx.max_new_tokens_per_batch * ctx.head_num
            self.output_buffer = vTensor(
                shape=(B, self.x2, self.y2),
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

        # Case B: caller-provided output — pick exact impl by (x_fmt, o_fmt).
        assert isinstance(output, vTensor), (
            f"{prefix}output must be vTensor, got {type(output)}"
        )
        assert output.dim() == 3, (
            f"{prefix}output must be 3D, got ndim={output.dim()} shape={tuple(output.shape)}"
        )
        assert output.shape[1] == self.x2 and output.shape[2] == self.y2, (
            f"{prefix}output inner shape must be ({self.x2}, {self.y2}), "
            f"got {tuple(output.shape)}"
        )

        o_fmt = output._format
        key = (x_fmt, o_fmt)
        assert key in self._impl_map, (
            f"{prefix}no implementation for (x_fmt={x_fmt}, o_fmt={o_fmt}). "
            f"Available keys: {list(self._impl_map.keys())}"
        )
        self.output_format = self._impl_map[key]

        assert x.device == output.device, (
            f"{prefix}x and output must be on the same device "
            f"(x.device={x.device}, output.device={output.device})"
        )

        self.quantization_type = self._resolve_quantization(x)

        # Register in the cache graph. Caller-provided ``output`` already has
        # a tensor_id in ``ctx.tensor_list``; we claim it as this op's product.
        ctx.output_tensor_to_op_list[output.tensor_id] = len(ctx.op_list)
        ctx.op_list.append(self)
        ctx.op_to_input_tensor_list.append([x.tensor_id])
        ctx.op_to_output_tensor_list.append([output.tensor_id])

        return output
