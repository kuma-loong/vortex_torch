import torch
from typing import Dict, Optional
from .context import Context
from ..abs import vTensor, FORMAT, vOp
from ..utils import Schedule

class Softmax(vOp):
    r"""
    Segmented scaled softmax over the packed sequence axis.

    :Math:
        The leading axis is a packed concatenation of :math:`B` per-request
        segments :math:`\mathcal{I}_b` (total length
        :math:`S = \sum_b S_b`). For each segment and each channel
        :math:`(d_0,d_1)`, softmax runs **within** the segment:

        .. math::

            Y_{s,d_0,d_1} = \frac{\exp(\text{scale}\cdot X_{s,d_0,d_1})}
                 {\sum_{s'\in\mathcal{I}_b}\exp(\text{scale}\cdot X_{s',d_0,d_1})},
            \qquad s\in\mathcal{I}_b.
    :__init__: ``Softmax(dim=0, scale=1.0)`` — ``dim`` must be ``0`` (the packed
        S axis); ``scale`` multiplies the logits before the exponential.
    :__call__: ``y = op(x, ctx=ctx)`` — ``x`` ``[S, D_0, D_1]`` → same shape.
    :Note: ``RAGGED`` only.
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
        r"""Trace-time: validate ``x`` ``[S, D_0, D_1]``, dispatch on
        ``x._format``, register the op, and return the output ``vTensor``
        view (segmented softmax over ``dim=0``)."""
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
    Segmented :math:`L_2` normalization over the packed sequence axis.

    :Math:
        The leading axis is a packed concatenation of :math:`B` per-request
        segments :math:`\mathcal{I}_b`. For each segment and each channel
        :math:`(d_0,d_1)`, the values are divided by their segment-local
        :math:`L_2` norm:

        .. math::

            Y_{s,d_0,d_1} = \frac{X_{s,d_0,d_1}}
                 {\sqrt{\sum_{s'\in\mathcal{I}_b} X_{s',d_0,d_1}^2}},
            \qquad s\in\mathcal{I}_b.
    :__init__: ``Normalize(dim=0)`` — ``dim`` must be ``0`` (the packed S axis).
    :__call__: ``y = op(x, ctx=ctx)`` — ``x`` ``[S, D_0, D_1]`` → same shape.
    :Note: ``RAGGED`` only.
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
        r"""Trace-time: validate ``x`` ``[S, D_0, D_1]``, dispatch on
        ``x._format``, register the op, and return the output ``vTensor``
        view (segmented :math:`L_2` normalization over ``dim=0``)."""
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
    Segmented depth-wise causal 1-D convolution over the packed sequence axis.

    :Math:
        The leading axis is a packed concatenation of :math:`B` per-request
        segments. Within each segment, every channel :math:`(d_0,d_1)` is
        convolved by its own :math:`K`-tap causal filter
        :math:`W\in\mathbb{R}^{K\times D_0\times D_1}`:

        .. math::

            Y_{s,d_0,d_1} = \sum_{k=0}^{K-1} W_{k,d_0,d_1}\, X_{s-k,\,d_0,d_1},

        with :math:`X_{s-k}=0` for :math:`s-k` before the segment. The op
        runs only on the mid-range :math:`[b_{\text{bos}},\,S_b-b_{\text{eos}})`
        (``ctx.block_reserved_bos`` / ``block_reserved_eos``); reserved
        BOS/EOS rows are neither read nor written.
    :__init__: ``Conv1d(weight, dim=0, dtype=torch.bfloat16, device=None)`` —
        ``weight`` is a nested Python list of shape ``[K, D_0, D_1]`` (kernel
        size :math:`K` = ``len(weight)``); ``dim`` must be ``0``.
    :__call__: ``y = op(x, ctx=ctx)`` — ``x`` ``[S, D_0, D_1]`` → same shape;
        ``weight``'s inner dims must match ``(D_0, D_1)``.
    :Note: ``RAGGED`` only.
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
        r"""Trace-time: validate ``x`` ``[S, D_0, D_1]`` (weight inner dims
        must match), migrate ``weight`` to ``x``'s device, register the op,
        and return the output ``vTensor`` view."""
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

        # ``__init__`` materializes ``self.weight`` on whatever device the
        # user passed (defaulting to CPU when none was given), but the
        # generated Triton kernel dereferences ``weight`` from the GPU.
        # Migrate here — ``profile`` is the first call where the
        # inference device is known. ``.to`` is a no-op if already on the
        # right device, so it's safe to call on every re-profile.
        if self.weight.device != x.device:
            self.weight = self.weight.to(device=x.device)
        if not self.weight.is_contiguous():
            self.weight = self.weight.contiguous()

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
