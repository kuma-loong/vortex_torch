import torch
from typing import Optional
from .context import Context
from ..abs import vTensor, FORMAT, vOp
from ..utils import ReduceType, Schedule

class Reduce(vOp):
    r"""
    Generic 1-D reduction over one axis of a rank-3 logical tensor.

    :Math:
        For input :math:`X\in\mathbb{R}^{N\times D_0\times D_1}` and a
        reduction :math:`\rho` (mean / max / min / L2-norm / sum, fixed by the
        subclass): ``dim=1`` →
        :math:`\text{out}[n,0,d_1]=\rho_{d_0}\,X[n,d_0,d_1]` (shape
        ``[N, 1, D_1]``); ``dim=2`` →
        :math:`\text{out}[n,d_0,0]=\rho_{d_1}\,X[n,d_0,d_1]` (shape
        ``[N, D_0, 1]``); ``dim=0`` collapses the packed leading axis to one
        row per ``(batch, kv_head)``.
    :__init__:
        ``Reduce(dim=1)`` — logical axis to reduce, one of ``0`` / ``1`` / ``2``.
    :__call__:
        ``y = op(x, ctx=ctx)`` — ``x`` is ``[N, D_0, D_1]``; the reduced axis is
        kept with size 1. For ``dim ∈ {1, 2}`` the output is ``BATCHED`` iff the
        input is; ``dim=0`` requires ``RAGGED`` input and returns ``BATCHED``.
    :Note:
        Use a concrete subclass — :class:`Max`, :class:`Min`, :class:`Mean`,
        :class:`L2Norm`, :class:`Sum`.
    """

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
        r"""Trace-time: validate ``x`` (``[N, D_0, D_1]``), resolve the output
        format, register the op, and return a ``vTensor`` view of the reduced
        output (see the class docstring for shapes)."""
        prefix = self._prefix()

        # Type & rank checks
        assert isinstance(x, vTensor), f"{prefix}profile expects x to be vTensor, got {type(x)}"
        assert x.dim() == 3, (
            f"{prefix}expected 3D input [N, D0, D1], "
            f"got ndim={x.dim()} shape={tuple(x.shape)}"
        )

        D0, D1 = x.shape[1], x.shape[2]

        if self.dim == 0:
            # Cross-row reduction: collapse the packed leading axis into one
            # summary per (batch, kv_head). Input must be RAGGED (per-page or
            # per-token); the compiler allocates a BATCHED buffer with leading
            # dim ``ctx.max_bs * ctx.num_kv_heads`` (see indexer interface).
            assert x._format == FORMAT.RAGGED, (
                f"{prefix}dim=0 reduce requires RAGGED input, got {x._format}"
            )
            self.output_format = FORMAT.BATCHED
            out_D0, out_D1 = D0, D1
        else:
            # Output is BATCHED iff the input is BATCHED; otherwise RAGGED.
            self.output_format = (
                FORMAT.BATCHED if x._format == FORMAT.BATCHED else FORMAT.RAGGED
            )
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
    Max reduction over one logical axis (a :class:`Reduce`).

    :Math: ``dim=1``: :math:`\text{out}[n,0,d_1]=\max_{d_0}X[n,d_0,d_1]`;
        ``dim=2``: :math:`\text{out}[n,d_0,0]=\max_{d_1}X[n,d_0,d_1]`.
    :__init__: ``Max(dim=1)`` — axis to reduce (``1`` → :math:`D_0`,
        ``2`` → :math:`D_1`).
    :__call__: ``y = op(x, ctx=ctx)`` — ``[N, D_0, D_1]`` → ``[N, 1, D_1]``
        (``dim=1``) or ``[N, D_0, 1]`` (``dim=2``).
    """
    def __init__(self, dim: int = 1):
        super().__init__(dim)
        self.reduce_type = ReduceType.Max


class Min(Reduce):
    r"""
    Min reduction over one logical axis (a :class:`Reduce`).

    :Math: ``dim=1``: :math:`\text{out}[n,0,d_1]=\min_{d_0}X[n,d_0,d_1]`;
        ``dim=2``: :math:`\text{out}[n,d_0,0]=\min_{d_1}X[n,d_0,d_1]`.
    :__init__: ``Min(dim=1)`` — axis to reduce (``1`` → :math:`D_0`,
        ``2`` → :math:`D_1`).
    :__call__: ``y = op(x, ctx=ctx)`` — ``[N, D_0, D_1]`` → ``[N, 1, D_1]``
        (``dim=1``) or ``[N, D_0, 1]`` (``dim=2``).
    """
    def __init__(self, dim: int = 1):
        super().__init__(dim)
        self.reduce_type = ReduceType.Min


class Mean(Reduce):
    r"""
    Mean reduction over one logical axis (a :class:`Reduce`).

    :Math: ``dim=1``: :math:`\text{out}[n,0,d_1]=\frac{1}{D_0}\sum_{d_0}X[n,d_0,d_1]`;
        ``dim=2``: :math:`\text{out}[n,d_0,0]=\frac{1}{D_1}\sum_{d_1}X[n,d_0,d_1]`.
    :__init__: ``Mean(dim=1)`` — axis to reduce (``1`` → :math:`D_0`,
        ``2`` → :math:`D_1`).
    :__call__: ``y = op(x, ctx=ctx)`` — ``[N, D_0, D_1]`` → ``[N, 1, D_1]``
        (``dim=1``) or ``[N, D_0, 1]`` (``dim=2``).
    """
    def __init__(self, dim: int = 1):
        super().__init__(dim)
        self.reduce_type = ReduceType.Mean


class L2Norm(Reduce):
    r"""
    L2-norm reduction over one logical axis (a :class:`Reduce`).

    :Math: ``dim=1``: :math:`\text{out}[n,0,d_1]=\sqrt{\sum_{d_0}X[n,d_0,d_1]^2}`;
        ``dim=2``: :math:`\text{out}[n,d_0,0]=\sqrt{\sum_{d_1}X[n,d_0,d_1]^2}`.
    :__init__: ``L2Norm(dim=1)`` — axis to reduce (``1`` → :math:`D_0`,
        ``2`` → :math:`D_1`).
    :__call__: ``y = op(x, ctx=ctx)`` — ``[N, D_0, D_1]`` → ``[N, 1, D_1]``
        (``dim=1``) or ``[N, D_0, 1]`` (``dim=2``).
    """
    def __init__(self, dim: int = 1):
        super().__init__(dim)
        self.reduce_type = ReduceType.L2Norm


class Sum(Reduce):
    r"""
    Sum reduction over one logical axis (a :class:`Reduce`).

    :Math: ``dim=1``: :math:`\text{out}[n,0,d_1]=\sum_{d_0}X[n,d_0,d_1]`;
        ``dim=2``: :math:`\text{out}[n,d_0,0]=\sum_{d_1}X[n,d_0,d_1]`.
    :__init__: ``Sum(dim=1)`` — axis to reduce (``1`` → :math:`D_0`,
        ``2`` → :math:`D_1`).
    :__call__: ``y = op(x, ctx=ctx)`` — ``[N, D_0, D_1]`` → ``[N, 1, D_1]``
        (``dim=1``) or ``[N, D_0, 1]`` (``dim=2``).
    """
    def __init__(self, dim: int = 1):
        super().__init__(dim)
        self.reduce_type = ReduceType.Sum
