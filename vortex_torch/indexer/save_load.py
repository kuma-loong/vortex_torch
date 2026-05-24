import torch
from typing import Dict, Optional
from .context import Context
from ..abs import vTensor, FORMAT, vOp
from ..utils import Schedule

class Save(vOp):
    r"""
    Persist a per-page indexer value across decode steps (paired with
    :class:`Load`) by copying it into a preallocated cache field.

    :Math:
        .. math::

            O \leftarrow X

        (a format/layout copy ``RAGGED`` → ``PAGED``; no arithmetic).
    :__init__: ``Save()`` — no arguments.
    :__call__: ``op(x, o, ctx=ctx)`` — ``x`` ``[S, D_0, D_1]`` (``RAGGED``) is
        written **in place** into the preallocated ``o`` (``PAGED``, matching
        ``D_0`` / ``D_1``). Returns nothing.
    :Note: write side of the persistent-state pattern; a flow that uses
        ``Save`` requires the engine to set ``disable_radix_cache=True``.
    """

    # Dispatch table keyed by x_format -> resolved output format.
    _impl_map: Dict[FORMAT, FORMAT] = {
        FORMAT.RAGGED: FORMAT.PAGED,
        # Add more entries if you support other formats.
    }

    def __init__(self):
        super().__init__()
        self.output_format: Optional[FORMAT] = None
        self.schedule = Schedule.W

    # ---------------- profile ----------------
    def profile(self, x: vTensor, o: vTensor, ctx: Context) -> vTensor:
        r"""Trace-time: validate ``x`` / ``o`` (rank-3, matching ``D_0`` /
        ``D_1``, compatible formats), register the op, and return ``o`` as the
        output view (no buffer is allocated)."""
        prefix = self._prefix()

        # Type & rank checks
        assert isinstance(x, vTensor), f"{prefix}profile expects x to be vTensor, got {type(x)}"
        assert isinstance(o, vTensor), f"{prefix}profile expects o to be vTensor, got {type(o)}"
        assert x.dim() == 3, f"{prefix}expected 3D x [S, D0, D1], got {tuple(x.shape)}"
        assert o.dim() == 3, f"{prefix}expected 3D o [S, D0, D1], got {tuple(o.shape)}"

        # Shape checks: D0/D1 must match (S may differ by layout; implementation handles it)
        assert x.shape[1] == o.shape[1], (
            f"{prefix}expected matching D0: x.shape[1]={x.shape[1]} vs o.shape[1]={o.shape[1]}"
        )
        assert x.shape[2] == o.shape[2], (
            f"{prefix}expected matching D1: x.shape[2]={x.shape[2]} vs o.shape[2]={o.shape[2]}"
        )

        # Dispatch by x format
        x_fmt = x._format
        assert x_fmt in self._impl_map, (
            f"{prefix}no implementation for x_fmt={x_fmt}. "
            f"Available keys: {list(self._impl_map.keys())}"
        )
        self.output_format = self._impl_map[x_fmt]

        # Output format must match the resolved format from dispatch
        assert o._format == self.output_format, (
            f"{prefix}output format mismatch. Expected {self.output_format}, got {o._format}"
        )

        # Device consistency
        assert x.device == o.device, (
            f"{prefix}x and o must be on the same device "
            f"(x.device={x.device}, o.device={o.device})"
        )

        # Save is a *side-effect writer*: the op stores back into a
        # caller-provided cache field. We intentionally do NOT claim
        # ``o.tensor_id`` as Save's producer in ``output_tensor_to_op_list``
        # — if Load elsewhere in the graph reads the same cache field, it
        # must see the previous-step value, not Save's updated value, and
        # overriding the producer slot would create a Load → Save cycle
        # through the DAG.
        #
        # Instead we register the op's id in ``side_effect_op_ids``; the
        # compiler seeds its op-DFS from that set so Save survives DCE,
        # and the target tensor is promoted to a final output so the
        # subgraph emits a ``tl.store`` for it.
        save_op_id = len(ctx.op_list)
        ctx.op_list.append(self)
        ctx.op_to_input_tensor_list.append([x.tensor_id])
        ctx.op_to_output_tensor_list.append([o.tensor_id])
        ctx.side_effect_op_ids.append(save_op_id)

        return o


class Load(vOp):
    r"""
    Read back a per-page value persisted by :class:`Save` (the read side of
    the cross-decode-step persistent-state pattern).

    :Math:
        .. math::

            Y \leftarrow X

        (a format/layout copy ``PAGED`` → ``RAGGED``; no arithmetic).
    :__init__: ``Load()`` — no arguments.
    :__call__: ``y = op(x, ctx=ctx)`` — ``x`` ``[S, D_0, D_1]`` (``PAGED``);
        returns a freshly-allocated ``RAGGED`` view of the same inner shape.
    :Note: read side of the persistent-state pattern (see :class:`Save`).
    """

    # Dispatch table keyed by x_format -> resolved output format.
    _impl_map: Dict[FORMAT, FORMAT] = {
        FORMAT.PAGED: FORMAT.RAGGED,
        # Add more entries if you support other formats.
    }

    def __init__(self):
        super().__init__()
        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[torch.Tensor] = None
        self.schedule = Schedule.W

    # ---------------- profile ----------------
    def profile(self, x: vTensor, ctx: Context) -> vTensor:
        r"""Trace-time: validate ``x`` ``[S, D_0, D_1]``, register the op, and
        return a freshly-allocated ``vTensor`` view of the loaded value (same
        inner shape, ``RAGGED``)."""
        prefix = self._prefix()

        # Type & rank checks
        assert isinstance(x, vTensor), f"{prefix}profile expects x to be vTensor, got {type(x)}"
        assert x.dim() == 3, f"{prefix}expected 3D x [S, D0, D1], got {tuple(x.shape)}"

        # Dispatch by x format
        x_fmt = x._format
        assert x_fmt in self._impl_map, (
            f"{prefix}no implementation for x_fmt={x_fmt}. "
            f"Available keys: {list(self._impl_map.keys())}"
        )
        self.output_format = self._impl_map[x_fmt]

        # Pure-metadata vTensor with a fresh tensor_id, mirroring the
        # Softmax pattern so the compiler graph can track the buffer.
        D0, D1 = x.shape[1], x.shape[2]
        self.output_buffer = vTensor(
            shape=(0, D0, D1),
            dtype=ctx.vortex_dtype,
            device=x.device,
            _format=self.output_format,
            tensor_id=len(ctx.tensor_list),
        )

        # Track in the context graph (same convention as Softmax / Conv1d).
        ctx.tensor_list.append(self.output_buffer)
        ctx.output_tensor_to_op_list.append(len(ctx.op_list))
        ctx.op_list.append(self)
        ctx.op_to_input_tensor_list.append([x.tensor_id])
        ctx.op_to_output_tensor_list.append([self.output_buffer.tensor_id])

        return self.output_buffer
