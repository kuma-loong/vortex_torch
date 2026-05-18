"""Inline (Schedule.W) codegen for cache ``MaskSlice``.

Cache per-block computations operate on 2-D ``(D_0, D_1)`` blocks (one
block per fired program; see :mod:`cache.compiler.triton_impl.kernel_gen`).
For a target axis :attr:`dim`:

  * ``dim == 1`` → block axis 0 (:math:`D_0`); mask broadcasts as ``[:, None]``.
  * ``dim == 2`` → block axis 1 (:math:`D_1`); mask broadcasts as ``[None, :]``.

Emitted snippet:

    _idx = tl.arange(0, size)
    _mask = (_idx >= start) & (_idx < end)
    tensor_out_block = tl.where(_mask[broadcast], alpha, beta)

The input block is loaded by the surrounding kernel but its values are
not consumed — ``MaskSlice`` is a pure position-based writer.
"""

from ..graph import Graph
from ...context import Context
from ...mask import MaskSlice


def generate_mask_slice_impl(graph: Graph, op_id: int, ctx: Context) -> str:
    input_tensor_id = graph.op_to_input_tensor_list[op_id][0]
    output_tensor_id = graph.op_to_output_tensor_list[op_id][0]
    op = graph.op_list[op_id]
    assert issubclass(op.__class__, MaskSlice), (
        f"Expected MaskSlice, got {op.__class__.__name__}"
    )

    t_i = graph.tensor_list[input_tensor_id]
    y = f"tensor_{output_tensor_id}_block"

    # Cache blocks are 2D: [padded_D_0, padded_D_1]. Map logical dim ->
    # block axis. ``tl.arange`` constexpr must be pow2 → use padded
    # length; the surrounding store masks out the trailing padded lanes.
    if op.dim == 1:
        size = t_i.padded_shape[1]
        broadcast = "[:, None]"
    else:  # op.dim == 2, enforced in profile
        size = t_i.padded_shape[2]
        broadcast = "[None, :]"

    idx = f"_mask_idx_{op_id}"
    mask = f"_mask_{op_id}"
    return "\n".join([
        f"{idx} = tl.arange(0, {size})",
        f"{mask} = ({idx} >= {op.start}) & ({idx} < {op.end})",
        f"{y} = tl.where({mask}{broadcast}, {op.alpha}, {op.beta})",
    ])
