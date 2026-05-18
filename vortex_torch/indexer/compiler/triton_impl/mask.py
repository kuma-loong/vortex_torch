"""Inline (Schedule.W) codegen for indexer ``MaskSlice``.

Emits a position-dependent mask over one inner axis of the workload
block (shape ``[workload_chunk_size, D_0, D_1]``).

For ``dim == 1`` (i.e. :math:`D_0`):

    _idx = tl.arange(0, D_0)
    _mask = (_idx >= start) & (_idx < end)
    tensor_out_block = tl.where(_mask[None, :, None], alpha, beta)

For ``dim == 2`` (i.e. :math:`D_1`): ``_mask`` broadcasts as
``[None, None, :]``.

The input block is loaded by the surrounding kernel but not read — the
output is a pure function of position. The dependency edge is preserved
via the graph-level ``op_to_input_tensor_list`` entry so neighbour
fusion still works.
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

    # Workload blocks are 3D: [workload_chunk_size, padded_D_0, padded_D_1].
    # ``tl.arange`` constexpr must be pow2 → use the padded length. The
    # mask is purely positional so trailing padded lanes still hold
    # ``op.beta`` (or whatever the false branch is); store-side codegen
    # masks them out via the inner-dim padding mask in ``kernel_gen``.
    if op.dim == 1:
        size = t_i.padded_shape[1]
        broadcast = "[None, :, None]"
    else:  # op.dim == 2, enforced in profile
        size = t_i.padded_shape[2]
        broadcast = "[None, None, :]"

    idx = f"_mask_idx_{op_id}"
    mask = f"_mask_{op_id}"
    return "\n".join([
        f"{idx} = tl.arange(0, {size})",
        f"{mask} = ({idx} >= {op.start}) & ({idx} < {op.end})",
        f"{y} = tl.where({mask}{broadcast}, {op.alpha}, {op.beta})",
    ])
