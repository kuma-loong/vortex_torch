"""Inline (Schedule.W) codegen for cache ``Fill``.

``Fill`` is modelled as a pure **producer** — its output tensor is the
target that gets overwritten with ``alpha``. The surrounding kernel
stores the produced block; no load is needed because Fill has no input
in the graph sense.
"""

from ..graph import Graph
from ...context import Context
from ...fill import Fill


def generate_fill_impl(graph: Graph, op_id: int, ctx: Context) -> str:
    output_tensor_id = graph.op_to_output_tensor_list[op_id][0]
    op = graph.op_list[op_id]
    assert issubclass(op.__class__, Fill), (
        f"Expected a Fill op, got {op.__class__.__name__}"
    )

    t_o = graph.tensor_list[output_tensor_id]
    z = f"tensor_{output_tensor_id}_block"
    a = float(op.alpha)

    # Emit an fp32 block of shape (D0, D1) filled with alpha.
    # ``tl.full`` block-shape constexprs must be pow2 → emit the padded
    # dims. Padded lanes hold the same value as real lanes; the
    # surrounding store masks out the trailing padded lanes so they
    # never land in memory.
    return (
        f"{z} = tl.full(({t_o.padded_shape[1]}, {t_o.padded_shape[2]}), "
        f"{a}, dtype=tl.float32)"
    )
