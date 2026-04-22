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
    output_tensor_id = graph.op_to_output_tensor_list[op_id]
    op = graph.op_list[op_id]
    assert issubclass(op.__class__, Fill), (
        f"Expected a Fill op, got {op.__class__.__name__}"
    )

    t_o = graph.tensor_list[output_tensor_id]
    z = f"tensor_{output_tensor_id}_block"
    a = float(op.alpha)

    # Emit an fp32 block of shape (D0, D1) filled with alpha.
    return f"{z} = tl.full(({t_o.shape[1]}, {t_o.shape[2]}), {a}, dtype=tl.float32)"
