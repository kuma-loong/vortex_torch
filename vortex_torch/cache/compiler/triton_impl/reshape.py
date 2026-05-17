"""Inline (Schedule.W) codegen for cache ``Reshape``.

The surrounding kernel (see :mod:`.kernel_gen`) loads the input tile
as a 2D ``(x1, y1)`` block in fp32 and stores the output block. This
op's codegen just emits a same-numel ``tl.reshape`` from
``(x1, y1)`` → ``(x2, y2)``. The constraint
``x1 * y1 == x2 * y2`` is enforced at :meth:`Reshape.profile` time;
``tl.reshape`` will fail at compile time on a numel mismatch.

The reshape is row-major: element ``(i, j)`` in the input flat-indexes
to ``i * y1 + j``, which maps to ``(k, l)`` in the output where
``k = (i * y1 + j) // y2`` and ``l = (i * y1 + j) % y2``.
"""

from ..graph import Graph
from ...context import Context
from ...reshape import Reshape


def generate_reshape_impl(graph: Graph, op_id: int, ctx: Context) -> str:
    input_tensor_id = graph.op_to_input_tensor_list[op_id][0]
    output_tensor_id = graph.op_to_output_tensor_list[op_id][0]
    op = graph.op_list[op_id]
    assert issubclass(op.__class__, Reshape), (
        f"Expected a Reshape op, got {op.__class__.__name__}"
    )

    x = f"tensor_{input_tensor_id}_block"
    y = f"tensor_{output_tensor_id}_block"

    # Cache blocks are 2D `(x1, y1)`; reshape to `(x2, y2)`.
    return f"{y} = tl.reshape({x}, ({op.x2}, {op.y2}))"
