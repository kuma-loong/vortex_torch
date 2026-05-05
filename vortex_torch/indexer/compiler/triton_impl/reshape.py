"""Inline (Schedule.W) codegen for indexer ``Reshape``.

Indexer per-workload tiles are loaded as 3D ``(W, x1, y1)`` blocks
(``W = ctx.workload_chunk_size`` for RAGGED/PAGED, ``W = 1`` for
BATCHED inputs that broadcast along the workload axis). Reshape
emits a single ``tl.reshape`` from ``(W, x1, y1)`` to ``(W, x2, y2)``
with ``x1 * y1 == x2 * y2`` (enforced at :meth:`Reshape.profile`
time; ``tl.reshape`` re-checks at compile time).

The reshape is row-major over the inner two axes — the leading
workload-axis position is preserved.
"""

from ..graph import Graph
from ...context import Context
from ....abs import FORMAT
from ...reshape import Reshape


def _block_leading_dim(t, ctx: Context) -> int:
    """Leading dim of a loaded block: 1 for BATCHED, W for everything else."""
    return 1 if t._format == FORMAT.BATCHED else ctx.workload_chunk_size


def generate_reshape_impl(graph: Graph, op_id: int, ctx: Context) -> str:
    input_tensor_id = graph.op_to_input_tensor_list[op_id][0]
    output_tensor_id = graph.op_to_output_tensor_list[op_id]
    op = graph.op_list[op_id]
    assert issubclass(op.__class__, Reshape), (
        f"Expected a Reshape op, got {op.__class__.__name__}"
    )

    t_x = graph.tensor_list[input_tensor_id]
    t_o = graph.tensor_list[output_tensor_id]

    x = f"tensor_{input_tensor_id}_block"
    y = f"tensor_{output_tensor_id}_block"

    Wx = _block_leading_dim(t_x, ctx)
    Wo = _block_leading_dim(t_o, ctx)
    # Same input/output format → same leading dim. Defensive assert anyway.
    assert Wx == Wo, (
        f"Reshape codegen: leading dim mismatch (input W={Wx}, output W={Wo}); "
        f"input format {t_x._format} -> output format {t_o._format}"
    )

    return f"{y} = tl.reshape({x}, ({Wo}, {op.x2}, {op.y2}))"
