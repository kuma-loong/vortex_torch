"""Inline (Schedule.W) codegen for cache ``GeMM``.

Computes ``O = Y @ X^T`` on per-block tiles. The surrounding kernel
already loads ``x`` and ``y`` as 2D fp32 blocks.

Per-block shapes:
  * ``y``: ``[y_D0, K]`` = ``[N_y, K]``
  * ``x``: ``[x_D0, K]`` = ``[N_x, K]``
  * ``o``: ``[y_D0, x_D0]`` = ``[N_y, N_x]``

Emits broadcast-then-sum — matches :mod:`vortex_torch.indexer.compiler.triton_impl.gemm`.
"""

from ..graph import Graph
from ...context import Context
from ...matmul import GeMM


def generate_gemm_impl(graph: Graph, op_id: int, ctx: Context) -> str:
    input_ids = graph.op_to_input_tensor_list[op_id]
    assert len(input_ids) == 2, f"GeMM expects 2 inputs, got {len(input_ids)}"
    x_id, y_id = input_ids
    out_id = graph.op_to_output_tensor_list[op_id][0]
    op = graph.op_list[op_id]
    assert issubclass(op.__class__, GeMM), (
        f"Expected a GeMM op, got {op.__class__.__name__}"
    )

    x = f"tensor_{x_id}_block"      # [N_x, K]
    y = f"tensor_{y_id}_block"      # [N_y, K]
    z = f"tensor_{out_id}_block"    # [N_y, N_x]

    # y[:, None, :] * x[None, :, :] -> [N_y, N_x, K], then sum over K (axis=2).
    return f"{z} = tl.sum({y}[:, None, :] * {x}[None, :, :], axis=2)"
