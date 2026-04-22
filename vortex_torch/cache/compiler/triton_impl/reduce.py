"""Inline (Schedule.W) codegen for cache ``Reduce`` ops.

The surrounding kernel (see :mod:`.kernel_gen`) already loads the input
as a 2D ``(D0, D1)`` block in fp32 and stores the output block.
This op's codegen just writes the reduction expression for the
appropriate axis.

Axis mapping:
  * Cache logical tensor is ``[B, D0, D1]`` with block-level programs
    reading the (D0, D1) slice of a single block.
  * ``op.dim == 1`` → reduce over the logical ``D0`` axis → block axis 0.
  * ``op.dim == 2`` → reduce over the logical ``D1`` axis → block axis 1.
"""

from ..graph import Graph
from ...context import Context
from ....utils import ReduceType
from ...reduce import Reduce


def generate_reduce_impl(graph: Graph, op_id: int, ctx: Context) -> str:
    input_tensor_id = graph.op_to_input_tensor_list[op_id][0]
    output_tensor_id = graph.op_to_output_tensor_list[op_id]
    op = graph.op_list[op_id]
    assert issubclass(op.__class__, Reduce), (
        f"Expected a Reduce op, got {op.__class__.__name__}"
    )

    t_i = graph.tensor_list[input_tensor_id]
    x = f"tensor_{input_tensor_id}_block"
    y = f"tensor_{output_tensor_id}_block"

    # Cache blocks are 2D (D0, D1); logical dim in [B, D0, D1] -> block axis.
    axis = op.dim - 1

    if op.reduce_type == ReduceType.Sum:
        return f"{y} = tl.sum({x}, keep_dims=True, axis={axis})"
    if op.reduce_type == ReduceType.Max:
        return f"{y} = tl.max({x}, keep_dims=True, axis={axis})"
    if op.reduce_type == ReduceType.Min:
        return f"{y} = tl.min({x}, keep_dims=True, axis={axis})"
    if op.reduce_type == ReduceType.L2Norm:
        # L2 (not RMS): sqrt(sum(x*x))
        return f"{y} = tl.sqrt(tl.sum({x} * {x}, keep_dims=True, axis={axis}))"
    if op.reduce_type == ReduceType.Mean:
        # Pre-divide by the reduced-axis length (constant at codegen time).
        inv_n = 1.0 / float(t_i.shape[op.dim])
        return f"{y} = tl.sum({x}, keep_dims=True, axis={axis}) * ({inv_n})"

    raise NotImplementedError(
        f"Reduce type {op.reduce_type!r} is not yet wired in cache codegen"
    )
