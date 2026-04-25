"""Inline (Schedule.W) codegen for cache ``Elementwise_Binary`` ops.

Surrounding kernel loads both inputs as fp32 blocks; this emits the
elementwise expression and writes the output block.
"""

from ..graph import Graph
from ...context import Context
from ....utils import ElementwiseBinaryOpType
from ...elementwise_binary import Elementwise_Binary


def generate_elementwise_binary_impl(graph: Graph, op_id: int, ctx: Context) -> str:
    input_ids = graph.op_to_input_tensor_list[op_id]
    assert len(input_ids) == 2, (
        f"Elementwise_Binary expects 2 inputs, got {len(input_ids)}"
    )
    input_tensor_id0, input_tensor_id1 = input_ids
    output_tensor_id = graph.op_to_output_tensor_list[op_id]
    op = graph.op_list[op_id]
    assert issubclass(op.__class__, Elementwise_Binary), (
        f"Expected an Elementwise_Binary op, got {op.__class__.__name__}"
    )

    x = f"tensor_{input_tensor_id0}_block"
    y = f"tensor_{input_tensor_id1}_block"
    z = f"tensor_{output_tensor_id}_block"
    a = float(op.alpha)
    b = float(op.beta)
    neg_inf = f"tl.full({x}.shape, -1e30, dtype=tl.float32)"

    if op.op_type == ElementwiseBinaryOpType.Maximum:
        return f"{z} = tl.maximum({x}, {y})"
    if op.op_type == ElementwiseBinaryOpType.Minimum:
        return f"{z} = tl.minimum({x}, {y})"
    if op.op_type == ElementwiseBinaryOpType.Add:
        # AXPBY: alpha * x + beta * y
        return f"{z} = {a} * {x} + {b} * {y}"
    if op.op_type == ElementwiseBinaryOpType.Mul:
        return f"{z} = {x} * {y}"
    if op.op_type == ElementwiseBinaryOpType.WhereEqual:
        return f"{z} = tl.where({x} == {y}, 0.0, {neg_inf})"
    if op.op_type == ElementwiseBinaryOpType.WhereNotEqual:
        return f"{z} = tl.where({x} != {y}, 0.0, {neg_inf})"
    if op.op_type == ElementwiseBinaryOpType.WhereGreater:
        return f"{z} = tl.where({x} > {y}, 0.0, {neg_inf})"
    if op.op_type == ElementwiseBinaryOpType.WhereGreaterEqual:
        return f"{z} = tl.where({x} >= {y}, 0.0, {neg_inf})"
    if op.op_type == ElementwiseBinaryOpType.WhereLess:
        return f"{z} = tl.where({x} < {y}, 0.0, {neg_inf})"
    if op.op_type == ElementwiseBinaryOpType.WhereLessEqual:
        return f"{z} = tl.where({x} <= {y}, 0.0, {neg_inf})"

    raise NotImplementedError(
        f"Elementwise_Binary op_type {op.op_type!r} is not yet wired in cache codegen"
    )
