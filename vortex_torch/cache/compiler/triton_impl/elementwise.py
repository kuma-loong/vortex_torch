"""Inline (Schedule.W) codegen for cache ``Elementwise`` ops.

Emits a single-line block computation referencing ``tensor_<in>_block``
(loaded as fp32 by the surrounding kernel) and writes ``tensor_<out>_block``
to be stored back as bf16 by the surrounding kernel.
"""

from ..graph import Graph
from ...context import Context
from ....utils import ElementwiseOpType
from ...elementwise import Elementwise


def generate_elementwise_impl(graph: Graph, op_id: int, ctx: Context) -> str:
    input_tensor_id = graph.op_to_input_tensor_list[op_id][0]
    output_tensor_id = graph.op_to_output_tensor_list[op_id][0]
    op = graph.op_list[op_id]
    assert issubclass(op.__class__, Elementwise), (
        f"Expected an Elementwise op, got {op.__class__.__name__}"
    )

    x = f"tensor_{input_tensor_id}_block"
    y = f"tensor_{output_tensor_id}_block"
    a = float(op.alpha)
    b = float(op.beta)

    if op.op_type == ElementwiseOpType.Relu:
        # piecewise: x >= alpha ? x : beta
        return f"{y} = tl.where({x} >= {a}, {x}, tl.full({x}.shape, {b}, dtype=tl.float32))"
    if op.op_type == ElementwiseOpType.Sigmoid:
        return f"{y} = 1.0 / (1.0 + tl.exp({b} * {x} + {a}))"
    if op.op_type == ElementwiseOpType.Silu:
        return f"{y} = {x} / (1.0 + tl.exp({b} * {x} + {a}))"
    if op.op_type == ElementwiseOpType.Abs:
        return f"{y} = tl.abs({b} * {x} + {a})"
    if op.op_type == ElementwiseOpType.Add_Mul:
        return f"{y} = {b} * {x} + {a}"
    if op.op_type == ElementwiseOpType.Log:
        return f"{y} = tl.log({b} * {x} + {a})"
    if op.op_type == ElementwiseOpType.Exp:
        return f"{y} = tl.exp({b} * {x} + {a})"

    raise NotImplementedError(
        f"Elementwise op_type {op.op_type!r} is not yet wired in cache codegen"
    )
