from ..graph import Graph
from typing import Dict, Tuple, Callable
from ...context import Context
from ...transpose import Transpose

def generate_transpose_impl(graph: Graph, op_id: int, ctx: Context) -> str:

    input_tensor_id = graph.op_to_input_tensor_list[op_id][0]
    output_tensor_id = graph.op_to_output_tensor_list[op_id]
    op = graph.op_list[op_id]
    assert issubclass(op.__class__, Transpose), f"Expected a transpose op, got {graph.op_list[op_id]}"

    impl_lines = [
        f"tensor_{output_tensor_id}_block = tl.trans(tensor_{input_tensor_id}_block, 0, 2, 1)",
    ]

    impl_str = "\n".join(impl_lines)
    return impl_str
