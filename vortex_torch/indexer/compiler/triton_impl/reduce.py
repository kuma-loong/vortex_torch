from ..graph import Graph
from typing import Dict, Tuple, Callable
from ...context import Context
from ...reduce import Reduce
from ....utils import ReduceType
def generate_reduce_impl(graph: Graph, op_id: int, ctx: Context) -> str:
    
    input_tensor_id = graph.op_to_input_tensor_list[op_id][0]
    output_tensor_id = graph.op_to_output_tensor_list[op_id]
    op = graph.op_list[op_id]
    t_i = graph.tensor_list[input_tensor_id]
    t_o = graph.tensor_list[output_tensor_id]
    t_o_shape_str = ", ".join(map(str, t_o.shape[1:]))
    assert issubclass(op.__class__, Reduce), f"Expected a reduce op, got {graph.op_list[op_id]}"
    if op.reduce_type == ReduceType.Sum:
        impl_lines = [
            f"tensor_{output_tensor_id}_block = tl.sum(tensor_{input_tensor_id}_block, keep_dims=True, axis={op.dim})",
        ]
    elif op.reduce_type == ReduceType.Max:
        impl_lines = [
            f"tensor_{output_tensor_id}_block = tl.max(tensor_{input_tensor_id}_block, keep_dims=True, axis={op.dim})",
        ]
    elif op.reduce_type == ReduceType.Min:
        impl_lines = [
            f"tensor_{output_tensor_id}_block = tl.min(tensor_{input_tensor_id}_block, keep_dims=True, axis={op.dim})",
        ]
    elif op.reduce_type == ReduceType.L2Norm:
        impl_lines = [
            f"tensor_{output_tensor_id}_block = tl.sqrt(tl.sum(tensor_{input_tensor_id}_block * tensor_{input_tensor_id}_block, keep_dims=True, axis={op.dim}))",
        ]
    elif op.reduce_type == ReduceType.Mean:
        impl_lines = [
            f"tensor_{output_tensor_id}_block = tl.sum(tensor_{input_tensor_id}_block, keep_dims=True, axis={op.dim}) / {t_i.shape[op.dim]}",
        ]
    impl_str = "\n".join(impl_lines)
    return impl_str