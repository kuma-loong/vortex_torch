from ..graph import Graph
from typing import Dict, Tuple, Callable
from ...context import Context
from ...elementwise import Elementwise
from ....utils import ElementwiseOpType

def generate_elementwise_impl(graph: Graph, op_id: int, ctx: Context) -> str:

    input_tensor_id_0 = graph.op_to_input_tensor_list[op_id][0]
    output_tensor_id = graph.op_to_output_tensor_list[op_id][0]
    op = graph.op_list[op_id]
    t_i0 = graph.tensor_list[input_tensor_id_0]
    t_o = graph.tensor_list[output_tensor_id]
    t_o_shape_str = ", ".join(map(str, t_o.shape))
    assert issubclass(op.__class__, Elementwise), f"Expected an elementwise op, got {graph.op_list[op_id]}"
    alpha = op.alpha
    beta = op.beta

    if op.op_type == ElementwiseOpType.Relu:
        impl_lines = [
            f"tensor_{output_tensor_id}_block = tl.where(tensor_{input_tensor_id_0}_block >= {alpha}, tensor_{input_tensor_id_0}_block, {beta})",
        ]
    elif op.op_type == ElementwiseOpType.Sigmoid:
        impl_lines = [
            f"tensor_{output_tensor_id}_block = 1.0 / (1.0 + tl.exp({beta} * tensor_{input_tensor_id_0}_block + {alpha}))",
        ]
    elif op.op_type == ElementwiseOpType.Silu:
        impl_lines = [
            f"tensor_{output_tensor_id}_block = tensor_{input_tensor_id_0}_block / (1.0 + tl.exp({beta} * tensor_{input_tensor_id_0}_block + {alpha}))",
        ]
    elif op.op_type == ElementwiseOpType.Abs:
        impl_lines = [
            f"tensor_{output_tensor_id}_block = tl.abs({beta} * tensor_{input_tensor_id_0}_block + {alpha})",
        ]
    elif op.op_type == ElementwiseOpType.Add_Mul:
        impl_lines = [
            f"tensor_{output_tensor_id}_block = {beta} * tensor_{input_tensor_id_0}_block + {alpha}",
        ]
    elif op.op_type == ElementwiseOpType.Log:
        impl_lines = [
            f"tensor_{output_tensor_id}_block = tl.log({beta} * tensor_{input_tensor_id_0}_block + {alpha})",
        ]
    elif op.op_type == ElementwiseOpType.Exp:
        impl_lines = [
            f"tensor_{output_tensor_id}_block = tl.exp({beta} * tensor_{input_tensor_id_0}_block + {alpha})",
        ]

    impl_str = "\n".join(impl_lines)
    return impl_str
