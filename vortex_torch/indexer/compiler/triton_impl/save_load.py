from ..graph import Graph
from typing import Dict, Tuple, Callable
from ...context import Context
from ....abs import FORMAT
from ...save_load import Save, Load


def generate_save_impl(graph: Graph, op_id: int, ctx: Context) -> str:
    r"""
    Inline (Schedule.W) codegen for ``Save`` (RAGGED -> PAGED).

    The surrounding workload kernel already handles the per-format
    addressing: it loads the RAGGED input chunk into ``tensor_<x>_block``
    and, for the PAGED output, scatters via ``ctx.dense_kv_indices``
    using ``page_idx_i32 = indices[ragged_idx_i32]``. So the op itself
    just renames the block.
    """
    input_tensor_id = graph.op_to_input_tensor_list[op_id][0]
    output_tensor_id = graph.op_to_output_tensor_list[op_id]
    op = graph.op_list[op_id]
    assert issubclass(op.__class__, Save), f"Expected a save op, got {graph.op_list[op_id]}"

    t_i = graph.tensor_list[input_tensor_id]
    t_o = graph.tensor_list[output_tensor_id]
    assert t_i._format == FORMAT.RAGGED, f"Expected ragged input tensor for save, got {t_i._format}"
    assert t_o._format == FORMAT.PAGED, f"Expected paged output tensor for save, got {t_o._format}"

    return f"tensor_{output_tensor_id}_block = tensor_{input_tensor_id}_block"


def generate_load_impl(graph: Graph, op_id: int, ctx: Context) -> str:
    r"""
    Inline (Schedule.W) codegen for ``Load`` (PAGED -> RAGGED).

    The surrounding workload kernel gathers the PAGED input via
    ``page_idx_i32 = ctx.dense_kv_indices[ragged_idx_i32]`` into
    ``tensor_<x>_block`` and stores the RAGGED output contiguously,
    so this op only needs to rename the block.
    """
    input_tensor_id = graph.op_to_input_tensor_list[op_id][0]
    output_tensor_id = graph.op_to_output_tensor_list[op_id]
    op = graph.op_list[op_id]
    assert issubclass(op.__class__, Load), f"Expected a load op, got {graph.op_list[op_id]}"

    t_i = graph.tensor_list[input_tensor_id]
    t_o = graph.tensor_list[output_tensor_id]
    assert t_i._format == FORMAT.PAGED, f"Expected paged input tensor for load, got {t_i._format}"
    assert t_o._format == FORMAT.RAGGED, f"Expected ragged output tensor for load, got {t_o._format}"

    return f"tensor_{output_tensor_id}_block = tensor_{input_tensor_id}_block"
