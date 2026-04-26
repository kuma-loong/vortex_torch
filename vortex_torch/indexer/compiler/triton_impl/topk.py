from ..graph import Graph
from typing import Dict, Tuple, Callable
from ...context import Context
from ....abs import FORMAT
from ....utils import INDENT


def _check_topk_io(graph: Graph, op_id: int) -> tuple[int, int]:
    input_tensor_id  = graph.op_to_input_tensor_list[op_id][0]
    output_tensor_id = graph.op_to_output_tensor_list[op_id]
    t_i = graph.tensor_list[input_tensor_id]
    t_o = graph.tensor_list[output_tensor_id]
    assert t_i._format == FORMAT.RAGGED, f"Expected ragged input tensor for topk, got {t_i._format}"
    assert t_o._format == FORMAT.RAGGED, f"Expected ragged output tensor for topk, got {t_o._format}"
    return input_tensor_id, output_tensor_id


def generate_topk_impl(graph: Graph, op_id: int, ctx: Context) -> str:
    ctx.compilation_header_lines.append("from vortex_torch_C import topk_output, topk_output_v2")  # Import the custom C++ extension for topk
    input_tensor_id, output_tensor_id = _check_topk_io(graph, op_id)

    impl_lines = [
        f"topk_output(",
        f"{INDENT * 2}tensor_{input_tensor_id},",
        f"{INDENT * 2}ctx.dense_kv_indptr,",
        f"{INDENT * 2}ctx.sparse_kv_indptr,",
        f"{INDENT * 2}ctx.dense_kv_indices,",
        f"{INDENT * 2}tensor_{output_tensor_id},",
        f"{INDENT * 2}ctx.batch_size * ctx.num_kv_heads,",
        f"{INDENT * 2}ctx.block_reserved_bos,",
        f"{INDENT * 2}ctx.block_reserved_eos,",
        f"{INDENT * 2}ctx.max_num_blocks_per_request,",
        f")",
    ]
    return "\n".join(impl_lines)


def generate_approx_topk_impl(graph: Graph, op_id: int, ctx: Context) -> str:
    """Codegen for `approxTopK(tolerate_ratio=...)` — the adaptive 8-bit
    radix variant. Reads `tolerate_ratio` off the op instance and
    inlines it as a literal in the C call."""
    ctx.compilation_header_lines.append("from vortex_torch_C import approx_topk_output")
    input_tensor_id, output_tensor_id = _check_topk_io(graph, op_id)

    op = graph.op_list[op_id]
    # Defensive: profile() runs first and the registry already pinned
    # the op type via (approxTopK, Schedule.S). Still guard the attribute.
    tolerate_ratio = float(getattr(op, "tolerate_ratio", 0.0))

    impl_lines = [
        f"approx_topk_output(",
        f"{INDENT * 2}tensor_{input_tensor_id},"
        f"{INDENT * 2}ctx.dense_kv_indptr,",
        f"{INDENT * 2}ctx.sparse_kv_indptr,",
        f"{INDENT * 2}ctx.dense_kv_indices,",
        f"{INDENT * 2}tensor_{output_tensor_id},",
        f"{INDENT * 2}ctx.batch_size * ctx.num_kv_heads,",
        f"{INDENT * 2}ctx.block_reserved_bos,",
        f"{INDENT * 2}ctx.block_reserved_eos,",
        f"{INDENT * 2}ctx.max_num_blocks_per_request,",
        f"{INDENT * 2}{tolerate_ratio!r},",
        f")",
    ]
    return "\n".join(impl_lines)
