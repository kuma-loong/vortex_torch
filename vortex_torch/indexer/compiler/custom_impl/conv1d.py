"""conv1d — launcher emitter (Schedule.S, backend-agnostic).

Kernel body lives in ``vortex_torch.custom_ops.conv_1d.<backend>.default.kernel``.
The leaf's ``dispatch()`` returns a plain Python callable (Triton kernels
wrapped via :func:`vortex_torch.custom_ops._triton_launcher.make_launcher`),
so the emitted call site matches the CUDA-backed leaves' shape.
"""
from ..graph import Graph
from ...context import Context
from ....utils import INDENT
from ....abs import FORMAT
from ...scan import Conv1d
from ..backend import get_backend


def generate_conv1d_impl(graph: Graph, op_id: int, ctx: Context) -> str:
    input_tensor_id = graph.op_to_input_tensor_list[op_id][0]
    output_tensor_id = graph.op_to_output_tensor_list[op_id][0]
    t_i = graph.tensor_list[input_tensor_id]
    t_o = graph.tensor_list[output_tensor_id]
    op = graph.op_list[op_id]
    assert issubclass(op.__class__, Conv1d), f"Expected a conv1d op, got {graph.op_list[op_id]}"
    assert t_i._format == FORMAT.RAGGED, f"Expected ragged input tensor for conv1d, got {t_i._format}"
    assert t_o._format == FORMAT.RAGGED, f"Expected ragged output tensor for conv1d, got {t_o._format}"

    global_op_id = ctx.op_list.index(op)
    K = op.kernel_size

    bk = get_backend(ctx)
    callable_name = f"_vortex_conv1d_kernel_{op_id}"
    ctx.compilation_header_lines.extend([
        "from vortex_torch.custom_ops import find as _vortex_custom_ops_find",
        f"{callable_name} = _vortex_custom_ops_find('conv_1d', '{bk.name}')()",
    ])

    if bk.name == "trtllm":
        per_row_launch = "ctx.metadata.dense_seqlens"
        extra_launch_block = (
            f"{INDENT*2}ctx.block_size,\n"
            f"{INDENT*2}ctx.metadata.dense_block_tables.shape[1],\n"
        )
    else:
        per_row_launch = "ctx.metadata.dense_kv_indptr"
        extra_launch_block = ""

    impl_lines = [
        f"{INDENT}eff_batch_size = ctx.metadata.batch_size * ctx.num_kv_heads",
        f"",
        f"{INDENT}{callable_name}(",
        f"{INDENT*2}tensor_{input_tensor_id},",
        f"{INDENT*2}tensor_{output_tensor_id},",
        f"{INDENT*2}ctx.op_list[{global_op_id}].weight,",
        f"{INDENT*2}{per_row_launch},",
        f"{INDENT*2}ctx.block_reserved_bos,",
        f"{INDENT*2}ctx.block_reserved_eos,",
        f"{INDENT*2}ctx.topk_val,",
        f"{INDENT*2}{K},",
        f"{INDENT*2}{t_i.shape[1]},",
        f"{INDENT*2}{t_i.shape[2]},",
        f"{INDENT*2}{t_i.padded_shape[1]},",
        f"{INDENT*2}{t_i.padded_shape[2]},",
        f"{extra_launch_block}{INDENT*2}eff_batch_size,",
        f"{INDENT})",
    ]
    return "\n".join(impl_lines)
