"""softmax — launcher emitter.

The kernel body lives in ``vortex_torch.custom_ops.softmax.<backend>.default.kernel``
(real ``@triton.jit`` functions, one per backend). The leaf's
``dispatch()`` returns a **plain Python callable** (Triton kernels are
wrapped via :func:`vortex_torch.custom_ops._triton_launcher.make_launcher`),
so the emitted call site is identical in shape to the CUDA-backed
leaves (no ``[grid]`` indexing, no ``num_warps`` / ``num_stages`` kwargs).
The trailing positional arg is ``eff_batch_size`` (the 1D grid size).
"""
from ..graph import Graph
from ...context import Context
from ....utils import INDENT
from ....abs import FORMAT
from ...scan import Softmax
from .backend import get_backend


def generate_softmax_impl(graph: Graph, op_id: int, ctx: Context) -> str:
    input_tensor_id = graph.op_to_input_tensor_list[op_id][0]
    output_tensor_id = graph.op_to_output_tensor_list[op_id][0]
    t_i = graph.tensor_list[input_tensor_id]
    t_o = graph.tensor_list[output_tensor_id]
    op = graph.op_list[op_id]
    assert issubclass(op.__class__, Softmax), f"Expected a softmax op, got {graph.op_list[op_id]}"
    assert t_i._format == FORMAT.RAGGED, f"Expected ragged input tensor for softmax, got {t_i._format}"
    assert t_o._format == FORMAT.RAGGED, f"Expected ragged output tensor for softmax, got {t_o._format}"

    bk = get_backend(ctx)
    callable_name = f"_vortex_softmax_kernel_{op_id}"
    ctx.compilation_header_lines.extend([
        "from vortex_torch.custom_ops import find as _vortex_custom_ops_find",
        f"{callable_name} = _vortex_custom_ops_find('softmax', '{bk.name}')()",
    ])

    # Backend-specific launcher tail:
    #   flashinfer: per-row = dense_kv_indptr, no extra constexpr args.
    #   trtllm    : per-row = dense_seqlens, plus block_size + max_blocks_per_seq.
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
        f"{INDENT*2}{per_row_launch},",
        f"{INDENT*2}{op.scale},",
        f"{INDENT*2}ctx.block_reserved_bos,",
        f"{INDENT*2}ctx.block_reserved_eos,",
        f"{INDENT*2}ctx.topk_val,",
        f"{INDENT*2}tensor_{input_tensor_id}.shape[-2],",
        f"{INDENT*2}tensor_{input_tensor_id}.shape[-1],",
        f"{extra_launch_block}{INDENT*2}eff_batch_size,",
        f"{INDENT})",
    ]
    return "\n".join(impl_lines)
