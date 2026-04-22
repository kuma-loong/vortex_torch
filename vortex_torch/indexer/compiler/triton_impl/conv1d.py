from ..graph import Graph
from typing import Dict, Tuple, Callable
from ...context import Context
from ....utils import INDENT
from ....abs import FORMAT
from ...scan import Conv1d
def generate_conv1d_impl(graph: Graph, op_id: int, ctx: Context) -> str:

    input_tensor_id = graph.op_to_input_tensor_list[op_id][0]
    output_tensor_id = graph.op_to_output_tensor_list[op_id]
    t_i = graph.tensor_list[input_tensor_id]
    t_o = graph.tensor_list[output_tensor_id]
    op = graph.op_list[op_id]
    assert issubclass(op.__class__, Conv1d), f"Expected a conv1d op, got {graph.op_list[op_id]}"
    assert t_i._format == FORMAT.RAGGED, f"Expected ragged input tensor for conv1d, got {t_i._format}"
    assert t_o._format == FORMAT.RAGGED, f"Expected ragged output tensor for conv1d, got {t_o._format}"

    global_op_id = ctx.op_list.index(op)
    K = op.kernel_size

    func_def_lines = [
    f"@triton.jit",
    f"def conv1d_kernel(",
    f"{INDENT}x,",
    f"{INDENT}out,",
    f"{INDENT}weight,",
    f"{INDENT}indptr,",
    f"{INDENT}bos: tl.constexpr,",
    f"{INDENT}eos: tl.constexpr,",
    f"{INDENT}topk_val: tl.constexpr,",
    f"{INDENT}K: tl.constexpr,",
    f"{INDENT}x_D0: tl.constexpr,",
    f"{INDENT}x_D1: tl.constexpr,",
    f"{INDENT}BLOCK_P: tl.constexpr = 128,",
    f"):",
    f"{INDENT}pid = tl.program_id(0)",
    f"",
    f"{INDENT}start = tl.load(indptr + pid)",
    f"{INDENT}end = tl.load(indptr + pid + 1)",
    f"{INDENT}num_pages_this_seq = end - start",
    f"",
    f"{INDENT}threshold: tl.constexpr = bos + eos + topk_val",
    f"{INDENT}if num_pages_this_seq <= threshold:",
    f"{INDENT*2}return",
    f"",
    f"{INDENT}num_pages_to_compute = num_pages_this_seq - bos - eos",
    f"{INDENT}if num_pages_to_compute <= 0:",
    f"{INDENT*2}return",
    f"",
    f"{INDENT}page_stride = x_D0 * x_D1",
    f"",
    f"{INDENT}x_base_ptr = x + (start + bos) * page_stride",
    f"{INDENT}out_base_ptr = out + (start + bos) * page_stride",
    f"",
    f"{INDENT}d0_idx = tl.arange(0, x_D0)",
    f"{INDENT}d1_idx = tl.arange(0, x_D1)",
    f"{INDENT}p_idx = tl.arange(0, BLOCK_P)",
    f"",
    f"{INDENT}for p in range(0, num_pages_to_compute, BLOCK_P):",
    f"{INDENT*2}kp = tl.minimum(BLOCK_P, num_pages_to_compute - p)",
    f"{INDENT*2}p_mask = p_idx < kp",
    f"",
    f"{INDENT*2}acc = tl.zeros((BLOCK_P, x_D0, x_D1), dtype=tl.float32)",
    f"",
    f"{INDENT*2}for k in tl.static_range(K):",
    f"{INDENT*3}in_pos = p + p_idx - k",
    f"{INDENT*3}in_valid = (in_pos >= 0) & p_mask",
    f"{INDENT*3}in_pos_clamped = tl.maximum(in_pos, 0)",
    f"",
    f"{INDENT*3}x_offs = (",
    f"{INDENT*4}in_pos_clamped[:, None, None] * page_stride",
    f"{INDENT*4}+ d0_idx[None, :, None] * x_D1",
    f"{INDENT*4}+ d1_idx[None, None, :]",
    f"{INDENT*3}).to(tl.int32)",
    f"",
    f"{INDENT*3}x_val = tl.load(x_base_ptr + x_offs, mask=in_valid[:, None, None], other=0.0).to(tl.float32)",
    f"",
    f"{INDENT*3}w_offs = (",
    f"{INDENT*4}k * page_stride",
    f"{INDENT*4}+ d0_idx[:, None] * x_D1",
    f"{INDENT*4}+ d1_idx[None, :]",
    f"{INDENT*3}).to(tl.int32)",
    f"{INDENT*3}w_val = tl.load(weight + w_offs).to(tl.float32)",
    f"",
    f"{INDENT*3}acc = acc + x_val * w_val[None, :, :]",
    f"",
    f"{INDENT*2}out_offs = (",
    f"{INDENT*3}(p + p_idx)[:, None, None] * page_stride",
    f"{INDENT*3}+ d0_idx[None, :, None] * x_D1",
    f"{INDENT*3}+ d1_idx[None, None, :]",
    f"{INDENT*2}).to(tl.int32)",
    f"",
    f"{INDENT*2}tl.store(out_base_ptr + out_offs, acc.to(tl.bfloat16), mask=p_mask[:, None, None])",
    ]
    ctx.auxilary_func_def_lines.extend(func_def_lines)

    impl_lines = [
    f"{INDENT}eff_batch_size = ctx.batch_size * ctx.num_kv_heads",
    f"",
    f"{INDENT}conv1d_kernel[(eff_batch_size,)](",
    f"{INDENT*2}tensor_{input_tensor_id},",
    f"{INDENT*2}tensor_{output_tensor_id},",
    f"{INDENT*2}ctx.op_list[{global_op_id}].weight,",
    f"{INDENT*2}ctx.dense_kv_indptr,",
    f"{INDENT*2}ctx.block_reserved_bos,",
    f"{INDENT*2}ctx.block_reserved_eos,",
    f"{INDENT*2}ctx.topk_val,",
    f"{INDENT*2}{K},",
    f"{INDENT*2}tensor_{input_tensor_id}.shape[-2],",
    f"{INDENT*2}tensor_{input_tensor_id}.shape[-1],",
    f"{INDENT*2}num_warps=4,",
    f"{INDENT*2}num_stages=1,",
    f"{INDENT})",
    ]

    return "\n".join(impl_lines)
