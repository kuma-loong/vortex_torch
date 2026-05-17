from ..graph import Graph
from ...context import Context
from ...reduce import Reduce
from ....abs import FORMAT
from ....utils import INDENT, ReduceType, Schedule
from .backend import get_backend


def generate_reduce_impl(graph: Graph, op_id: int, ctx: Context) -> str:
    """Schedule.W reduce — inline ``tl.sum`` / ``tl.max`` / etc.

    Operates on the per-workload block ``tensor_<id>_block`` of shape
    ``[workload_chunk_size, D_0, D_1]``; ``op.dim`` is in ``{1, 2}``.
    """
    input_tensor_id = graph.op_to_input_tensor_list[op_id][0]
    output_tensor_id = graph.op_to_output_tensor_list[op_id][0]
    op = graph.op_list[op_id]
    t_i = graph.tensor_list[input_tensor_id]
    assert issubclass(op.__class__, Reduce), f"Expected a reduce op, got {graph.op_list[op_id]}"
    assert op.dim in (1, 2), (
        f"generate_reduce_impl handles dim in {{1, 2}}; got dim={op.dim}. "
        f"dim=0 is Schedule.S — see ``generate_reduce_dim0_impl``."
    )
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
            f"tensor_{output_tensor_id}_block = tl.sqrt(tl.sum((tensor_{input_tensor_id}_block * tensor_{input_tensor_id}_block).to(tl.float32), keep_dims=True, axis={op.dim}))",
        ]
    elif op.reduce_type == ReduceType.Mean:
        impl_lines = [
            f"tensor_{output_tensor_id}_block = tl.sum(tensor_{input_tensor_id}_block, keep_dims=True, axis={op.dim}) * ({1.0 / t_i.shape[op.dim]})",
        ]
    impl_str = "\n".join(impl_lines)
    return impl_str


# ---------------------------------------------------------------------------
# Schedule.S: cross-row reduce over the packed RAGGED leading axis.
#
# One Triton program per (batch, kv_head). Each program walks the
# ``[start, end)`` segment of *blocks* (block_size tokens each) described
# by the backend-specific per-row source (``dense_kv_indptr`` in CSR mode,
# ``dense_seqlens`` + ``block_size`` in trtllm mode), applies the
# reduction over the block axis, and writes a single ``[D_0, D_1]`` tile
# into the BATCHED output at index ``pid``.
#
# bos/eos mirror the softmax convention: the first ``bos`` and last ``eos``
# blocks of each segment are excluded (they're typically always-on tokens).
# Set them to ``0`` if you want the whole segment.
#
# The kernel body lives in ``vortex_torch.custom_ops.reduce_dim0.<backend>.
# <reduce_type>.<kernel>`` (one specialised ``@triton.jit`` per reduce_type
# × backend). Each leaf's ``dispatch()`` returns a plain Python callable
# (Triton kernels wrapped via
# :func:`vortex_torch.custom_ops._triton_launcher.make_launcher`), so the
# call site is identical in shape to the CUDA-backed leaves' — no
# ``[grid]`` indexing, no ``num_warps`` / ``num_stages`` kwargs;
# ``eff_batch_size`` is the trailing positional arg.
# ---------------------------------------------------------------------------

def generate_reduce_dim0_impl(graph: Graph, op_id: int, ctx: Context) -> str:
    """Schedule.S reduce — emit a launcher that resolves the per-reduce_type
    kernel via :func:`vortex_torch.custom_ops.find` at runtime."""
    input_tensor_id = graph.op_to_input_tensor_list[op_id][0]
    output_tensor_id = graph.op_to_output_tensor_list[op_id][0]
    op = graph.op_list[op_id]
    t_i = graph.tensor_list[input_tensor_id]
    t_o = graph.tensor_list[output_tensor_id]

    assert issubclass(op.__class__, Reduce), f"Expected a reduce op, got {op}"
    assert op.dim == 0, f"generate_reduce_dim0_impl: expected dim=0, got dim={op.dim}"
    assert op.schedule == Schedule.S, (
        f"generate_reduce_dim0_impl: expected Schedule.S, got {op.schedule}"
    )
    assert t_i._format == FORMAT.RAGGED, (
        f"generate_reduce_dim0_impl: input must be RAGGED, got {t_i._format}"
    )
    assert t_o._format == FORMAT.BATCHED, (
        f"generate_reduce_dim0_impl: output must be BATCHED, got {t_o._format}"
    )

    rt_name = op.reduce_type.name.lower()
    bk = get_backend(ctx)
    callable_name = f"_vortex_reduce_dim0_{rt_name}_kernel_{op_id}"
    ctx.compilation_header_lines.extend([
        "from vortex_torch.custom_ops import find as _vortex_custom_ops_find",
        f"{callable_name} = _vortex_custom_ops_find("
        f"'reduce_dim0', '{bk.name}', variant='{rt_name}')()",
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
        f"{INDENT}{callable_name}(",
        f"{INDENT*2}tensor_{input_tensor_id},",
        f"{INDENT*2}tensor_{output_tensor_id},",
        f"{INDENT*2}{per_row_launch},",
        f"{INDENT*2}ctx.block_reserved_bos,",
        f"{INDENT*2}ctx.block_reserved_eos,",
        f"{INDENT*2}tensor_{input_tensor_id}.shape[-2],",
        f"{INDENT*2}tensor_{input_tensor_id}.shape[-1],",
        f"{extra_launch_block}{INDENT*2}eff_batch_size,",
        f"{INDENT})",
    ]
    return "\n".join(impl_lines)