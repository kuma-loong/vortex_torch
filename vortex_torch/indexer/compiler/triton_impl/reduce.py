from ..graph import Graph
from typing import Dict, Tuple, Callable
from ...context import Context
from ...reduce import Reduce
from ....abs import FORMAT
from ....utils import INDENT, ReduceType, Schedule
from .dtype_cast import store_cast_expr as _store_cast_expr


def generate_reduce_impl(graph: Graph, op_id: int, ctx: Context) -> str:
    """Schedule.W reduce — inline ``tl.sum`` / ``tl.max`` / etc.

    Operates on the per-workload block ``tensor_<id>_block`` of shape
    ``[workload_chunk_size, D_0, D_1]``; ``op.dim`` is in ``{1, 2}``.
    """
    input_tensor_id = graph.op_to_input_tensor_list[op_id][0]
    output_tensor_id = graph.op_to_output_tensor_list[op_id]
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
# ``[start, end)`` segment of pages described by ``ctx.dense_kv_indptr``,
# applies the reduction over the page axis, and writes a single
# ``[D_0, D_1]`` tile into the BATCHED output at index ``pid``.
#
# bos/eos mirror the softmax convention: the first ``bos`` and last ``eos``
# pages of each segment are excluded (they're typically always-on tokens).
# Set them to ``0`` if you want the whole segment.
# ---------------------------------------------------------------------------

def _reduce_dim0_kernel_body(reduce_type: ReduceType, store_cast: str) -> str:
    """Build the kernel body for one reduce_type, with the right init /
    accumulate / finalize snippets and a dtype-correct store cast."""
    if reduce_type == ReduceType.Sum:
        init = "tl.zeros((x_D0, x_D1), dtype=tl.float32)"
        accum = "acc += tl.sum(slab, axis=0)"
        finalize = ""
    elif reduce_type == ReduceType.Mean:
        init = "tl.zeros((x_D0, x_D1), dtype=tl.float32)"
        accum = "acc += tl.sum(slab, axis=0)"
        finalize = "acc = acc / num_pages.to(tl.float32)"
    elif reduce_type == ReduceType.Max:
        init = "tl.full((x_D0, x_D1), -1e30, dtype=tl.float32)"
        accum = (
            "slab = tl.where(p_mask[:, None, None], slab, -1e30)\n"
            "        acc = tl.maximum(acc, tl.max(slab, axis=0))"
        )
        finalize = ""
    elif reduce_type == ReduceType.Min:
        init = "tl.full((x_D0, x_D1), 1e30, dtype=tl.float32)"
        accum = (
            "slab = tl.where(p_mask[:, None, None], slab, 1e30)\n"
            "        acc = tl.minimum(acc, tl.min(slab, axis=0))"
        )
        finalize = ""
    elif reduce_type == ReduceType.L2Norm:
        init = "tl.zeros((x_D0, x_D1), dtype=tl.float32)"
        accum = "acc += tl.sum(slab * slab, axis=0)"
        finalize = "acc = tl.sqrt(acc)"
    else:
        raise NotImplementedError(
            f"reduce_dim0 codegen: unsupported reduce_type {reduce_type}"
        )

    finalize_block = f"\n    {finalize}" if finalize else ""

    return f"""
@triton.jit
def {{kernel_name}}(
    x,
    out,
    indptr,
    bos: tl.constexpr,
    eos: tl.constexpr,
    x_D0: tl.constexpr,
    x_D1: tl.constexpr,
    BLOCK_P: tl.constexpr = 512,
):
    pid = tl.program_id(0)
    start = tl.load(indptr + pid) + bos
    end = tl.load(indptr + pid + 1) - eos

    page_stride = x_D0 * x_D1
    d0_idx = tl.arange(0, x_D0)
    d1_idx = tl.arange(0, x_D1)
    out_ptr = out + pid * page_stride + d0_idx[:, None] * x_D1 + d1_idx[None, :]

    if end <= start:
        zero = tl.zeros((x_D0, x_D1), dtype=tl.float32)
        tl.store(out_ptr, {store_cast.format(block_expr='zero')})
        return

    num_pages = end - start
    acc = {init}

    p_idx = tl.arange(0, BLOCK_P)
    base_ptr = x + start * page_stride

    for p in range(0, num_pages, BLOCK_P):
        kp = tl.minimum(BLOCK_P, num_pages - p)
        p_mask = p_idx < kp
        offs = (
            (p + p_idx)[:, None, None] * page_stride
            + d0_idx[None, :, None] * x_D1
            + d1_idx[None, None, :]
        ).to(tl.int32)
        slab = tl.load(base_ptr + offs, mask=p_mask[:, None, None], other=0.0).to(tl.float32)
        {accum}{finalize_block}

    tl.store(out_ptr, {store_cast.format(block_expr='acc')})
"""


def generate_reduce_dim0_impl(graph: Graph, op_id: int, ctx: Context) -> str:
    """Schedule.S reduce — emit a standalone kernel that collapses the
    packed leading axis of a RAGGED input into a BATCHED output."""
    input_tensor_id = graph.op_to_input_tensor_list[op_id][0]
    output_tensor_id = graph.op_to_output_tensor_list[op_id]
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

    # Each reduce_type gets its own specialized kernel — disambiguated by
    # ``sparse_attention_name`` so multiple flow instances don't collide.
    rt_name = op.reduce_type.name.lower()
    kernel_name = f"{ctx.sparse_attention_name}_reduce_dim0_{rt_name}_kernel"

    # ``_store_cast_expr`` returns a fully-formed ``<expr>.to(...)`` string.
    # We re-template it via ``{block_expr}`` so the kernel body can drop in
    # either ``acc`` or ``zero`` at the right places.
    store_cast_template = _store_cast_expr("{block_expr}", t_o.dtype)

    func_def = _reduce_dim0_kernel_body(op.reduce_type, store_cast_template).format(
        kernel_name=kernel_name
    )
    ctx.auxilary_func_def_lines.append(func_def)

    impl_lines = [
        f"{INDENT}eff_batch_size = ctx.batch_size * ctx.num_kv_heads",
        f"{INDENT}{kernel_name}[(eff_batch_size,)](",
        f"{INDENT*2}tensor_{input_tensor_id},",
        f"{INDENT*2}tensor_{output_tensor_id},",
        f"{INDENT*2}ctx.dense_kv_indptr,",
        f"{INDENT*2}ctx.block_reserved_bos,",
        f"{INDENT*2}ctx.block_reserved_eos,",
        f"{INDENT*2}tensor_{input_tensor_id}.shape[-2],",
        f"{INDENT*2}tensor_{input_tensor_id}.shape[-1],",
        f"{INDENT*2}num_warps=4,",
        f"{INDENT*2}num_stages=1,",
        f"{INDENT})",
    ]
    return "\n".join(impl_lines)