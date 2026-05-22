from ..graph import Graph
from ...context import Context
from ...reduce import Reduce
from ....utils import ReduceType
from .dtype_cast import compute_tl_dtype as _compute_tl_dtype


def generate_reduce_impl(graph: Graph, op_id: int, ctx: Context) -> str:
    """Schedule.W reduce — inline ``tl.sum`` / ``tl.max`` / etc.

    Operates on the per-workload block ``tensor_<id>_block`` of shape
    ``[workload_chunk_size, D_0, D_1]``; ``op.dim`` is in ``{1, 2}``.
    The cross-row ``dim == 0`` form is Schedule.S and lives in
    :mod:`indexer.compiler.custom_impl.reduce_dim0`.

    Tensor-core (bf16-compute) mode. When ``ctx.use_tensor_core`` is set
    the input block is bf16. Reductions that *accumulate* (Sum / Mean /
    L2Norm) promote to fp32 for the running total — bf16's 8-bit
    mantissa loses too much precision summing a wide head dim — then
    downcast the result back to bf16 so the block dtype stays uniform
    across the op chain. ``Max`` / ``Min`` are selection reductions with
    no accumulation error, so they stay in the input (bf16) dtype.
    """
    input_tensor_id = graph.op_to_input_tensor_list[op_id][0]
    output_tensor_id = graph.op_to_output_tensor_list[op_id][0]
    op = graph.op_list[op_id]
    t_i = graph.tensor_list[input_tensor_id]
    assert issubclass(op.__class__, Reduce), f"Expected a reduce op, got {graph.op_list[op_id]}"
    assert op.dim in (1, 2), (
        f"generate_reduce_impl handles dim in {{1, 2}}; got dim={op.dim}. "
        f"dim=0 is Schedule.S — see indexer.compiler.custom_impl.reduce_dim0."
    )

    in_block = f"tensor_{input_tensor_id}_block"
    out_block = f"tensor_{output_tensor_id}_block"
    use_tc = getattr(ctx, "use_tensor_core", False)
    cdt = _compute_tl_dtype(ctx)

    def _downcast(expr: str) -> str:
        # In tensor-core mode the accumulating reducers compute in fp32;
        # downcast the result to the bf16 compute dtype. No-op otherwise.
        return f"({expr}).to({cdt})" if use_tc else expr

    if op.reduce_type == ReduceType.Sum:
        src = f"{in_block}.to(tl.float32)" if use_tc else in_block
        impl_lines = [
            f"{out_block} = {_downcast(f'tl.sum({src}, keep_dims=True, axis={op.dim})')}",
        ]
    elif op.reduce_type == ReduceType.Max:
        impl_lines = [
            f"{out_block} = tl.max({in_block}, keep_dims=True, axis={op.dim})",
        ]
    elif op.reduce_type == ReduceType.Min:
        impl_lines = [
            f"{out_block} = tl.min({in_block}, keep_dims=True, axis={op.dim})",
        ]
    elif op.reduce_type == ReduceType.L2Norm:
        impl_lines = [
            f"{out_block} = {_downcast(f'tl.sqrt(tl.sum(({in_block} * {in_block}).to(tl.float32), keep_dims=True, axis={op.dim}))')}",
        ]
    elif op.reduce_type == ReduceType.Mean:
        src = f"{in_block}.to(tl.float32)" if use_tc else in_block
        impl_lines = [
            f"{out_block} = {_downcast(f'tl.sum({src}, keep_dims=True, axis={op.dim}) * ({1.0 / t_i.shape[op.dim]})')}",
        ]
    impl_str = "\n".join(impl_lines)
    return impl_str
