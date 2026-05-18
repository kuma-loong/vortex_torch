from ..graph import Graph
from ...context import Context
from ...reduce import Reduce
from ....utils import ReduceType


def generate_reduce_impl(graph: Graph, op_id: int, ctx: Context) -> str:
    """Schedule.W reduce — inline ``tl.sum`` / ``tl.max`` / etc.

    Operates on the per-workload block ``tensor_<id>_block`` of shape
    ``[workload_chunk_size, D_0, D_1]``; ``op.dim`` is in ``{1, 2}``.
    The cross-row ``dim == 0`` form is Schedule.S and lives in
    :mod:`indexer.compiler.custom_impl.reduce_dim0`.
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
