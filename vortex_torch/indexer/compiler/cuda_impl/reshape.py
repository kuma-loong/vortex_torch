"""CUDA codegen for the indexer ``Reshape`` op.

Same-numel reshape of the inner two axes: input
``[leading, x1, y1]`` → output ``[leading, x2, y2]`` with
``x1 * y1 == x2 * y2`` and the leading axis preserved
(``leading = workload_chunk_size`` for RAGGED, ``1`` for BATCHED —
input and output formats are always equal by construction in
:class:`indexer.reshape.Reshape.profile`).

In Triton this is a single ``tl.reshape(block, (W, x2, y2))`` on the
in-register tile. In CUDA the smem tiles are separate physical
buffers, so the codegen emits a strided cooperative copy of the
``leading * x1 * y1`` elements from input smem to output smem in
their natural row-major byte order. The two tiles have identical
total element counts (the same-numel guarantee), so flat-indexed
``(&tensor_X_smem[0][0][0])[j]`` reads and ``(&tensor_O_smem[0][0][0])[j]``
writes line up element-for-element without per-iteration index
recomputation.

The leading ``__syncthreads()`` is correctness-required: the input
tile may have been written cross-thread by a previous op (load /
GEMM / etc.) and this reshape may read those slots from a different
thread.
"""

from ..graph import Graph
from ...context import Context
from ...reshape import Reshape
from ....abs import FORMAT
from ....utils import INDENT

from .dtype_cast import cast_smem_to_float, cast_float_to_smem


def generate_reshape_impl(graph: Graph, op_id: int, ctx: Context) -> str:
    input_tensor_id  = graph.op_to_input_tensor_list[op_id][0]
    output_tensor_id = graph.op_to_output_tensor_list[op_id]
    op = graph.op_list[op_id]
    assert issubclass(op.__class__, Reshape), (
        f"Expected a Reshape op, got {op.__class__.__name__}"
    )

    t_x = graph.tensor_list[input_tensor_id]
    t_o = graph.tensor_list[output_tensor_id]

    in_numel  = t_x.shape[1] * t_x.shape[2]
    out_numel = t_o.shape[1] * t_o.shape[2]
    assert in_numel == out_numel, (
        f"Reshape: same-numel required, got input "
        f"{t_x.shape[1]}*{t_x.shape[2]}={in_numel}  vs  output "
        f"{t_o.shape[1]}*{t_o.shape[2]}={out_numel}"
    )

    leading = 1 if t_x._format == FORMAT.BATCHED else ctx.workload_chunk_size
    # Format-preserving by Reshape.profile contract — defensive check.
    out_leading = 1 if t_o._format == FORMAT.BATCHED else ctx.workload_chunk_size
    assert leading == out_leading, (
        f"Reshape: leading-dim mismatch (in={leading}, out={out_leading}); "
        f"format in={t_x._format!r}, out={t_o._format!r}"
    )
    n_total = leading * in_numel

    src = f"(&tensor_{input_tensor_id}_smem[0][0][0])[j]"
    dst = f"(&tensor_{output_tensor_id}_smem[0][0][0])[j]"

    if t_x.dtype is t_o.dtype:
        body_inner = f"{dst} = {src};"
    else:
        load_expr  = cast_smem_to_float(src, t_x.dtype)
        store_expr = cast_float_to_smem("val", t_o.dtype)
        body_inner = (
            f"const float val = {load_expr};\n"
            f"{INDENT}{INDENT}{dst} = {store_expr};"
        )

    return (
        f"__syncthreads();\n"
        f"{{\n"
        f"{INDENT}for (int j = tid; j < {n_total}; j += blockDim.x) {{\n"
        f"{INDENT}{INDENT}{body_inner}\n"
        f"{INDENT}}}\n"
        f"}}"
    )
