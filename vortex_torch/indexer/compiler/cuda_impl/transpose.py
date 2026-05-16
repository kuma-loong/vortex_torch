"""CUDA codegen for the ``Transpose`` op.

Transposes the inner two axes of a per-workload smem tile:

    tensor_O_smem[chunk_i][c_o][d_o] = tensor_X_smem[chunk_i][d_o][c_o]

The input tile has shape ``[leading, D0, D1]`` and the output has
``[leading, D1, D0]`` — output index ``(c_o, d_o)`` corresponds to
input index ``(d_o, c_o)`` per the rank-3 swap defined by
:class:`indexer.transpose.Transpose`.

Mirrors :mod:`triton_impl.transpose` (``tl.trans(block, 0, 2, 1)``)
but emits per-thread strided access against smem instead of a single
program-level tile transpose. Cross-thread reads of the input — a
thread's output slot at ``(c_o, d_o)`` pulls from the input slot at
``(d_o, c_o)``, which may have been written by a different thread in
a prior op — so the leading ``__syncthreads()`` is required.
"""

from ..graph import Graph
from ...context import Context
from ...transpose import Transpose
from ....abs import FORMAT
from ....utils import INDENT

from .dtype_cast import cast_smem_to_float, cast_float_to_smem


def generate_transpose_impl(graph: Graph, op_id: int, ctx: Context) -> str:
    input_tensor_id  = graph.op_to_input_tensor_list[op_id][0]
    output_tensor_id = graph.op_to_output_tensor_list[op_id]
    op = graph.op_list[op_id]

    assert issubclass(op.__class__, Transpose), (
        f"Expected a transpose op, got {op.__class__.__name__}"
    )

    t_x = graph.tensor_list[input_tensor_id]
    t_o = graph.tensor_list[output_tensor_id]

    # Output is the transpose: t_o.shape[1] == t_x.shape[2],
    # t_o.shape[2] == t_x.shape[1]. Defensive assert.
    assert t_o.shape[1] == t_x.shape[2] and t_o.shape[2] == t_x.shape[1], (
        f"Transpose: output shape {t_o.shape} not a swap of input "
        f"shape {t_x.shape}"
    )

    C_o, D_o = t_o.shape[1], t_o.shape[2]
    leading = 1 if t_o._format == FORMAT.BATCHED else ctx.workload_chunk_size
    n_out = leading * C_o * D_o

    src_subscript = f"tensor_{input_tensor_id}_smem[chunk_i][d_o][c_o]"
    dst_subscript = f"tensor_{output_tensor_id}_smem[chunk_i][c_o][d_o]"

    if t_x.dtype is t_o.dtype:
        # Same storage dtype: a direct multi-dim copy is faster than
        # round-tripping through fp32 — for ``__nv_bfloat16`` /
        # ``__half`` / ``float`` / ``uint8_t`` the implicit copy ABI
        # collapses to a single load-store pair.
        body = f"{dst_subscript} = {src_subscript};"
    else:
        load_expr  = cast_smem_to_float(src_subscript, t_x.dtype)
        store_expr = cast_float_to_smem("val", t_o.dtype)
        body = (
            f"const float val = {load_expr};\n"
            f"{INDENT}{INDENT}{dst_subscript} = {store_expr};"
        )

    return (
        f"__syncthreads();\n"
        f"{{\n"
        f"{INDENT}for (int j_out = tid; j_out < {n_out}; j_out += blockDim.x) {{\n"
        f"{INDENT}{INDENT}const int chunk_i = j_out / {C_o * D_o};\n"
        f"{INDENT}{INDENT}const int c_o     = (j_out / {D_o}) % {C_o};\n"
        f"{INDENT}{INDENT}const int d_o     =  j_out % {D_o};\n"
        f"{INDENT}{INDENT}{body}\n"
        f"{INDENT}}}\n"
        f"}}"
    )
