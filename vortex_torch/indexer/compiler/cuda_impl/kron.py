"""CUDA codegen for the Schedule.W ``Kron`` op.

Kronecker product along the axes listed in ``op.dim`` ⊆ ``{1, 2}``,
elementwise (with broadcast) along the others. Output index
decomposition for one element ``(chunk_i, c_o, d_o)``:

  * Axis 1 in ``op.dim``  →  ``c_x = c_o / y1``,  ``c_y = c_o % y1``.
                              Output size on this axis is ``x1 * y1``.
  * Axis 1 elementwise    →  ``c_x = (x1 == 1) ? 0 : c_o``,
                              ``c_y = (y1 == 1) ? 0 : c_o``. Profile
                              has guaranteed ``x1 == y1`` or one is 1.
  * Axis 2 mirrors the same pattern via ``y2`` and ``d_o``.

The output value is a single fp32 multiply per element:

    O[chunk_i, c_o, d_o] = X[chunk_i_x, c_x, d_x] * Y[chunk_i_y, c_y, d_y]

with format-aware chunk indexing: a BATCHED input has ``chunk_i_* = 0``
(broadcast along the workload axis), a RAGGED/PAGED input has
``chunk_i_* = chunk_i``. The fp32 intermediate prevents the bf16/fp16
multiply from rounding twice (input cast → bf16 multiply → bf16 write).

Mirrors :mod:`triton_impl.kron`: Triton does this via two
``tl.reshape`` + a single broadcast multiply + a final reshape; CUDA
does it as one per-thread index decomposition + one fp32 multiply +
one store. Both produce the same row-major Kronecker layout.
"""

from ..graph import Graph
from ...context import Context
from ...kron import Kron
from ....abs import FORMAT
from ....utils import INDENT

from .dtype_cast import cast_smem_to_float, cast_float_to_smem


def generate_kron_impl(graph: Graph, op_id: int, ctx: Context) -> str:
    input_ids = graph.op_to_input_tensor_list[op_id]
    assert len(input_ids) == 2, (
        f"Kron expects 2 inputs, got {len(input_ids)}"
    )
    input_id_x, input_id_y = input_ids
    output_id = graph.op_to_output_tensor_list[op_id]
    op = graph.op_list[op_id]
    assert issubclass(op.__class__, Kron), (
        f"Expected a Kron op, got {op.__class__.__name__}"
    )

    t_x = graph.tensor_list[input_id_x]
    t_y = graph.tensor_list[input_id_y]
    t_o = graph.tensor_list[output_id]

    x1, x2 = t_x.shape[1], t_x.shape[2]
    y1, y2 = t_y.shape[1], t_y.shape[2]
    C_o, D_o = t_o.shape[1], t_o.shape[2]

    leading = 1 if t_o._format == FORMAT.BATCHED else ctx.workload_chunk_size
    n_out = leading * C_o * D_o

    chunk_i_x = "0" if t_x._format == FORMAT.BATCHED else "chunk_i"
    chunk_i_y = "0" if t_y._format == FORMAT.BATCHED else "chunk_i"

    # Per-input axis-1 / axis-2 index expressions in terms of ``c_o`` /
    # ``d_o`` (the output coordinates). For Kron axes the size factors
    # are baked at codegen time so the runtime sees two integer
    # divisions / mods on compile-time-constant divisors. For
    # elementwise axes a broadcasted input (size 1) collapses to a
    # literal ``0``; otherwise the axis passes through.
    if 1 in op.dim:
        c_x_expr = f"c_o / {y1}"
        c_y_expr = f"c_o % {y1}"
    else:
        c_x_expr = "0" if x1 == 1 else "c_o"
        c_y_expr = "0" if y1 == 1 else "c_o"

    if 2 in op.dim:
        d_x_expr = f"d_o / {y2}"
        d_y_expr = f"d_o % {y2}"
    else:
        d_x_expr = "0" if x2 == 1 else "d_o"
        d_y_expr = "0" if y2 == 1 else "d_o"

    x_subscript = (
        f"tensor_{input_id_x}_smem"
        f"[{chunk_i_x}][{c_x_expr}][{d_x_expr}]"
    )
    y_subscript = (
        f"tensor_{input_id_y}_smem"
        f"[{chunk_i_y}][{c_y_expr}][{d_y_expr}]"
    )
    out_subscript = f"tensor_{output_id}_smem[chunk_i][c_o][d_o]"

    x_load = cast_smem_to_float(x_subscript, t_x.dtype)
    y_load = cast_smem_to_float(y_subscript, t_y.dtype)
    store_expr = cast_float_to_smem("xv * yv", t_o.dtype)

    return (
        f"__syncthreads();\n"
        f"{{\n"
        f"{INDENT}for (int j_out = tid; j_out < {n_out}; j_out += blockDim.x) {{\n"
        f"{INDENT}{INDENT}const int chunk_i = j_out / {C_o * D_o};\n"
        f"{INDENT}{INDENT}const int c_o     = (j_out / {D_o}) % {C_o};\n"
        f"{INDENT}{INDENT}const int d_o     =  j_out % {D_o};\n"
        f"\n"
        f"{INDENT}{INDENT}const float xv = {x_load};\n"
        f"{INDENT}{INDENT}const float yv = {y_load};\n"
        f"{INDENT}{INDENT}{out_subscript} = {store_expr};\n"
        f"{INDENT}}}\n"
        f"}}"
    )
