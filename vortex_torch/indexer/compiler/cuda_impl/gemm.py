"""CUDA codegen for the Schedule.W ``GeMM`` op.

Per-slice matrix-matrix product

    O[s, i, j] = sum_k  Y[s, i, k] * X[s_or_b, j, k]

with operand and output shapes

    X : [B_or_S, Nx, K]   (input 0; ``s_or_b`` is 0 when X is BATCHED)
    Y : [S,       Ny, K]   (input 1; ``s`` is 0 when Y is BATCHED)
    O : [S,       Ny, Nx]  (BATCHED iff both inputs are BATCHED)

Mirrors :mod:`triton_impl.gemm` op-for-op; translated from Triton's
broadcasted tile multiply + ``tl.sum`` to a per-thread fp32-accumulated
inner-product. Each thread owns one output element ``(chunk_i, i_o,
j_o)`` and sweeps the K axis sequentially in registers. No
cross-thread coordination — output positions are independent. fp32
accumulator with explicit ``fmaf(xv, yv, acc)`` so nvcc emits a
single rounding per FMA (matches Triton's fp32 ``tl.sum`` precision
profile).

Format-aware chunk indexing per input handles the BATCHED-broadcast
cases (X BATCHED + Y RAGGED, the GEMV-special case Triton has a
dedicated codegen path for, etc.) uniformly: a BATCHED input's
``chunk_i_*`` is the literal ``0``, baked at codegen time so nvcc
constant-propagates the unused term.
"""

from ..graph import Graph
from ...context import Context
from ...matmul import GeMM
from ....abs import FORMAT
from ....utils import INDENT

from .dtype_cast import cast_smem_to_float, cast_float_to_smem


def generate_gemm_impl(graph: Graph, op_id: int, ctx: Context) -> str:
    input_id_x, input_id_y = graph.op_to_input_tensor_list[op_id]
    output_id = graph.op_to_output_tensor_list[op_id][0]
    op = graph.op_list[op_id]
    assert issubclass(op.__class__, GeMM), (
        f"Expected a GeMM op, got {op.__class__.__name__}"
    )

    t_x = graph.tensor_list[input_id_x]
    t_y = graph.tensor_list[input_id_y]
    t_o = graph.tensor_list[output_id]

    Nx = t_x.shape[1]
    Ny = t_y.shape[1]
    assert t_x.shape[2] == t_y.shape[2], (
        f"GeMM: K mismatch — X.shape[2]={t_x.shape[2]}, "
        f"Y.shape[2]={t_y.shape[2]}"
    )
    K = t_x.shape[2]

    assert t_o.shape[1] == Ny and t_o.shape[2] == Nx, (
        f"GeMM: output shape mismatch — expected (Ny={Ny}, Nx={Nx}), "
        f"got ({t_o.shape[1]}, {t_o.shape[2]})"
    )

    leading = 1 if t_o._format == FORMAT.BATCHED else ctx.workload_chunk_size
    n_out = leading * Ny * Nx

    chunk_i_x = "0" if t_x._format == FORMAT.BATCHED else "chunk_i"
    chunk_i_y = "0" if t_y._format == FORMAT.BATCHED else "chunk_i"

    x_subscript = f"tensor_{input_id_x}_smem[{chunk_i_x}][j_o][k]"
    y_subscript = f"tensor_{input_id_y}_smem[{chunk_i_y}][i_o][k]"
    out_subscript = f"tensor_{output_id}_smem[chunk_i][i_o][j_o]"

    x_load = cast_smem_to_float(x_subscript, t_x.dtype)
    y_load = cast_smem_to_float(y_subscript, t_y.dtype)
    store_expr = cast_float_to_smem("acc", t_o.dtype)

    return (
        f"__syncthreads();\n"
        f"{{\n"
        f"{INDENT}for (int j_out = tid; j_out < {n_out}; j_out += blockDim.x) {{\n"
        f"{INDENT}{INDENT}const int chunk_i = j_out / {Ny * Nx};\n"
        f"{INDENT}{INDENT}const int i_o     = (j_out / {Nx}) % {Ny};\n"
        f"{INDENT}{INDENT}const int j_o     =  j_out % {Nx};\n"
        f"\n"
        f"{INDENT}{INDENT}float acc = 0.0f;\n"
        f"{INDENT}{INDENT}for (int k = 0; k < {K}; ++k) {{\n"
        f"{INDENT}{INDENT}{INDENT}const float xv = {x_load};\n"
        f"{INDENT}{INDENT}{INDENT}const float yv = {y_load};\n"
        f"{INDENT}{INDENT}{INDENT}acc = fmaf(xv, yv, acc);\n"
        f"{INDENT}{INDENT}}}\n"
        f"{INDENT}{INDENT}{out_subscript} = {store_expr};\n"
        f"{INDENT}}}\n"
        f"}}"
    )
