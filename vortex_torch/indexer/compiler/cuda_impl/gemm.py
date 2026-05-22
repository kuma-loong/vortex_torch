"""CUDA codegen for the Schedule.W ``GeMM`` op.

Per-slice matrix-matrix product

    O[s, i, j] = sum_k  Y[s, i, k] * X[s_or_b, j, k]

with operand and output shapes

    X : [B_or_S, Nx, K]   (input 0; ``s_or_b`` is 0 when X is BATCHED)
    Y : [S,       Ny, K]   (input 1; ``s`` is 0 when Y is BATCHED)
    O : [S,       Ny, Nx]  (BATCHED iff both inputs are BATCHED)

Mirrors :mod:`triton_impl.gemm` op-for-op; translated from Triton's
broadcasted tile multiply + ``tl.sum`` to a per-thread fp32-accumulated
inner-product.

Two thread bindings, selected at codegen time on ``K``:

  * **Warp-per-output** (``K >= 32``) — each warp owns one output
    ``(chunk_i, i_o, j_o)``. Lanes stride the K axis, each accumulates
    ``K / 32`` FMAs into a fp32 register, then a 5-step
    ``__shfl_xor_sync`` butterfly collapses the 32 partials. Lane 0
    writes the smem slot. For the canonical inner product
    ``q @ k`` (D=128, Nx=Ny=1) this turns ``128`` serial FMAs into
    ``4`` FMAs + a shfl tree; with the 8-warp block (256 threads) we
    drive 8 outputs in parallel per block-stride iteration vs. 1
    thread out of 256 under the old binding.

  * **Thread-per-output** (``K < 32``) — each thread owns one output
    and sweeps K sequentially. The shfl tree's overhead would dominate
    for tiny K, so we keep the original codegen.

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

    # ---- Warp-per-output specialisation (K >= 32) -------------------
    if K >= 32:
        shfl_steps = "\n".join(
            f"{INDENT}{INDENT}acc += __shfl_xor_sync(0xffffffffu, acc, {offset});"
            for offset in (16, 8, 4, 2, 1)
        )
        return (
            f"__syncthreads();\n"
            f"{{\n"
            f"{INDENT}const int __wid   = tid >> 5;\n"
            f"{INDENT}const int __lane  = tid & 31;\n"
            f"{INDENT}const int __warps = blockDim.x >> 5;\n"
            f"{INDENT}for (int j_out = __wid; j_out < {n_out}; "
            f"j_out += __warps) {{\n"
            f"{INDENT}{INDENT}const int chunk_i = j_out / {Ny * Nx};\n"
            f"{INDENT}{INDENT}const int i_o     = (j_out / {Nx}) % {Ny};\n"
            f"{INDENT}{INDENT}const int j_o     =  j_out % {Nx};\n"
            f"\n"
            f"{INDENT}{INDENT}float acc = 0.0f;\n"
            f"{INDENT}{INDENT}for (int k = __lane; k < {K}; k += 32) {{\n"
            f"{INDENT}{INDENT}{INDENT}const float xv = {x_load};\n"
            f"{INDENT}{INDENT}{INDENT}const float yv = {y_load};\n"
            f"{INDENT}{INDENT}{INDENT}acc = fmaf(xv, yv, acc);\n"
            f"{INDENT}{INDENT}}}\n"
            f"{shfl_steps}\n"
            f"{INDENT}{INDENT}if (__lane == 0) {{\n"
            f"{INDENT}{INDENT}{INDENT}{out_subscript} = {store_expr};\n"
            f"{INDENT}{INDENT}}}\n"
            f"{INDENT}}}\n"
            f"}}"
        )

    # ---- Thread-per-output fallback (K < 32) ------------------------
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
