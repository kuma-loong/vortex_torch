"""CUDA codegen for the ``Elementwise_Binary`` op family.

Unlike the unary case, binary ops can mix tensors of different shapes /
formats via broadcasting:

  * **Chunk axis** — a BATCHED tensor (leading dim ``1`` in its smem
    tile) combined with a RAGGED tensor (leading dim
    ``workload_chunk_size``) makes the BATCHED value broadcast across
    the chunk. The op-profile sets ``output_format = BATCHED`` only when
    both inputs are BATCHED, else RAGGED — so the output's leading
    dimension follows the wider of the two inputs.
  * **C / D axes** — ``shape[1]`` or ``shape[2]`` of ``1`` on an input
    broadcasts across the corresponding output axis (the profile asserts
    ``shape == 1 or shape == output_shape`` on each input axis).

To handle all three broadcasting axes uniformly we decompose the flat
output index ``j_out`` into ``(chunk_i, c, d)`` and rebuild a
per-input index expression with the broadcasting collapse rule
(broadcasted axis → ``0``). The compute itself runs in fp32 (matching
Triton's compute precision) with storage-dtype casts on the smem
boundary.

Mirrors :mod:`triton_impl.elementwise_binary` op-by-op (Add / Mul /
Maximum / Minimum / WhereEqual / WhereNotEqual / Where{Greater,Less}
{,Equal}), translated from Triton tile ops to per-thread strided
access. The ``Where*`` ops emit the standard ``-1e30f`` sentinel
(matching Triton's ``tl.full(..., -1e30)`` — a "near-infinity" value
that won't propagate NaNs through a downstream softmax the way a true
``-inf`` would).
"""

from typing import Dict

from ..graph import Graph
from ...context import Context
from ...elementwise_binary import Elementwise_Binary
from ....abs import FORMAT
from ....utils import ElementwiseBinaryOpType, INDENT

from .dtype_cast import cast_smem_to_float, cast_float_to_smem


# Map ElementwiseBinaryOpType -> fp32 compute expression in terms of
# local ``float x`` / ``float y``, ``alpha``, ``beta``. ``NEG_INF``
# resolves to ``-1e30f`` so generated code reads cleanly.
NEG_INF = "-1.0e30f"
_COMPUTE_EXPR: Dict[ElementwiseBinaryOpType, str] = {
    ElementwiseBinaryOpType.Add:     "{alpha}f * x + {beta}f * y",
    ElementwiseBinaryOpType.Mul:     "x * y",
    ElementwiseBinaryOpType.Maximum: "fmaxf(x, y)",
    ElementwiseBinaryOpType.Minimum: "fminf(x, y)",
    ElementwiseBinaryOpType.WhereEqual:        "(x == y) ? 0.0f : " + NEG_INF,
    ElementwiseBinaryOpType.WhereNotEqual:     "(x != y) ? 0.0f : " + NEG_INF,
    ElementwiseBinaryOpType.WhereGreater:      "(x >  y) ? 0.0f : " + NEG_INF,
    ElementwiseBinaryOpType.WhereGreaterEqual: "(x >= y) ? 0.0f : " + NEG_INF,
    ElementwiseBinaryOpType.WhereLess:         "(x <  y) ? 0.0f : " + NEG_INF,
    ElementwiseBinaryOpType.WhereLessEqual:    "(x <= y) ? 0.0f : " + NEG_INF,
}


def _input_index_decls(tid: int, t) -> str:
    """Per-iteration ``chunk_i_<tid>`` / ``c_<tid>`` / ``d_<tid>``
    declarations that collapse broadcasted axes to ``0``.

    The naming is unique per input tensor id so the binary's two input
    index sets don't collide. The collapse decisions are baked at
    codegen time — ``shape[1] == 1`` makes ``c_<tid>`` a literal ``0``,
    not a runtime conditional, so nvcc folds out the unused term.
    """
    chunk_i_x = "0" if t._format == FORMAT.BATCHED else "chunk_i"
    c_x       = "0" if t.shape[1] == 1 else "c"
    d_x       = "0" if t.shape[2] == 1 else "d"
    return (
        f"const int chunk_i_{tid} = {chunk_i_x};\n"
        f"const int c_{tid}       = {c_x};\n"
        f"const int d_{tid}       = {d_x};"
    )


def generate_elementwise_binary_impl(
    graph: Graph, op_id: int, ctx: Context,
) -> str:
    """Emit the per-block CUDA C++ source for an Elementwise_Binary op.

    Layout::

        __syncthreads();
        {
            for (int j_out = tid; j_out < N_out; j_out += blockDim.x) {
                const int chunk_i = j_out / (C_out * D_out);
                const int c       = (j_out / D_out) % C_out;
                const int d       =  j_out % D_out;

                <per-input broadcast-collapse decls for X>
                <per-input broadcast-collapse decls for Y>

                const float x   = <cast smem -> float>(tensor_X_smem
                                       [chunk_i_X][c_X][d_X]);
                const float y   = <cast smem -> float>(tensor_Y_smem
                                       [chunk_i_Y][c_Y][d_Y]);
                const float out = <op-specific expression in x, y, alpha, beta>;
                tensor_O_smem[chunk_i][c][d] = <cast float -> smem>(out);
            }
        }

    The output decomposition ``(chunk_i, c, d)`` parameterizes each
    input's flat index — a broadcasted input axis (size ``1``) folds
    its corresponding index to ``0`` at codegen time (see
    :func:`_input_index_decls`) so the generated multi-dim subscript
    is a literal ``[0]`` and nvcc constant-propagates.

    Leading ``__syncthreads()`` covers the broadcasting case where
    several threads read the **same** smem slot of a broadcasted input
    (e.g. all threads in the block reading ``tensor_X_smem[0][c][d]``
    when X is BATCHED): the producer of that slot may have been a
    different thread, so we need its writes visible.
    """
    input_tensor_id_0 = graph.op_to_input_tensor_list[op_id][0]
    input_tensor_id_1 = graph.op_to_input_tensor_list[op_id][1]
    output_tensor_id  = graph.op_to_output_tensor_list[op_id][0]
    op = graph.op_list[op_id]
    t_x = graph.tensor_list[input_tensor_id_0]
    t_y = graph.tensor_list[input_tensor_id_1]
    t_o = graph.tensor_list[output_tensor_id]

    assert issubclass(op.__class__, Elementwise_Binary), (
        f"Expected an elementwise binary op, got {graph.op_list[op_id]}"
    )

    try:
        compute_template = _COMPUTE_EXPR[op.op_type]
    except KeyError:
        raise NotImplementedError(
            f"cuda_impl.elementwise_binary: no compute expression for "
            f"{op.op_type!r}"
        )
    compute_expr = compute_template.format(alpha=op.alpha, beta=op.beta)

    C_o, D_o = t_o.shape[1], t_o.shape[2]
    leading_o = 1 if t_o._format == FORMAT.BATCHED else ctx.workload_chunk_size
    n_out = leading_o * C_o * D_o

    x_decls = _input_index_decls(input_tensor_id_0, t_x)
    y_decls = _input_index_decls(input_tensor_id_1, t_y)

    x_load = cast_smem_to_float(
        f"tensor_{input_tensor_id_0}_smem"
        f"[chunk_i_{input_tensor_id_0}]"
        f"[c_{input_tensor_id_0}]"
        f"[d_{input_tensor_id_0}]",
        t_x.dtype,
    )
    y_load = cast_smem_to_float(
        f"tensor_{input_tensor_id_1}_smem"
        f"[chunk_i_{input_tensor_id_1}]"
        f"[c_{input_tensor_id_1}]"
        f"[d_{input_tensor_id_1}]",
        t_y.dtype,
    )
    store_dst = f"tensor_{output_tensor_id}_smem[chunk_i][c][d]"
    store_val = cast_float_to_smem("out", t_o.dtype)

    # Indent the per-input decls into the loop body (one level past
    # ``for (...)``, i.e. two levels from the outer scope).
    x_decls_in = "\n".join(f"{INDENT}{INDENT}{line}" for line in x_decls.splitlines())
    y_decls_in = "\n".join(f"{INDENT}{INDENT}{line}" for line in y_decls.splitlines())

    return (
        f"__syncthreads();\n"
        f"{{\n"
        f"{INDENT}for (int j_out = tid; j_out < {n_out}; j_out += blockDim.x) {{\n"
        f"{INDENT}{INDENT}const int chunk_i = j_out / {C_o * D_o};\n"
        f"{INDENT}{INDENT}const int c       = (j_out / {D_o}) % {C_o};\n"
        f"{INDENT}{INDENT}const int d       =  j_out % {D_o};\n"
        f"\n"
        f"{x_decls_in}\n"
        f"{y_decls_in}\n"
        f"\n"
        f"{INDENT}{INDENT}const float x   = {x_load};\n"
        f"{INDENT}{INDENT}const float y   = {y_load};\n"
        f"{INDENT}{INDENT}const float out = {compute_expr};\n"
        f"{INDENT}{INDENT}{store_dst} = {store_val};\n"
        f"{INDENT}}}\n"
        f"}}"
    )
