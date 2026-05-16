"""CUDA codegen for the ``Elementwise`` (unary) op family.

The unary case is purely thread-local: input shape == output shape and
input format == output format, so each thread strides through the same
flat index range in both tiles. No broadcasting, no cross-thread reads.

Mirrors :mod:`triton_impl.elementwise` op-by-op (Relu / Sigmoid / Silu
/ Abs / Add_Mul / Log / Exp), translated from Triton ``tl.*`` ops to
CUDA ``__expf`` / ``__logf`` / ``fabsf`` intrinsics and from per-program
register tiles to per-thread strided smem access.
"""

from typing import Dict

from ..graph import Graph
from ...context import Context
from ...elementwise import Elementwise
from ....abs import FORMAT
from ....utils import ElementwiseOpType, INDENT

from .dtype_cast import cast_smem_to_float, cast_float_to_smem


# Map ElementwiseOpType -> fp32 compute expression in terms of a local
# ``float x`` (the cast-to-float input value), ``alpha``, ``beta``.
# ``__expf`` / ``__logf`` / ``fabsf`` are CUDA single-precision
# fast-math intrinsics (``<cuda_fp16.h>`` / ``<cmath>``-provided),
# matching the precision profile of Triton's ``tl.exp`` etc.
_COMPUTE_EXPR: Dict[ElementwiseOpType, str] = {
    ElementwiseOpType.Relu:    "(x >= {alpha}f) ? x : {beta}f",
    ElementwiseOpType.Sigmoid: "1.0f / (1.0f + __expf({beta}f * x + {alpha}f))",
    ElementwiseOpType.Silu:    "x / (1.0f + __expf({beta}f * x + {alpha}f))",
    ElementwiseOpType.Abs:     "fabsf({beta}f * x + {alpha}f)",
    ElementwiseOpType.Add_Mul: "{beta}f * x + {alpha}f",
    ElementwiseOpType.Log:     "__logf({beta}f * x + {alpha}f)",
    ElementwiseOpType.Exp:     "__expf({beta}f * x + {alpha}f)",
}


def generate_elementwise_impl(graph: Graph, op_id: int, ctx: Context) -> str:
    """Emit the per-block CUDA C++ source for a unary Elementwise op.

    Layout::

        __syncthreads();
        {
            for (int j = tid; j < N; j += blockDim.x) {
                const float x   = <cast smem -> float>;
                const float out = <op-specific expression in x, alpha, beta>;
                <smem write> = <cast float -> smem>;
            }
        }

    ``N = leading * shape[1] * shape[2]`` where ``leading`` is ``1`` for
    BATCHED and ``workload_chunk_size`` otherwise (matches the smem
    tile shape from :func:`generate_initialization_str`). Flat indexing
    via ``(&tensor_<id>_smem[0][0][0])[j]`` works because the 3D smem
    array is contiguous; no per-iteration index decomposition is
    needed since input and output share the same shape and layout.

    The leading ``__syncthreads()`` ensures the input tile is consistent
    — even though this op's own reads/writes are thread-local, the
    producer of the input may have written cross-thread (the load
    codegen does so when populating multi-page PAGED tiles, and future
    GEMM / reduce ops will too).
    """
    input_tensor_id = graph.op_to_input_tensor_list[op_id][0]
    output_tensor_id = graph.op_to_output_tensor_list[op_id]
    op = graph.op_list[op_id]
    t_i = graph.tensor_list[input_tensor_id]
    t_o = graph.tensor_list[output_tensor_id]

    assert issubclass(op.__class__, Elementwise), (
        f"Expected an elementwise op, got {graph.op_list[op_id]}"
    )
    # Unary elementwise inherits format and shape from its input.
    assert t_i._format == t_o._format, (
        f"Elementwise expects matching formats; got input {t_i._format!r} "
        f"vs output {t_o._format!r}"
    )
    assert t_i.shape[1] == t_o.shape[1] and t_i.shape[2] == t_o.shape[2], (
        f"Elementwise expects matching (C, D); got input {t_i.shape} "
        f"vs output {t_o.shape}"
    )

    try:
        compute_template = _COMPUTE_EXPR[op.op_type]
    except KeyError:
        raise NotImplementedError(
            f"cuda_impl.elementwise: no compute expression for {op.op_type!r}"
        )
    compute_expr = compute_template.format(alpha=op.alpha, beta=op.beta)

    leading = 1 if t_o._format == FORMAT.BATCHED else ctx.workload_chunk_size
    n_elems = leading * t_o.shape[1] * t_o.shape[2]

    load_expr = cast_smem_to_float(
        f"(&tensor_{input_tensor_id}_smem[0][0][0])[j]", t_i.dtype,
    )
    store_dst = f"(&tensor_{output_tensor_id}_smem[0][0][0])[j]"
    store_val = cast_float_to_smem("out", t_o.dtype)

    return (
        f"__syncthreads();\n"
        f"{{\n"
        f"{INDENT}for (int j = tid; j < {n_elems}; j += blockDim.x) {{\n"
        f"{INDENT}{INDENT}const float x   = {load_expr};\n"
        f"{INDENT}{INDENT}const float out = {compute_expr};\n"
        f"{INDENT}{INDENT}{store_dst} = {store_val};\n"
        f"{INDENT}}}\n"
        f"}}"
    )
