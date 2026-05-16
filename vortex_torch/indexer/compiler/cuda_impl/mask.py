"""CUDA codegen for the ``MaskSlice`` op.

Position-dependent mask over one inner axis of the per-workload tile
``[leading, C, D]`` (``leading = workload_chunk_size`` for RAGGED /
PAGED, ``1`` for BATCHED). For ``dim == 1`` (axis :math:`D_0`) writes

    alpha   if start <= c < end
    beta    otherwise

uniformly across the chunk and ``D`` axes. For ``dim == 2`` (axis
:math:`D_1`) the test is on ``d`` instead.

Mirrors :mod:`triton_impl.mask` op-for-op; translated from Triton's
``tl.where(_mask[broadcast], alpha, beta)`` (which broadcasts a
length-``D_dim`` boolean across the other two axes) to a per-thread
strided write where each thread independently checks its own
``(c, d)`` against ``[start, end)``.

The input tensor is not read — :class:`indexer.mask.MaskSlice` is a
pure position-based writer; the input edge exists only so the
fuser keeps the op adjacent to its neighbours. The codegen still
emits a leading ``__syncthreads()`` to stay uniform with the rest
of the Schedule.W compute ops — any previous op may have written
``tensor_<output>_smem`` (e.g. an earlier iteration's compute or a
load that targeted the same tile under aliasing) and we want a
clean barrier before the position-dependent writes start.
"""

from ..graph import Graph
from ...context import Context
from ...mask import MaskSlice
from ....abs import FORMAT
from ....utils import INDENT

from .dtype_cast import cast_float_to_smem


def generate_mask_slice_impl(graph: Graph, op_id: int, ctx: Context) -> str:
    """Emit the per-block CUDA C++ source for a ``MaskSlice`` op.

    Layout::

        {
            for (int j_out = tid; j_out < N_out; j_out += blockDim.x) {
                const int chunk_i = j_out / (C * D);
                const int c       = (j_out / D) % C;
                const int d       =  j_out % D;
                const float val = (<pos> >= start && <pos> < end)
                                      ? alpha : beta;
                tensor_<id>_smem[chunk_i][c][d] = <cast float -> smem>(val);
            }
        }

    Where ``<pos>`` is ``c`` if ``op.dim == 1`` and ``d`` if
    ``op.dim == 2`` — picked at codegen time so the generated branch
    is a single integer compare per thread.
    """
    input_tensor_id  = graph.op_to_input_tensor_list[op_id][0]
    output_tensor_id = graph.op_to_output_tensor_list[op_id]
    op = graph.op_list[op_id]

    assert issubclass(op.__class__, MaskSlice), (
        f"Expected MaskSlice, got {op.__class__.__name__}"
    )

    t_o = graph.tensor_list[output_tensor_id]
    C, D = t_o.shape[1], t_o.shape[2]
    leading = 1 if t_o._format == FORMAT.BATCHED else ctx.workload_chunk_size
    n_out = leading * C * D

    # The position variable for the slice test. ``op.dim`` is asserted
    # to be in ``{1, 2}`` by :class:`MaskSlice.__init__`.
    pos_var = "c" if op.dim == 1 else "d"

    val_expr = (
        f"({pos_var} >= {op.start} && {pos_var} < {op.end}) "
        f"? {op.alpha}f : {op.beta}f"
    )

    store_dst = f"tensor_{output_tensor_id}_smem[chunk_i][c][d]"
    store_val = cast_float_to_smem("val", t_o.dtype)

    return (
        f"__syncthreads();\n"
        f"{{\n"
        f"{INDENT}for (int j_out = tid; j_out < {n_out}; j_out += blockDim.x) {{\n"
        f"{INDENT}{INDENT}const int chunk_i = j_out / {C * D};\n"
        f"{INDENT}{INDENT}const int c       = (j_out / {D}) % {C};\n"
        f"{INDENT}{INDENT}const int d       =  j_out % {D};\n"
        f"{INDENT}{INDENT}const float val = {val_expr};\n"
        f"{INDENT}{INDENT}{store_dst} = {store_val};\n"
        f"{INDENT}}}\n"
        f"}}"
    )
