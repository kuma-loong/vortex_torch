"""CUDA codegen for the Schedule.W ``Reduce`` op (dim ∈ {1, 2}).

Reduces one inner axis of the per-workload tile ``[leading, D_0, D_1]``
to size 1, preserving the leading dim:

  * ``dim == 1`` →  ``out[chunk_i, 0, d_o]   = REDUCE_{k ∈ [0, D_0)} in[chunk_i, k, d_o]``
                    Output shape ``[leading, 1, D_1]``.
  * ``dim == 2`` →  ``out[chunk_i, c_o, 0]   = REDUCE_{k ∈ [0, D_1)} in[chunk_i, c_o, k]``
                    Output shape ``[leading, D_0, 1]``.

The cross-row form (``dim == 0``, Schedule.S, RAGGED → BATCHED) is
**not** handled here — it's reused from
:mod:`triton_impl.reduce.generate_reduce_dim0_impl` via the registry.

Thread mapping is per-output-element: each thread claims one
``(chunk_i, c_o, d_o)`` output position and sequentially sweeps the
reduction axis in fp32 registers. No cross-thread coordination — the
output positions are independent. This keeps the codegen simple at
the cost of underutilization when the output is smaller than the
block size (e.g. ``dim == 2`` with small ``D_0``); warp-level shuffle
reductions are a future optimization.

Compute precision is fp32 throughout the accumulation (matching
Triton, which promotes inputs via ``.to(tl.float32)`` before reducing).
Storage-dtype casts on the smem boundary use the shared helpers in
:mod:`cuda_impl.dtype_cast`.
"""

from typing import Tuple

from ..graph import Graph
from ...context import Context
from ...reduce import Reduce
from ....abs import FORMAT
from ....utils import INDENT, ReduceType, Schedule

from .dtype_cast import cast_smem_to_float, cast_float_to_smem


def _reduce_init_update_finalize(
    reduce_type: ReduceType, reduce_len: int,
) -> Tuple[str, str, str]:
    """Return ``(init_expr, update_expr, finalize_expr)`` for a given
    ``ReduceType``.

    ``update_expr`` is in terms of ``acc`` (running fp32 accumulator)
    and ``val`` (the just-loaded fp32 input element). ``finalize_expr``
    is in terms of ``acc`` and is the empty string when no
    post-reduction step is needed.
    """
    if reduce_type == ReduceType.Sum:
        return "0.0f", "acc + val", ""
    if reduce_type == ReduceType.Mean:
        # Multiplication by the precomputed reciprocal mirrors Triton's
        # ``* (1.0 / D)``; faster than ``acc / D`` and matches Triton's
        # rounding profile on non-power-of-two ``D``.
        return "0.0f", "acc + val", f"acc * {1.0 / reduce_len}f"
    if reduce_type == ReduceType.Max:
        # ``-1e30f`` (not ``-INFINITY``) matches Triton's choice — a
        # finite sentinel so a downstream softmax / arithmetic op
        # doesn't propagate NaN if it sees the init value.
        return "-1.0e30f", "fmaxf(acc, val)", ""
    if reduce_type == ReduceType.Min:
        return "1.0e30f", "fminf(acc, val)", ""
    if reduce_type == ReduceType.L2Norm:
        return "0.0f", "acc + val * val", "sqrtf(acc)"
    raise NotImplementedError(
        f"cuda_impl.reduce: unsupported reduce_type {reduce_type!r}"
    )


def generate_reduce_impl(graph: Graph, op_id: int, ctx: Context) -> str:
    """Emit the per-block CUDA C++ source for a Schedule.W ``Reduce``.

    Layout::

        __syncthreads();
        {
            for (int j_out = tid; j_out < N_out; j_out += blockDim.x) {
                const int chunk_i = j_out / (out_C * out_D);
                const int c_o     = (j_out / out_D) % out_C;
                const int d_o     =  j_out % out_D;

                float acc = <init>;
                for (int k = 0; k < reduce_len; ++k) {
                    const float val = <cast smem -> float>(
                        tensor_X_smem[chunk_i][<c_in>][<d_in>]
                    );
                    acc = <update>;
                }
                <finalize: ``acc = ...;``  if non-empty>;
                tensor_O_smem[chunk_i][c_o][d_o] = <cast float -> smem>(acc);
            }
        }

    ``<c_in>, <d_in>`` is ``(k, d_o)`` when ``dim == 1`` and
    ``(c_o, k)`` when ``dim == 2`` — picked at codegen time so the
    generated subscript has exactly one sweep variable.
    """
    input_tensor_id  = graph.op_to_input_tensor_list[op_id][0]
    output_tensor_id = graph.op_to_output_tensor_list[op_id][0]
    op = graph.op_list[op_id]
    assert issubclass(op.__class__, Reduce), (
        f"Expected a Reduce op, got {op.__class__.__name__}"
    )
    assert op.dim in (1, 2), (
        f"cuda_impl.reduce.generate_reduce_impl: dim must be in {{1, 2}}; "
        f"got dim={op.dim}. dim=0 is Schedule.S — see the Triton "
        f"``generate_reduce_dim0_impl`` reused via the registry."
    )
    assert op.schedule == Schedule.W, (
        f"cuda_impl.reduce.generate_reduce_impl: expected Schedule.W, got "
        f"{op.schedule}"
    )

    t_x = graph.tensor_list[input_tensor_id]
    t_o = graph.tensor_list[output_tensor_id]

    # Output preserves the leading-dim layout (BATCHED iff input
    # BATCHED), per :class:`Reduce.profile`.
    leading = 1 if t_o._format == FORMAT.BATCHED else ctx.workload_chunk_size
    out_C, out_D = t_o.shape[1], t_o.shape[2]
    n_out = leading * out_C * out_D

    # Reduction axis length and per-iteration input-subscript shape.
    if op.dim == 1:
        reduce_len = t_x.shape[1]      # D_0
        c_in_expr  = "k"
        d_in_expr  = "d_o"
    else:                              # op.dim == 2
        reduce_len = t_x.shape[2]      # D_1
        c_in_expr  = "c_o"
        d_in_expr  = "k"

    init_expr, update_expr, finalize_expr = _reduce_init_update_finalize(
        op.reduce_type, reduce_len,
    )

    in_subscript  = f"tensor_{input_tensor_id}_smem[chunk_i][{c_in_expr}][{d_in_expr}]"
    out_subscript = f"tensor_{output_tensor_id}_smem[chunk_i][c_o][d_o]"

    load_expr  = cast_smem_to_float(in_subscript, t_x.dtype)
    store_expr = cast_float_to_smem("acc", t_o.dtype)

    # Optional post-loop finalize line (mean reciprocal, L2 sqrt, ...).
    finalize_line = (
        f"{INDENT}{INDENT}acc = {finalize_expr};\n" if finalize_expr else ""
    )

    return (
        f"__syncthreads();\n"
        f"{{\n"
        f"{INDENT}for (int j_out = tid; j_out < {n_out}; j_out += blockDim.x) {{\n"
        f"{INDENT}{INDENT}const int chunk_i = j_out / {out_C * out_D};\n"
        f"{INDENT}{INDENT}const int c_o     = (j_out / {out_D}) % {out_C};\n"
        f"{INDENT}{INDENT}const int d_o     =  j_out % {out_D};\n"
        f"\n"
        f"{INDENT}{INDENT}float acc = {init_expr};\n"
        f"{INDENT}{INDENT}for (int k = 0; k < {reduce_len}; ++k) {{\n"
        f"{INDENT}{INDENT}{INDENT}const float val = {load_expr};\n"
        f"{INDENT}{INDENT}{INDENT}acc = {update_expr};\n"
        f"{INDENT}{INDENT}}}\n"
        f"{finalize_line}"
        f"{INDENT}{INDENT}{out_subscript} = {store_expr};\n"
        f"{INDENT}}}\n"
        f"}}"
    )
