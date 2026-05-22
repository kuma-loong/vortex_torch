from ..graph import Graph
from typing import Dict, Tuple, Callable
from ...context import Context
from ....abs import FORMAT
from .dtype_cast import compute_tl_dtype as _compute_tl_dtype


# Minimum M/N/K for a tl.dot to be worth dispatching to tensor cores.
# Below this the MMA setup cost dominates and the broadcast-multiply-sum
# form is faster; Triton also rejects dots smaller than the MMA tile.
_TC_MIN_DIM = 16


def generate_gemm_impl(graph: Graph, op_id: int, ctx: Context) -> str:
    r"""Schedule.W GeMM: ``O[s, i, j] = sum_k Y[s, i, k] * X[s_or_b, j, k]``.

    Operands (3D per-workload blocks):

      * ``X`` = input 0 — ``(Wx, Nx, K)`` (``Wx == 1`` when BATCHED).
      * ``Y`` = input 1 — ``(W,  Ny, K)``.
      * ``O`` = output    — ``(W,  Ny, Nx)``.

    Two codegen paths:

    * **Default (fp32 compute).** The historical broadcast-multiply +
      ``tl.sum`` form, all in fp32.

    * **Tensor-core (``ctx.use_tensor_core``).** Compute blocks are bf16.
      The reduction over ``K`` is the "accumulation" that must stay
      fp32: the broadcast path casts the product to fp32 before the
      ``tl.sum`` and downcasts the result to bf16. When the operand
      tiles are MMA-friendly we instead emit ``tl.dot`` — bf16 inputs,
      fp32 accumulator (Triton's default), result downcast to bf16.

      MMA-friendliness is judged on the **padded** tile dims, not the
      logical ones. Loaded blocks are already padded to the next pow2
      on axes 1 / 2 (``vTensor.padded_shape``) and the padding lanes are
      zero-filled by the masked / boundary-checked loads, so a logical
      dim of e.g. 12 lands in a clean 16-wide tile that ``tl.dot`` can
      consume directly: the zero K-lanes contribute nothing to the
      contraction, and the zero N/M lanes produce zero output columns /
      rows that the store masks out (identical to what the broadcast
      path already does). The GEMV-shaped ops (logical ``Nx == 1`` →
      padded 1) stay below the MMA minimum and take the fp32-accumulated
      broadcast path.
    """
    input_tensor_id0, input_tensor_id1 = graph.op_to_input_tensor_list[op_id]
    output_tensor_id = graph.op_to_output_tensor_list[op_id][0]
    t_i0 = graph.tensor_list[input_tensor_id0]   # X : (Wx, Nx, K)
    t_i1 = graph.tensor_list[input_tensor_id1]   # Y : (W,  Ny, K)
    t_o = graph.tensor_list[output_tensor_id]    # O : (W,  Ny, Nx)

    x = f"tensor_{input_tensor_id0}_block"
    y = f"tensor_{input_tensor_id1}_block"
    o = f"tensor_{output_tensor_id}_block"

    use_tc = getattr(ctx, "use_tensor_core", False)
    cdt = _compute_tl_dtype(ctx)

    Nx = t_i0.shape[1]
    Ny = t_i1.shape[1]
    K = t_i0.shape[2]

    # ---- Tensor-core tl.dot fast path (only when MMA-friendly) -------
    # The common case in these flows is a BATCHED X (one shared operand
    # reused across the whole workload chunk). Rather than broadcasting
    # X up to ``W`` copies (which would materialise a W-fold tile in
    # shared memory), we *flatten* the workload and ``Ny`` axes of Y
    # into a single matmul row dim — the "flatten dimensions" trick:
    #
    #   O[w, i, j] = sum_k Y[w, i, k] * X[0, j, k]
    #             => O2d[(w*Ny + i), j] = sum_k Y2d[(w*Ny + i), k] * X2d[j, k]
    #             => O2d = Y2d @ X2d^T,   Y2d:(W*Ny, K)  X2d:(Nx, K)
    #
    # That keeps a single shared X tile and uses one clean 2D MMA with
    # M = W*Ny, N = Nx, K = K.
    #
    # The reshape/trans/dot all run on the **padded** tile dims — the
    # block in registers is ``padded_shape``, the padding is zero (so
    # the dot stays numerically exact), and the trailing zero output
    # lanes are masked at store. This lets a logical dim that pads up to
    # the MMA minimum (e.g. 12 -> 16) still use tensor cores. Padded K
    # must agree between the two operands (it does: the logical K's match
    # per profile() and pow2-rounding is deterministic).
    W = ctx.workload_chunk_size
    Nx_p = t_i0.padded_shape[1]
    Ny_p = t_i1.padded_shape[1]
    Kx_p = t_i0.padded_shape[2]
    Ky_p = t_i1.padded_shape[2]
    can_dot = (
        use_tc
        and t_i0._format == FORMAT.BATCHED
        and Kx_p == Ky_p
        and Kx_p >= _TC_MIN_DIM
        and Nx_p >= _TC_MIN_DIM
        and (W * Ny_p) >= _TC_MIN_DIM
    )
    if can_dot:
        y2d = f"tl.reshape({y}, ({W * Ny_p}, {Ky_p})).to(tl.bfloat16)"
        x2d_t = f"tl.trans(tl.reshape({x}, ({Nx_p}, {Kx_p})).to(tl.bfloat16), 1, 0)"
        impl_lines = [
            f"{o} = tl.reshape(tl.dot({y2d}, {x2d_t}), ({W}, {Ny_p}, {Nx_p})).to({cdt})",
        ]
        return "\n".join(impl_lines)

    # ---- Broadcast-multiply + tl.sum (default / GEMV) ----------------
    if t_i0._format == FORMAT.BATCHED and t_i1.shape[1] == 1:
        prod = f"({x} * {y})"
        if use_tc:
            prod = f"{prod}.to(tl.float32)"
        red = f"tl.sum({prod}, axis=2)[:,None,:]"
    else:
        prod = f"({x}[:,None,:,:] * {y}[:,:,None,:])"
        if use_tc:
            prod = f"{prod}.to(tl.float32)"
        red = f"tl.sum({prod}, axis=3)"

    if use_tc:
        red = f"({red}).to({cdt})"

    return f"{o} = {red}"
