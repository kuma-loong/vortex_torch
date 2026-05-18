"""Inline (Schedule.W) codegen for cache ``ReduceInterleave`` ops.

The surrounding kernel (see :mod:`.kernel_gen`) already loads the input
as a 2D ``(D0, D1)`` block in fp32 and stores the output block.
This op's codegen reshapes the input block so that consecutive groups
of ``k`` elements along the reduced axis become an inner axis, then
emits the reduction expression along that inner axis.

Axis mapping:
  * Cache logical tensor is ``[B, D0, D1]`` with block-level programs
    reading the (D0, D1) slice of a single block.
  * ``op.dim == 1`` → reduce over the logical ``D0`` axis →
    reshape ``(D0, D1)`` → ``(D0/k, k, D1)``, reduce on axis 1.
  * ``op.dim == 2`` → reduce over the logical ``D1`` axis →
    reshape ``(D0, D1)`` → ``(D0, D1/k, k)``, reduce on axis 2.

The output block is 2D and its shape matches ``(out_D0, out_D1)`` as
allocated by the op's ``profile``.
"""

from ..graph import Graph
from ...context import Context
from ....utils import ReduceType
from ...reduce_interleave import ReduceInterleave


def generate_reduce_interleave_impl(graph: Graph, op_id: int, ctx: Context) -> str:
    input_tensor_id = graph.op_to_input_tensor_list[op_id][0]
    output_tensor_id = graph.op_to_output_tensor_list[op_id][0]
    op = graph.op_list[op_id]
    assert issubclass(op.__class__, ReduceInterleave), (
        f"Expected a ReduceInterleave op, got {op.__class__.__name__}"
    )

    t_i = graph.tensor_list[input_tensor_id]
    t_o = graph.tensor_list[output_tensor_id]
    x = f"tensor_{input_tensor_id}_block"
    y = f"tensor_{output_tensor_id}_block"

    # ``tl.reshape`` axis sizes must be pow2; the loaded block carries
    # padded inner dims, so the (D0/k, k, D1) split must use the padded
    # extents. Real-shape divisibility against ``k`` is still the
    # correctness contract — generic non-pow2-divisible-by-k inputs
    # require an inter-block repack we haven't implemented.
    assert not (t_i.needs_padding() or t_o.needs_padding()), (
        f"ReduceInterleave under non-pow2 inner dims is not yet supported "
        f"(input {t_i.shape!r}, output {t_o.shape!r})."
    )
    D0, D1 = t_i.shape[1], t_i.shape[2]
    k = op.k

    # Build the (3D) reshape target and the axis along which the inner
    # group of k elements gets reduced. ``D0``/``D1`` and ``k`` are all
    # constexpr at codegen time, so the shape tuple is a static literal.
    if op.dim == 1:
        assert D0 % k == 0, (
            f"reduce_interleave codegen: D0={D0} not divisible by k={k}"
        )
        new_shape = f"({D0 // k}, {k}, {D1})"
        axis = 1
    else:  # op.dim == 2
        assert D1 % k == 0, (
            f"reduce_interleave codegen: D1={D1} not divisible by k={k}"
        )
        new_shape = f"({D0}, {D1 // k}, {k})"
        axis = 2

    x_grouped = f"tl.reshape({x}, {new_shape})"

    if op.reduce_type == ReduceType.Sum:
        return f"{y} = tl.sum({x_grouped}, axis={axis})"
    if op.reduce_type == ReduceType.Max:
        return f"{y} = tl.max({x_grouped}, axis={axis})"
    if op.reduce_type == ReduceType.Min:
        return f"{y} = tl.min({x_grouped}, axis={axis})"
    if op.reduce_type == ReduceType.L2Norm:
        # L2 (not RMS): sqrt(sum(x*x)). Square first, then reshape, so
        # the codegen emits a single reshape rather than two.
        x_sq_grouped = f"tl.reshape({x} * {x}, {new_shape})"
        return f"{y} = tl.sqrt(tl.sum({x_sq_grouped}, axis={axis}))"
    if op.reduce_type == ReduceType.Mean:
        # Pre-divide by the group size (constant at codegen time).
        inv_k = 1.0 / float(k)
        return f"{y} = tl.sum({x_grouped}, axis={axis}) * ({inv_k})"

    raise NotImplementedError(
        f"ReduceInterleave type {op.reduce_type!r} is not yet wired in cache codegen"
    )
