"""Inline (Schedule.W) codegen for indexer ``Kron`` ops.

Per-block tiles are already loaded as 3D ``(W, C, D)`` blocks in fp32
(``W`` is the workload chunk size, or ``1`` for ``BATCHED`` inputs that
broadcast along the workload axis). For inputs

    x_block : (Wx, x1, x2)
    y_block : (Wy, y1, y2)

we form, axis by axis:

  * For an axis in ``op.dim`` we add a singleton dim — after the size in
    ``x``, before the size in ``y`` — so the broadcast multiply realizes
    the outer product.
  * For an axis NOT in ``op.dim`` we keep the original size; broadcasting
    is the regular elementwise rule (sizes must match or one is 1).

After the multiply, the (size_x, size_y) pairs along the Kron axes are
collapsed back to ``size_x * size_y`` in row-major order — which is
exactly the Kronecker layout::

    out[s, ..., i*y_a + j, ...] = x[s, ..., i, ...] * y[s, ..., j, ...]

Examples (with workload-axis broadcast omitted for clarity):

  * ``dim == (1, 2)`` — full Kron::

        x : (x1, 1,  x2, 1)
        y : (1,  y1, 1,  y2)
        out shape: (x1*y1, x2*y2)

  * ``dim == (1,)`` — row-Kron, last axis elementwise::

        x : (x1, 1,  d)
        y : (1,  y1, d)
        out shape: (x1*y1, d)

  * ``dim == (2,)`` — col-Kron, mid axis elementwise::

        x : (c, x2, 1)
        y : (c, 1,  y2)
        out shape: (c, x2*y2)
"""

from typing import List

from ..graph import Graph
from ...context import Context
from ....abs import FORMAT
from ...kron import Kron


def _block_leading_dim(t, ctx: Context) -> int:
    """Leading dim of a loaded block: 1 for BATCHED, W for everything else."""
    return 1 if t._format == FORMAT.BATCHED else ctx.workload_chunk_size


def _shape_str(parts: List[int]) -> str:
    """Render a Triton shape tuple literal (always >= 1 element here)."""
    return "(" + ", ".join(str(p) for p in parts) + ")"


def generate_kron_impl(graph: Graph, op_id: int, ctx: Context) -> str:
    input_ids = graph.op_to_input_tensor_list[op_id]
    assert len(input_ids) == 2, (
        f"Kron expects 2 inputs, got {len(input_ids)}"
    )
    input_tensor_id_0, input_tensor_id_1 = input_ids
    output_tensor_id = graph.op_to_output_tensor_list[op_id]
    op = graph.op_list[op_id]
    assert issubclass(op.__class__, Kron), (
        f"Expected a Kron op, got {op.__class__.__name__}"
    )

    t_x = graph.tensor_list[input_tensor_id_0]
    t_y = graph.tensor_list[input_tensor_id_1]
    t_o = graph.tensor_list[output_tensor_id]

    x = f"tensor_{input_tensor_id_0}_block"
    y = f"tensor_{input_tensor_id_1}_block"
    z = f"tensor_{output_tensor_id}_block"

    Wx = _block_leading_dim(t_x, ctx)
    Wy = _block_leading_dim(t_y, ctx)
    Wo = _block_leading_dim(t_o, ctx)

    # Build per-input reshape targets and the output shape, axis by axis.
    # Each Kron axis contributes a (size_x, 1) pair to ``shape_x`` and a
    # (1, size_y) pair to ``shape_y`` so the broadcast multiply realizes
    # the outer product. Non-Kron axes pass through.
    shape_x: List[int] = [Wx]
    shape_y: List[int] = [Wy]
    shape_o: List[int] = [Wo]

    for axis in (1, 2):
        sx, sy = t_x.shape[axis], t_y.shape[axis]
        if axis in op.dim:
            shape_x.extend([sx, 1])
            shape_y.extend([1, sy])
            shape_o.append(sx * sy)
        else:
            # Elementwise axis: profile() has already enforced that the
            # sizes match or one is 1; the broadcast shape is just max.
            shape_x.append(sx)
            shape_y.append(sy)
            shape_o.append(max(sx, sy))

    x_reshaped = f"tl.reshape({x}, {_shape_str(shape_x)})"
    y_reshaped = f"tl.reshape({y}, {_shape_str(shape_y)})"

    return (
        f"{z} = tl.reshape({x_reshaped} * {y_reshaped}, "
        f"{_shape_str(shape_o)})"
    )
