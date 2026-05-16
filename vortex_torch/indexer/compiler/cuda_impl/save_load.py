"""CUDA codegen for the indexer ``Save`` and ``Load`` ops.

Both ops are conceptual format converters between cache-side PAGED
storage and indexer-side RAGGED storage:

  * **Save** copies a RAGGED input tile into a PAGED output cache field.
  * **Load** copies a PAGED input cache field into a RAGGED output tile.

In the Triton backend these compile to a single Python-level rename
(``tensor_O_block = tensor_X_block``) because the per-program tile
*is* the register set — no copy needed. In the CUDA backend the
per-tensor smem tiles are physically separate buffers
(:func:`generate_initialization_str` allocates one per ``tensor_list``
entry), so Save and Load have to do an actual smem-to-smem copy.

The copy is a strided ``for (j = tid; j < N; j += blockDim.x)`` over
the flat element count ``N = workload_chunk_size * shape[1] * shape[2]``.
Input and output have identical inner shapes (the cache field and the
ragged tile share ``(shape[1], shape[2])``); only the storage FORMAT
differs, and that gating happens at the kernel-template level in
:func:`generate_load_tensor_str` / :func:`generate_store_tensor_str`
when the cache field crosses the global-memory boundary.

Same-dtype copy is a direct memory move (no cast); cross-dtype copy
goes through fp32 via :mod:`cuda_impl.dtype_cast`. (Save can target a
narrower or different cache dtype than the indexer's compute dtype.)

The leading ``__syncthreads()`` is correctness-required: the input
tile may have been populated by load_tensor_str (cross-thread writes
on multi-page PAGED inputs) or by a previous compute op with a
different per-thread access pattern.
"""

from typing import Tuple

from ..graph import Graph
from ...context import Context
from ...save_load import Save, Load
from ....abs import FORMAT
from ....utils import INDENT

from .dtype_cast import cast_smem_to_float, cast_float_to_smem


def _emit_smem_to_smem_copy(
    in_tid: int, in_t, out_tid: int, out_t, ctx: Context,
) -> str:
    """Emit a strided cooperative copy of one workload tile of
    ``tensor_<in_tid>_smem`` into ``tensor_<out_tid>_smem``.

    Flat element count is ``leading * inner_numel`` where ``leading``
    is ``1`` for BATCHED tiles and ``workload_chunk_size`` otherwise
    (matching the layout chosen by
    :func:`generate_initialization_str`). The flat-indexed write
    ``(&tensor_<id>_smem[0][0][0])[j]`` is legal because the 3D
    ``__shared__`` array is contiguous.
    """
    assert in_t.shape[1] == out_t.shape[1] and in_t.shape[2] == out_t.shape[2], (
        f"save_load: inner shapes must match — got input {in_t.shape}, "
        f"output {out_t.shape}"
    )
    # Save: RAGGED -> PAGED. Load: PAGED -> RAGGED. Both formats use
    # the chunk-leading layout, so leading is the same for input and
    # output. Defensive assert against a hypothetical BATCHED variant.
    in_leading  = 1 if in_t._format  == FORMAT.BATCHED else ctx.workload_chunk_size
    out_leading = 1 if out_t._format == FORMAT.BATCHED else ctx.workload_chunk_size
    assert in_leading == out_leading, (
        f"save_load: leading-dim mismatch (in={in_leading}, out={out_leading})"
    )
    n_total = in_leading * in_t.shape[1] * in_t.shape[2]

    src = f"(&tensor_{in_tid}_smem[0][0][0])[j]"
    dst = f"(&tensor_{out_tid}_smem[0][0][0])[j]"

    if in_t.dtype is out_t.dtype:
        body_inner = f"{dst} = {src};"
    else:
        load_expr  = cast_smem_to_float(src, in_t.dtype)
        store_expr = cast_float_to_smem("val", out_t.dtype)
        body_inner = (
            f"const float val = {load_expr};\n"
            f"{INDENT}{INDENT}{dst} = {store_expr};"
        )

    return (
        f"__syncthreads();\n"
        f"{{\n"
        f"{INDENT}for (int j = tid; j < {n_total}; j += blockDim.x) {{\n"
        f"{INDENT}{INDENT}{body_inner}\n"
        f"{INDENT}}}\n"
        f"}}"
    )


def generate_save_impl(graph: Graph, op_id: int, ctx: Context) -> str:
    """CUDA codegen for ``Save`` (RAGGED → PAGED).

    The actual PAGED-side global write is done by
    :func:`generate_store_tensor_str` against the output tile this op
    populates; here we only mirror the input tile into the output
    tile (with optional dtype cast). Save is a side-effect op so its
    output sits in ``side_effect_op_ids`` / ``output_tensor_ids`` and
    the store codegen knows to flush it.
    """
    input_tensor_id  = graph.op_to_input_tensor_list[op_id][0]
    output_tensor_id = graph.op_to_output_tensor_list[op_id]
    op = graph.op_list[op_id]
    assert issubclass(op.__class__, Save), (
        f"Expected a Save op, got {op.__class__.__name__}"
    )

    t_x = graph.tensor_list[input_tensor_id]
    t_o = graph.tensor_list[output_tensor_id]
    assert t_x._format == FORMAT.RAGGED, (
        f"Save: expected RAGGED input, got {t_x._format!r}"
    )
    assert t_o._format == FORMAT.PAGED, (
        f"Save: expected PAGED output, got {t_o._format!r}"
    )

    return _emit_smem_to_smem_copy(
        input_tensor_id, t_x, output_tensor_id, t_o, ctx,
    )


def generate_load_impl(graph: Graph, op_id: int, ctx: Context) -> str:
    """CUDA codegen for ``Load`` (PAGED → RAGGED).

    The PAGED-side global read happens in
    :func:`generate_load_tensor_str` against the input tile; this op
    just mirrors the populated PAGED tile into the RAGGED output tile.
    """
    input_tensor_id  = graph.op_to_input_tensor_list[op_id][0]
    output_tensor_id = graph.op_to_output_tensor_list[op_id]
    op = graph.op_list[op_id]
    assert issubclass(op.__class__, Load), (
        f"Expected a Load op, got {op.__class__.__name__}"
    )

    t_x = graph.tensor_list[input_tensor_id]
    t_o = graph.tensor_list[output_tensor_id]
    assert t_x._format == FORMAT.PAGED, (
        f"Load: expected PAGED input, got {t_x._format!r}"
    )
    assert t_o._format == FORMAT.RAGGED, (
        f"Load: expected RAGGED output, got {t_o._format!r}"
    )

    return _emit_smem_to_smem_copy(
        input_tensor_id, t_x, output_tensor_id, t_o, ctx,
    )
