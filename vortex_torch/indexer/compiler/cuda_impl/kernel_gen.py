"""CUDA kernel generator for fused indexer subgraphs.

Mirrors the Triton backend layout: a subgraph is one of:

  - ``Schedule.W`` (workload-scheduled): emit a CUDA ``__global__``
    kernel and a host-side C++ launcher; embed both as a Python raw
    string that ``torch.utils.cpp_extension.load_inline`` JIT-compiles
    at module-import time; wrap the compiled launcher in a Python
    function that forwards tensor args + scalars pulled from ``ctx``.
  - non-W: a single op whose impl function is inlined directly into a
    plain Python wrapper (no kernel launch needed). The op's
    ``get_impl_func`` does all codegen.

Step-by-step scaffold. Only the top-level dispatch
(:func:`generate_cuda_impl`), the W-scheduled wrapper
(:func:`_generate_w_impl`), and the non-W wrapper
(:func:`_generate_non_w_impl`) are filled in. The kernel-body codegen
helpers (initialization, load, store, computation, kernel/launcher
source) are stubbed with ``pass`` and will be added in follow-up
commits — calling :func:`_generate_w_impl` before they are implemented
will raise ``TypeError`` from the still-empty helpers.

Public API: :func:`generate_cuda_impl`. The other ``generate_*`` /
helper symbols are exported for in-tree debugging and tests.
"""
from __future__ import annotations

from typing import Iterable, List, Tuple

import torch

from ..graph import Graph
from ...context import Context
from ....abs import FORMAT
from ....utils import Schedule, INDENT
from .register import get_impl_func
from .dtype_cast import cast_float_to_smem


# --------------------------------------------------------------------------- #
# Torch dtype -> C++ type tables
# --------------------------------------------------------------------------- #
#
# Each entry is ``(aten_t, cuda_t)``:
#   * ``aten_t`` is the type used for ``tensor.data_ptr<...>()`` (the ATen-side
#     surface; e.g. ``at::Half`` for fp16, ``at::BFloat16`` for bf16). It is
#     the only type torch's ``data_ptr<T>`` template instantiates for those
#     half-precision dtypes.
#   * ``cuda_t`` is the type the generated kernel signature uses, picked so
#     in-kernel arithmetic uses the native CUDA intrinsic types
#     (``__half`` / ``__nv_bfloat16``). Where ``aten_t != cuda_t`` the
#     launcher emits a ``reinterpret_cast<cuda_t*>`` — those types are
#     binary-compatible so the cast is safe.
#
# FP8 dtypes are rebound to ``uint8`` on the Python side via
# ``.view(torch.uint8)`` before the launcher call, mirroring the Triton
# backend convention; their entries here are uint8 for both sides.
_TORCH_TO_CPP_TYPES = {
    torch.float32:       ("float",         "float"),
    torch.float16:       ("at::Half",      "__half"),
    torch.bfloat16:      ("at::BFloat16",  "__nv_bfloat16"),
    torch.int32:         ("int32_t",       "int32_t"),
    torch.int64:         ("int64_t",       "int64_t"),
    torch.uint8:         ("uint8_t",       "uint8_t"),
    torch.int8:          ("int8_t",        "int8_t"),
    torch.float8_e5m2:   ("uint8_t",       "uint8_t"),
    torch.float8_e4m3fn: ("uint8_t",       "uint8_t"),
}


def _is_fp8(t) -> bool:
    """True iff the tensor's dtype is one of the fp8 variants the framework
    accepts. Matches :func:`triton_impl.dtype_cast.is_fp8` so the cuda and
    triton backends agree on which tensors get the ``.view(torch.uint8)``
    rebind treatment.
    """
    return t.dtype in (torch.float8_e5m2, torch.float8_e4m3fn)


def _cpp_kernel_ptr_type(dtype) -> str:
    """Return the kernel-side pointer type (``cuda_t*``) for a torch dtype."""
    try:
        return f"{_TORCH_TO_CPP_TYPES[dtype][1]}*"
    except KeyError:
        raise NotImplementedError(
            f"cuda_impl.kernel_gen: unsupported tensor dtype {dtype!r}"
        )


def _cpp_kernel_value_type(dtype) -> str:
    """Return the kernel-side value type (``cuda_t``) for a torch dtype.

    Sibling of :func:`_cpp_kernel_ptr_type`. Used by anything that
    declares a value-typed local / shared-memory tile rather than a
    pointer.
    """
    try:
        return _TORCH_TO_CPP_TYPES[dtype][1]
    except KeyError:
        raise NotImplementedError(
            f"cuda_impl.kernel_gen: unsupported tensor dtype {dtype!r}"
        )




def _cpp_data_ptr_expr(name: str, dtype) -> str:
    """``<name>.data_ptr<aten_t>()``, ``reinterpret_cast``-ed to ``cuda_t*``
    when the two differ (fp16 / bf16). Used by the launcher to forward
    each ``torch::Tensor`` argument as the right pointer type to the
    kernel.
    """
    try:
        aten_t, cuda_t = _TORCH_TO_CPP_TYPES[dtype]
    except KeyError:
        raise NotImplementedError(
            f"cuda_impl.kernel_gen: unsupported tensor dtype {dtype!r}"
        )
    base = f"{name}.data_ptr<{aten_t}>()"
    if aten_t == cuda_t:
        return base
    return f"reinterpret_cast<{cuda_t}*>({base})"


# Includes shared by every JIT-compiled subgraph TU:
#   * ``<torch/extension.h>``        — ``torch::Tensor``, pybind11 glue.
#   * ``<ATen/cuda/CUDAContext.h>``  — ``at::cuda::getCurrentCUDAStream()``
#     used by the launcher emitted by :func:`generate_cuda_launcher`.
#   * ``<cuda_fp16.h>`` / ``<cuda_bf16.h>`` — ``__half`` / ``__nv_bfloat16``
#     names used in the kernel signature and the launcher's
#     ``reinterpret_cast``s.
#   * ``<cstdint>``                   — ``int32_t``, ``int64_t``, ``uint8_t``.
_CUDA_KERNEL_PREAMBLE = (
    "#include <torch/extension.h>\n"
    "#include <ATen/cuda/CUDAContext.h>\n"
    "#include <cuda_fp16.h>\n"
    "#include <cuda_bf16.h>\n"
    "#include <cstdint>"
)


# --------------------------------------------------------------------------- #
# Small utilities (pure, backend-agnostic)
# --------------------------------------------------------------------------- #

def _has_batched_output(sub_graph) -> bool:
    """True iff the subgraph stores any BATCHED tensor.

    Same role as in the Triton backend: drives the
    ``winfo_is_first_workload_per_batch`` kernel argument / launcher
    binding and the per-workload "first" gate in the store body.
    """
    return any(
        sub_graph.tensor_list[tid]._format == FORMAT.BATCHED
        for tid in sub_graph.output_tensor_ids
    )


def indent_block(text: str, level: int = 1) -> str:
    """Indent every non-blank line in ``text`` by ``level`` ``INDENT`` units."""
    prefix = INDENT * level
    return "\n".join(
        prefix + line if line.strip() else line for line in text.splitlines()
    )


def _join_sections(sections: Iterable[str], fallback: str) -> str:
    """Join non-empty ``sections`` with ``\\n\\n``; return ``fallback`` if all empty."""
    pass


def _iter_kernel_tensors(sub_graph) -> Iterable[Tuple[int, str]]:
    """Yield ``(tensor_id, role)`` for each kernel argument tensor.

    A tensor that's both an input (``Load``) and an output (``Save``)
    appears exactly once, with ``role='output'``. Otherwise role is
    ``'input'`` (input-only) or ``'output'`` (output-only).
    """
    output_set = set(sub_graph.output_tensor_ids)
    for tid in sub_graph.input_tensor_ids:
        if tid in output_set:
            continue
        yield tid, "input"
    for tid in sub_graph.output_tensor_ids:
        yield tid, "output"


# --------------------------------------------------------------------------- #
# Initialization / Load / Store / Computation codegen (CUDA C++)
# --------------------------------------------------------------------------- #

def generate_initialization_str(sub_graph: Graph, ctx: Context) -> str:
    """Declare a static ``__shared__`` tile for **every** tensor in the
    subgraph — inputs, outputs, AND intermediates.

    Iterates :attr:`Graph.tensor_list` directly (one entry per unique
    local tensor id, populated by
    :func:`graph._build_local_graph` from the union of every op's
    input + output tensors). That covers:

      * Subgraph inputs — populated by :func:`generate_load_tensor_str`.
      * Subgraph outputs — drained by :func:`generate_store_tensor_str`.
      * Intermediates — written by one op and consumed by a later op
        within the same subgraph. They never touch global memory; the
        smem tile is their only storage.

    Load+Save tensors (in both input and output sets) appear as a
    single entry in ``tensor_list`` and therefore get exactly one tile,
    just like every other tensor.

    Each tile is sized by the tensor's format:

      * RAGGED / PAGED:
            __shared__ <cuda_t> tensor_<id>_smem
                [workload_chunk_size][t.shape[1]][t.shape[2]];
        Matches Triton's per-workload tile shape — one chunk of
        ragged blocks / paged blocks per iteration. (Intermediates
        from Elementwise / Elementwise_Binary follow this layout when
        they inherit RAGGED format.)
      * BATCHED:
            __shared__ <cuda_t> tensor_<id>_smem
                [1][t.shape[1]][t.shape[2]];
        BATCHED tensors are one slot per ``(batch, head)`` and the
        body only ever touches index ``0`` of the leading dim, so
        collapsing it to ``1`` saves the otherwise-wasted
        ``(workload_chunk_size - 1) * shape[1] * shape[2]`` slots of
        shared memory. Load / compute / store codegens index BATCHED
        tiles via ``tensor_<id>_smem[0][...]`` explicitly.

    The element type is the tensor's **storage** dtype (``__half`` /
    ``__nv_bfloat16`` / ``float`` / ``uint8_t`` / ``int32_t`` /
    ``int8_t``). Storing in the narrow dtype halves the smem footprint
    vs fp32 and matches typical attention-kernel patterns; body codegen
    is responsible for promoting to fp32 in registers during arithmetic
    (see :mod:`cuda_impl.dtype_cast`) and
    for any cooperative zero-initialization a BATCHED accumulator
    might need before the workload loop (followed by a
    ``__syncthreads()`` before the first read).
    """
    lines: List[str] = []
    chunk = ctx.workload_chunk_size

    for tid, t in enumerate(sub_graph.tensor_list):
        cuda_t = _cpp_kernel_value_type(t.dtype)
        leading = 1 if t._format == FORMAT.BATCHED else chunk
        lines.append(
            f"__shared__ {cuda_t} tensor_{tid}_smem"
            f"[{leading}][{t.shape[1]}][{t.shape[2]}];"
        )

    return "\n".join(lines) if lines else "// No initialization required"


def _emit_contig_load(
    tid: int, cuda_t: str, src_offset_expr: str, n_elems: int,
) -> str:
    """Strided block-cooperative copy of ``n_elems`` contiguous elements
    from ``tensor_<tid>_ptr + src_offset_expr`` into
    ``tensor_<tid>_smem`` (flattened).

    Used for BATCHED, RAGGED, and PAGED single-page loads — all three
    map to a contiguous span in global memory and differ only in
    ``src_offset_expr``. The strided pattern
    ``for (j = tid; j < N; j += blockDim.x)`` makes ``N`` (and the
    block_size baked in by nvcc) the only thing that varies with
    ``workload_chunk_size`` — chunk in [16, 32, 64, 128] just scales
    the iteration count per thread.
    """
    return (
        f"{{\n"
        f"{INDENT}auto* dst = &tensor_{tid}_smem[0][0][0];\n"
        f"{INDENT}const {cuda_t}* src = tensor_{tid}_ptr + {src_offset_expr};\n"
        f"{INDENT}for (int j = tid; j < {n_elems}; j += blockDim.x) {{\n"
        f"{INDENT}{INDENT}dst[j] = src[j];\n"
        f"{INDENT}}}\n"
        f"}}"
    )


def _emit_multipage_load(
    tid: int, cuda_t: str, npw: int, nbp: int, block_elems: int,
    zero_expr: str,
) -> str:
    """Page-by-page gather for PAGED inputs spanning multiple pages.

    For each ``p`` in ``[0, num_pages_per_workload)``:

      * If ``p < _page``: look up the start-of-page block index via
        ``indices[ragged_idx_i32 + p * num_blocks_per_page]``, then copy
        ``num_blocks_per_page * block_elems`` contiguous elements from
        that page into ``tensor_<tid>_smem[p * num_blocks_per_page ...]``.
      * Otherwise: zero-fill the same smem slot using ``zero_expr`` so
        compute can read uniformly without a per-page mask.
        ``zero_expr`` must already be a value of the storage dtype —
        the caller builds it via :func:`cuda_impl.dtype_cast.cast_float_to_smem`
        so the half-precision types get the right intrinsic
        (``__float2half_rn(0.0f)`` / ``__float2bfloat16_rn(0.0f)``)
        rather than a constructor call, which PyTorch's cpp_extension
        disables via ``__CUDA_NO_BFLOAT16_CONVERSIONS__``.

    Pages are written contiguously in smem (page p → slots
    ``[p*nbp, (p+1)*nbp)`` of the chunk axis), so compute and store
    index by ``chunk_idx`` and don't need to know the page layout.
    """
    page_elems = nbp * block_elems
    return (
        f"{{\n"
        f"{INDENT}auto* dst = &tensor_{tid}_smem[0][0][0];\n"
        f"{INDENT}for (int p = 0; p < {npw}; ++p) {{\n"
        f"{INDENT}{INDENT}if (p < _page) {{\n"
        f"{INDENT}{INDENT}{INDENT}const int block_idx = "
        f"indices[ragged_idx_i32 + p * {nbp}];\n"
        f"{INDENT}{INDENT}{INDENT}const {cuda_t}* src = "
        f"tensor_{tid}_ptr + block_idx * {block_elems};\n"
        f"{INDENT}{INDENT}{INDENT}for (int j = tid; j < {page_elems}; "
        f"j += blockDim.x) {{\n"
        f"{INDENT}{INDENT}{INDENT}{INDENT}dst[p * {page_elems} + j] = src[j];\n"
        f"{INDENT}{INDENT}{INDENT}}}\n"
        f"{INDENT}{INDENT}}} else {{\n"
        f"{INDENT}{INDENT}{INDENT}for (int j = tid; j < {page_elems}; "
        f"j += blockDim.x) {{\n"
        f"{INDENT}{INDENT}{INDENT}{INDENT}dst[p * {page_elems} + j] = {zero_expr};\n"
        f"{INDENT}{INDENT}{INDENT}}}\n"
        f"{INDENT}{INDENT}}}\n"
        f"{INDENT}}}\n"
        f"}}"
    )


def generate_load_tensor_str(sub_graph: Graph, ctx: Context) -> str:
    """Cooperatively copy every subgraph **input** from global memory
    into its ``__shared__`` tile.

    Iterates :attr:`Graph.input_tensor_ids` directly — a Load+Save
    tensor sits in both ``input_tensor_ids`` and ``output_tensor_ids``,
    and the load side has to fire here regardless of whether the
    store side will later write back to the same tile. (Note:
    :func:`_iter_kernel_tensors` would yield such a tensor with role
    ``'output'`` and skip the load.)

    Output shape per format:

      * **BATCHED** — ``shape[1] * shape[2]`` contiguous elements from
        ``tensor_<id>_ptr + new_batch_idx_i32 * (shape[1] * shape[2])``
        into ``tensor_<id>_smem[0]``. One row per ``(batch, head)``
        slot; BATCHED smem tiles only have a single leading slot per
        the format gating in :func:`generate_initialization_str`.
      * **RAGGED** — ``workload_chunk_size * shape[1] * shape[2]``
        contiguous elements from
        ``tensor_<id>_ptr + ragged_idx_i32 * (shape[1] * shape[2])``
        into the full ``tensor_<id>_smem`` tile. The chunk is
        contiguous in global memory by construction.
      * **PAGED single-page** (``num_pages_per_workload == 1``) —
        identical pattern to RAGGED, but the global offset is
        ``page_idx_i32 * (shape[1] * shape[2])`` where
        ``page_idx_i32 = indices[ragged_idx_i32]`` is the start-of-page
        block index. The single page contains exactly the workload's
        ``num_blocks_per_page == workload_chunk_size`` blocks.
      * **PAGED multi-page** (``num_pages_per_workload > 1``) —
        page-by-page gather: each page p has block index
        ``indices[ragged_idx_i32 + p * num_blocks_per_page]``; valid
        pages (``p < _page``) are copied into smem slots
        ``[p*nbp, (p+1)*nbp)``, invalid pages are zero-filled so the
        compute path can read uniformly. See
        :func:`_emit_multipage_load`.

    Thread mapping: ``blockDim.x = 256`` threads stride through the
    flat element index ``j``. The strided loop ``for (j = tid; j < N;
    j += blockDim.x)`` is uniform across all ``workload_chunk_size``
    values — chunk in [16, 32, 64, 128] only scales ``N`` and therefore
    the per-thread iteration count.

    No ``__syncthreads()`` is emitted at the end of the load section.
    The next consumer of smem syncs itself: every Schedule.W compute
    op (Elementwise / Reduce / GeMM / ...) starts with
    ``__syncthreads()``, and :func:`generate_store_tensor_str` does
    the same. Cross-iteration safety is covered by the store's
    end-sync that runs just before the next iteration's load.
    """
    chunk = ctx.workload_chunk_size
    npw   = ctx.num_pages_per_workload
    nbp   = ctx.num_blocks_per_page

    batched: List[Tuple[int, object]] = []
    paged:   List[Tuple[int, object]] = []
    ragged:  List[Tuple[int, object]] = []
    for tid in sub_graph.input_tensor_ids:
        t = sub_graph.tensor_list[tid]
        if t._format == FORMAT.BATCHED:
            batched.append((tid, t))
        elif t._format == FORMAT.PAGED:
            paged.append((tid, t))
        elif t._format == FORMAT.RAGGED:
            ragged.append((tid, t))

    if not (batched or paged or ragged):
        return "// No tensor loading required"

    sections: List[str] = []

    # ---- 1. Per-format index loads (each emitted at most once) -------
    idx_lines: List[str] = []
    if batched:
        idx_lines.append("const int new_batch_idx_i32 = winfo_x_indices[i];")
    if paged or ragged:
        idx_lines.append("const int ragged_idx_i32 = winfo_y_offsets[i];")
    if paged and npw == 1:
        idx_lines.append("const int page_idx_i32 = indices[ragged_idx_i32];")
    if idx_lines:
        sections.append("\n".join(idx_lines))

    # ---- 2. Per-tensor load blocks -----------------------------------
    for tid, t in batched:
        block_elems = t.shape[1] * t.shape[2]
        sections.append(_emit_contig_load(
            tid, _cpp_kernel_value_type(t.dtype),
            src_offset_expr=f"new_batch_idx_i32 * {block_elems}",
            n_elems=block_elems,
        ))

    for tid, t in ragged:
        block_elems = t.shape[1] * t.shape[2]
        sections.append(_emit_contig_load(
            tid, _cpp_kernel_value_type(t.dtype),
            src_offset_expr=f"ragged_idx_i32 * {block_elems}",
            n_elems=chunk * block_elems,
        ))

    for tid, t in paged:
        block_elems = t.shape[1] * t.shape[2]
        cuda_t = _cpp_kernel_value_type(t.dtype)
        if npw == 1:
            sections.append(_emit_contig_load(
                tid, cuda_t,
                src_offset_expr=f"page_idx_i32 * {block_elems}",
                n_elems=chunk * block_elems,
            ))
        else:
            sections.append(_emit_multipage_load(
                tid, cuda_t, npw=npw, nbp=nbp, block_elems=block_elems,
                zero_expr=cast_float_to_smem("0.0f", t.dtype),
            ))

    # No trailing ``__syncthreads()`` — see docstring; consumers of
    # this load's smem (compute or store) emit their own start-sync.
    return "\n\n".join(sections)


def _emit_contig_store(
    tid: int, cuda_t: str, dst_offset_expr: str, n_elems_expr: str,
) -> str:
    """Strided block-cooperative copy of ``n_elems_expr`` contiguous
    elements from ``tensor_<tid>_smem`` (flattened) to
    ``tensor_<tid>_ptr + dst_offset_expr``.

    Inverse of :func:`_emit_contig_load`. Used for BATCHED, RAGGED, and
    PAGED single-page stores — all three are a single contiguous span
    in global memory and differ only in ``dst_offset_expr`` and the
    element count (RAGGED uses ``_len * block_elems`` to skip the
    trailing invalid blocks; BATCHED uses ``block_elems`` because each
    batch slot is exactly one row; PAGED single-page uses
    ``chunk * block_elems`` since the page is always fully populated).
    """
    return (
        f"{{\n"
        f"{INDENT}{cuda_t}* dst = tensor_{tid}_ptr + {dst_offset_expr};\n"
        f"{INDENT}const auto* src = &tensor_{tid}_smem[0][0][0];\n"
        f"{INDENT}for (int j = tid; j < {n_elems_expr}; j += blockDim.x) {{\n"
        f"{INDENT}{INDENT}dst[j] = src[j];\n"
        f"{INDENT}}}\n"
        f"}}"
    )


def _emit_multipage_store(
    tid: int, cuda_t: str, npw: int, nbp: int, block_elems: int,
) -> str:
    """Page-by-page scatter for PAGED outputs spanning multiple pages.

    Inverse of :func:`_emit_multipage_load`. For each page ``p`` in
    ``[0, num_pages_per_workload)`` with ``p < _page``: look up the
    start-of-page block index via
    ``indices[ragged_idx_i32 + p * num_blocks_per_page]``, then copy
    ``num_blocks_per_page * block_elems`` contiguous smem elements
    (from chunk slots ``[p*nbp, (p+1)*nbp)``) into that page in global
    memory.

    Invalid pages (``p >= _page``) are simply skipped — unlike the
    load side, there's nothing to zero-fill since global memory there
    is owned by the page allocator and won't be read at this index.
    """
    page_elems = nbp * block_elems
    return (
        f"{{\n"
        f"{INDENT}const auto* src = &tensor_{tid}_smem[0][0][0];\n"
        f"{INDENT}for (int p = 0; p < {npw}; ++p) {{\n"
        f"{INDENT}{INDENT}if (p < _page) {{\n"
        f"{INDENT}{INDENT}{INDENT}const int block_idx = "
        f"indices[ragged_idx_i32 + p * {nbp}];\n"
        f"{INDENT}{INDENT}{INDENT}{cuda_t}* dst = "
        f"tensor_{tid}_ptr + block_idx * {block_elems};\n"
        f"{INDENT}{INDENT}{INDENT}for (int j = tid; j < {page_elems}; "
        f"j += blockDim.x) {{\n"
        f"{INDENT}{INDENT}{INDENT}{INDENT}dst[j] = src[p * {page_elems} + j];\n"
        f"{INDENT}{INDENT}{INDENT}}}\n"
        f"{INDENT}{INDENT}}}\n"
        f"{INDENT}}}\n"
        f"}}"
    )


def generate_store_tensor_str(sub_graph: Graph, ctx: Context) -> str:
    """Cooperatively write subgraph **outputs** back from smem to global.

    Iterates :attr:`Graph.output_tensor_ids` — a Load+Save tensor sits
    in both input and output sets; the smem tile holds whatever the
    compute path produced and gets flushed here, possibly via the same
    addressing the load side used to populate it.

    Output shape per format:

      * **BATCHED** — gated by ``_is_first_workload = winfo_is_first_
        workload_per_batch[i]``. Only one workload per ``(batch, head)``
        commits the accumulator to global; the rest of the workloads
        for that slot already added their partial contributions into
        the same smem accumulator. ``shape[1] * shape[2]`` contiguous
        elements written to ``tensor_<id>_ptr + new_batch_idx_i32 *
        (shape[1] * shape[2])``. The gate is block-uniform so all
        threads take the same branch — no warp divergence.
      * **RAGGED** — ``_len * shape[1] * shape[2]`` contiguous elements
        written to ``tensor_<id>_ptr + ragged_idx_i32 *
        (shape[1] * shape[2])``. The bound is ``_len`` (not ``chunk``)
        so the trailing invalid blocks in a last-workload are skipped;
        this replaces Triton's per-lane ``valid`` mask vector.
      * **PAGED single-page** (``num_pages_per_workload == 1``) —
        ``chunk * shape[1] * shape[2]`` contiguous elements written to
        ``tensor_<id>_ptr + page_idx_i32 * (shape[1] * shape[2])``.
        No mask; the page is always fully populated for a single-page
        workload.
      * **PAGED multi-page** (``num_pages_per_workload > 1``) — see
        :func:`_emit_multipage_store`. Per-page scatter with a
        ``p < _page`` skip on invalid pages.

    Scoping: the store body lives in its own ``{ ... }`` block. That
    isolates store's index-load declarations (``new_batch_idx_i32``,
    ``ragged_idx_i32``, ``page_idx_i32``, ``_is_first_workload``) from
    :func:`generate_load_tensor_str`'s identically-named declarations
    at iteration scope. nvcc CSEs the duplicate ``winfo_*[i]`` loads
    so the source-level duplication has no runtime cost. The wrapping
    scope also means store can run even when load is empty (no
    subgraph inputs).

    Syncs: one ``__syncthreads()`` before the scope (compute may have
    written smem, store reads it) and one after (store reads smem,
    next iteration's load will write the same slots). Both are
    correctness-critical for compute paths that do cross-thread smem
    reductions; the cost is two block barriers per workload iteration.
    """
    chunk = ctx.workload_chunk_size
    npw   = ctx.num_pages_per_workload
    nbp   = ctx.num_blocks_per_page

    batched: List[Tuple[int, object]] = []
    paged:   List[Tuple[int, object]] = []
    ragged:  List[Tuple[int, object]] = []
    for tid in sub_graph.output_tensor_ids:
        t = sub_graph.tensor_list[tid]
        if t._format == FORMAT.BATCHED:
            batched.append((tid, t))
        elif t._format == FORMAT.PAGED:
            paged.append((tid, t))
        elif t._format == FORMAT.RAGGED:
            ragged.append((tid, t))

    if not (batched or paged or ragged):
        return "// No tensor storing required"

    sections: List[str] = []

    # ---- 1. Per-format index loads (own scope; isolated from load) ---
    idx_lines: List[str] = []
    if batched:
        idx_lines.append("const int new_batch_idx_i32 = winfo_x_indices[i];")
        idx_lines.append(
            "const uint8_t _is_first_workload = "
            "winfo_is_first_workload_per_batch[i];"
        )
    if paged or ragged:
        idx_lines.append("const int ragged_idx_i32 = winfo_y_offsets[i];")
    if paged and npw == 1:
        idx_lines.append("const int page_idx_i32 = indices[ragged_idx_i32];")
    if idx_lines:
        sections.append("\n".join(idx_lines))

    # ---- 2. BATCHED stores, all gated by _is_first_workload ---------
    if batched:
        batched_blocks: List[str] = []
        for tid, t in batched:
            block_elems = t.shape[1] * t.shape[2]
            batched_blocks.append(_emit_contig_store(
                tid, _cpp_kernel_value_type(t.dtype),
                dst_offset_expr=f"new_batch_idx_i32 * {block_elems}",
                n_elems_expr=str(block_elems),
            ))
        sections.append(
            "if (_is_first_workload) {\n"
            f"{indent_block(chr(10).join(batched_blocks), 1)}\n"
            "}"
        )

    # ---- 3. RAGGED stores, bound by _len -----------------------------
    for tid, t in ragged:
        block_elems = t.shape[1] * t.shape[2]
        sections.append(_emit_contig_store(
            tid, _cpp_kernel_value_type(t.dtype),
            dst_offset_expr=f"ragged_idx_i32 * {block_elems}",
            n_elems_expr=f"_len * {block_elems}",
        ))

    # ---- 4. PAGED stores --------------------------------------------
    for tid, t in paged:
        block_elems = t.shape[1] * t.shape[2]
        cuda_t = _cpp_kernel_value_type(t.dtype)
        if npw == 1:
            sections.append(_emit_contig_store(
                tid, cuda_t,
                dst_offset_expr=f"page_idx_i32 * {block_elems}",
                n_elems_expr=str(chunk * block_elems),
            ))
        else:
            sections.append(_emit_multipage_store(
                tid, cuda_t, npw=npw, nbp=nbp, block_elems=block_elems,
            ))

    body = "\n\n".join(sections)
    body_indented = indent_block(body, 1)
    return (
        f"__syncthreads();\n"
        f"{{\n"
        f"{body_indented}\n"
        f"}}\n"
        f"__syncthreads();"
    )


def generate_computation_str(sub_graph: Graph, ctx: Context) -> str:
    """Emit per-op computation source by dispatching each op in
    :attr:`Graph.op_list` (already in topological order) through the
    CUDA codegen registry at :func:`register.get_impl_func`.

    The return value is the concatenation of each op's CUDA C++ source,
    separated by blank lines. Each op codegen is expected to:

      * Read inputs from ``tensor_<id>_smem`` tiles populated by
        :func:`generate_load_tensor_str`.
      * Write outputs into ``tensor_<id>_smem`` tiles that
        :func:`generate_store_tensor_str` will flush back to global.
      * Promote storage dtypes to fp32 in registers for arithmetic
        and demote on the way back to smem (matches the
        narrow-storage / wide-compute pattern of the load/store
        codegens).
      * Emit ``__syncthreads()`` itself when its computation needs
        cross-thread smem visibility — the surrounding load/store
        syncs only bracket the section, not individual op
        boundaries.

    Until CUDA op codegens land in
    :mod:`cuda_impl.register.IMPL_REGISTRY`, this function will raise
    ``NotImplementedError`` from ``get_impl_func`` for any subgraph
    that actually contains ops. Empty subgraphs return the no-op
    fallback comment.
    """
    lines = [
        get_impl_func(op)(sub_graph, op_id, ctx)
        for op_id, op in enumerate(sub_graph.op_list)
    ]
    return "\n\n".join(lines) if lines else "// No computation required"


# --------------------------------------------------------------------------- #
# Kernel definition (W-scheduled)
# --------------------------------------------------------------------------- #

def _build_kernel_signature(sub_graph) -> List[str]:
    """Return kernel-argument declarations (CUDA C++), in declaration order.

    Order is 1:1 with :func:`generate_cuda_launcher`'s kernel-call
    expression list — that ordering is the ABI contract between the
    host launcher and the device kernel.

    Param qualifiers:

      * Planner pointers (``indices``, ``winfo_*``) are
        ``const T* __restrict__`` — read-only across the kernel, so
        nvcc can route loads through the read-only / texture path.
      * Per-tensor pointers are ``T* __restrict__``. Input-only tensors
        get an extra ``const``; output tensors (and Load+Save tensors,
        which :func:`_iter_kernel_tensors` yields with role
        ``'output'``) drop the ``const`` so writes are allowed.
      * ``tensor_<id>_dim0`` is ``int64_t`` to match the ``int64_t``
        arg type emitted by the launcher — no narrowing at the kernel
        call site.

    Each list entry is a single parameter **without** a trailing
    comma; the kernel template at :func:`generate_cuda_kernel` joins
    with ``",\\n"`` so the last param has no dangling comma.
    """
    args: List[str] = [
        "const int32_t* __restrict__ indices",
        "const int32_t* __restrict__ winfo_x_indices",
    ]
    if _has_batched_output(sub_graph):
        args.append("const uint8_t* __restrict__ winfo_is_first_workload_per_batch")
    args += [
        "const int32_t* __restrict__ winfo_y_offsets",
        "const int32_t* __restrict__ winfo_y_lens",
        "const int32_t* __restrict__ winfo_num_workloads",
    ]
    for tid, role in _iter_kernel_tensors(sub_graph):
        t = sub_graph.tensor_list[tid]
        cuda_t = _cpp_kernel_value_type(t.dtype)
        const_q = "const " if role == "input" else ""
        args.append(f"{const_q}{cuda_t}* __restrict__ tensor_{tid}_ptr")
        args.append(f"int64_t tensor_{tid}_dim0")
    return args


def _build_workload_preamble(ctx: Context) -> str:
    """Per-iteration ``_len`` / ``_page`` declarations (CUDA C++).

    Emitted at the top of every iteration of the workload loop in the
    kernel body. Two scalar locals, a deliberate subset of Triton's
    four-line preamble:

      * ``_len``  — number of valid blocks in this workload, loaded
        from ``winfo_y_lens[i]``. Every thread loads the same value
        and nvcc keeps it in a register.
      * ``_page`` — number of pages those blocks span; ceil-divide
        ``(_len + num_blocks_per_page - 1) / num_blocks_per_page`` with
        ``num_blocks_per_page`` baked in as a literal so the divide
        folds at compile time.

    Triton additionally emits two per-lane masks (``valid`` of length
    ``workload_chunk_size`` and ``page_valid`` of length
    ``num_pages_per_workload``). The CUDA backend doesn't precompute
    those — the load / store body codegens evaluate per-block validity
    inline against ``_len`` / ``_page`` when they do the actual
    addressing. Keeping the preamble scalar-only lets every thread run
    it without coordination.
    """
    nbp = ctx.num_blocks_per_page
    return "\n".join([
        "const int _len  = winfo_y_lens[i];",
        f"const int _page = (_len + {nbp - 1}) / {nbp};",
    ])


def generate_cuda_kernel(
    sub_graph: Graph,
    sub_graph_id: int,
    ctx: Context,
) -> str:
    """Emit the ``__global__`` kernel CUDA C++ source for a W-scheduled
    subgraph (kernel definition only — the host-side launcher comes
    from :func:`generate_cuda_launcher`).

    Template structure (mirrors the Triton kernel one-to-one):

      <preamble: #include directives shared with the launcher>

      __global__ void <name>_subgraph_<id>_kernel(
          <signature from _build_kernel_signature>
      ) {
          // Block-level partitioning of workloads. Each block claims
          // [start, end) — same per/r split Triton uses.
          const int pid       = blockIdx.x;
          const int num_progs = gridDim.x;
          const int tid       = threadIdx.x;

          const int n_workloads = winfo_num_workloads[0];

          const int per   = n_workloads / num_progs;
          const int r     = n_workloads % num_progs;
          const int start = pid * per + (pid < r ? pid : r);
          const int end   = start + per + (pid < r ? 1 : 0);

          <initialization_str>

          for (int i = start; i < end; ++i) {
              <prepare_workload_str>
              <load_tensor_str>
              <computation_str>
              <store_tensor_str>
          }
      }

    Body fragments come from the helpers below (currently ``pass``):

      * :func:`_build_kernel_signature`  — kernel parameter list.
      * :func:`generate_initialization_str` — pre-loop scratch decls.
      * :func:`_build_workload_preamble`    — per-iteration mask setup.
      * :func:`generate_load_tensor_str`    — input loads.
      * :func:`generate_computation_str`    — op computations.
      * :func:`generate_store_tensor_str`   — output stores.

    Until those land, calling this function will raise ``TypeError`` /
    ``AttributeError`` from the ``pass`` returns — the template body is
    here so subsequent steps can wire each helper in independently.
    """
    sa_name = ctx.sparse_attention_name
    kernel_fn = f"{sa_name}_subgraph_{sub_graph_id}_kernel"

    kernel_args = ",\n".join(
        f"{INDENT}{arg}" for arg in _build_kernel_signature(sub_graph)
    )

    initialization_str   = indent_block(generate_initialization_str(sub_graph, ctx), 1)
    prepare_workload_str = indent_block(_build_workload_preamble(ctx),               2)
    load_tensor_str      = indent_block(generate_load_tensor_str(sub_graph, ctx),    2)
    computation_str      = indent_block(generate_computation_str(sub_graph, ctx),    2)
    store_tensor_str     = indent_block(generate_store_tensor_str(sub_graph, ctx),   2)

    return (
        f"{_CUDA_KERNEL_PREAMBLE}\n"
        f"\n"
        f"\n"
        f"__global__ void {kernel_fn}(\n"
        f"{kernel_args}\n"
        f") {{\n"
        f"{INDENT}// ------------------------------------------------------------\n"
        f"{INDENT}// Block-level partitioning of workloads (mirrors the Triton\n"
        f"{INDENT}// kernel's per/r/start/end split: each block claims a\n"
        f"{INDENT}// contiguous span of [start, end) workloads).\n"
        f"{INDENT}// ------------------------------------------------------------\n"
        f"{INDENT}const int pid       = static_cast<int>(blockIdx.x);\n"
        f"{INDENT}const int num_progs = static_cast<int>(gridDim.x);\n"
        f"{INDENT}const int tid       = static_cast<int>(threadIdx.x);\n"
        f"\n"
        f"{INDENT}const int n_workloads = winfo_num_workloads[0];\n"
        f"\n"
        f"{INDENT}const int per   = n_workloads / num_progs;\n"
        f"{INDENT}const int r     = n_workloads % num_progs;\n"
        f"{INDENT}const int start = pid * per + (pid < r ? pid : r);\n"
        f"{INDENT}const int end   = start + per + (pid < r ? 1 : 0);\n"
        f"\n"
        f"{initialization_str}\n"
        f"\n"
        f"{INDENT}for (int i = start; i < end; ++i) {{\n"
        f"{prepare_workload_str}\n"
        f"{load_tensor_str}\n"
        f"{computation_str}\n"
        f"{store_tensor_str}\n"
        f"{INDENT}}}\n"
        f"}}"
    )


def _build_cuda_launcher_params(sub_graph) -> List[str]:
    """Return the C++ parameter list (one entry per param, no commas)
    of the host-side launcher emitted by :func:`generate_cuda_launcher`.

    Factored out so :func:`_generate_w_impl` can re-use the same list to
    emit a matching ``void <fn>(<params>);`` forward declaration in the
    ``cpp_sources`` arg of ``load_inline`` — pybind11 needs the symbol
    visible at link time when it generates the ``m.def(...)`` binding,
    so the declaration has to land in the host-compiled TU, not the
    nvcc-compiled one.
    """
    params: List[str] = [
        "torch::Tensor indices",
        "torch::Tensor winfo_x_indices",
    ]
    if _has_batched_output(sub_graph):
        params.append("torch::Tensor winfo_is_first_workload_per_batch")
    params += [
        "torch::Tensor winfo_y_offsets",
        "torch::Tensor winfo_y_lens",
        "torch::Tensor winfo_num_workloads",
    ]
    for tid, _role in _iter_kernel_tensors(sub_graph):
        params.append(f"torch::Tensor tensor_{tid}")
        params.append(f"int64_t tensor_{tid}_dim0")
    return params


def generate_cuda_launcher(
    sub_graph: Graph,
    sub_graph_id: int,
    ctx: Context,
) -> str:
    """Emit the host-side C++ launcher source for a W-scheduled subgraph.

    Signature mirrors the kernel signature emitted by
    :func:`generate_cuda_kernel`:

      * planner pointers — ``indices``, ``winfo_x_indices``,
        (``winfo_is_first_workload_per_batch`` iff BATCHED outputs),
        ``winfo_y_offsets``, ``winfo_y_lens``, ``winfo_num_workloads``.
        All are ``int32`` tensors except the per-batch flag which is
        ``uint8``.
      * per kernel tensor: a ``torch::Tensor`` arg + an ``int64_t`` leading
        dim. The leading dim's role matches Triton's ``tensor_X_dim0``
        ``tl.constexpr`` parameter; in CUDA we just pass it at runtime.

    The body launches the kernel with a 1-D grid of ``num_sms * 4``
    blocks (mirrors the Triton wrapper's ``(ctx.num_sms * 4,)`` grid)
    and 256 threads per block (8 warps × 32 threads, mirroring Triton's
    ``num_warps=8`` tuning). The kernel reads ``winfo_num_workloads[0]``
    itself and partitions the workload range across blocks, so the
    launcher does **not** sync to read the workload count on the host.

    The launcher uses ``at::cuda::getCurrentCUDAStream()`` —
    :func:`generate_cuda_kernel` must include
    ``<ATen/cuda/CUDAContext.h>`` (and the half-precision headers
    ``<cuda_fp16.h>`` / ``<cuda_bf16.h>``) in its preamble.

    The exposed function name matches the entry put in
    ``functions=[...]`` of the ``load_inline`` call emitted by
    :func:`_generate_w_impl`.
    """
    sa_name     = ctx.sparse_attention_name
    kernel_fn   = f"{sa_name}_subgraph_{sub_graph_id}_kernel"
    launcher_fn = f"{sa_name}_subgraph_{sub_graph_id}_launcher"

    # --- launcher parameters (host-side C++ signature) -----------------
    params = _build_cuda_launcher_params(sub_graph)

    # --- kernel-call arguments (device-side pointers + scalars) --------
    kcall: List[str] = [
        "indices.data_ptr<int32_t>()",
        "winfo_x_indices.data_ptr<int32_t>()",
    ]
    if _has_batched_output(sub_graph):
        kcall.append("winfo_is_first_workload_per_batch.data_ptr<uint8_t>()")
    kcall += [
        "winfo_y_offsets.data_ptr<int32_t>()",
        "winfo_y_lens.data_ptr<int32_t>()",
        "winfo_num_workloads.data_ptr<int32_t>()",
    ]
    for tid, _role in _iter_kernel_tensors(sub_graph):
        t = sub_graph.tensor_list[tid]
        kcall.append(_cpp_data_ptr_expr(f"tensor_{tid}", t.dtype))
        kcall.append(f"tensor_{tid}_dim0")

    # Grid + block: bake as ``constexpr int`` literals at codegen time so
    # the compiled SASS specializes on the host's SM count. The kernel
    # body itself reads ``gridDim.x`` / ``blockIdx.x`` to partition work.
    num_progs  = ctx.num_sms * 4
    block_size = 256  # 8 warps * 32 threads — matches Triton ``num_warps=8``

    params_str = ",\n".join(f"{INDENT}{p}" for p in params)
    kcall_str  = ",\n".join(f"{INDENT * 2}{a}" for a in kcall)

    return (
        f"void {launcher_fn}(\n"
        f"{params_str}\n"
        f") {{\n"
        f"{INDENT}const auto stream = at::cuda::getCurrentCUDAStream();\n"
        f"{INDENT}constexpr int num_progs = {num_progs};\n"
        f"{INDENT}constexpr int block_size = {block_size};\n"
        f"{INDENT}{kernel_fn}<<<num_progs, block_size, 0, stream>>>(\n"
        f"{kcall_str}\n"
        f"{INDENT});\n"
        f"}}"
    )


# --------------------------------------------------------------------------- #
# Python launcher / impl wrapper
# --------------------------------------------------------------------------- #

def _build_launcher_args(sub_graph) -> Tuple[List[str], List[str], List[str]]:
    """Return ``(wrapper_args, launcher_call_args, fp8_rebind_lines)``.

    ``wrapper_args``        — Python signature lines of the impl wrapper
                              (one ``tensor_<id>`` per kernel-arg
                              tensor, then ``ctx``).
    ``launcher_call_args``  — expressions passed into the C++ launcher,
                              in 1:1 order with the parameter list
                              emitted by :func:`generate_cuda_launcher`:
                              ``ctx.dense_kv_indices``,
                              ``ctx.winfo_q_indices``,
                              (``ctx.winfo_is_first_workload_per_batch``
                              iff BATCHED outputs),
                              ``ctx.winfo_kv_offsets``,
                              ``ctx.winfo_kv_lens``,
                              ``ctx.winfo_num_workloads``,
                              then for each kernel tensor
                              ``tensor_<id>, tensor_<id>.shape[0]``.
    ``fp8_rebind_lines``    — for any FP8 tensor, ``.view(torch.uint8)``
                              re-bind lines emitted before the launcher
                              call. The launcher's C++ signature for an
                              fp8 tensor declares ``uint8_t*`` (see
                              ``_TORCH_TO_CPP_TYPES``), so the rebind is
                              what makes the dtypes line up.
    """
    wrapper_args: List[str] = []
    launcher_call_args: List[str] = [
        "ctx.dense_kv_indices",
        "ctx.winfo_q_indices",
    ]
    if _has_batched_output(sub_graph):
        launcher_call_args.append("ctx.winfo_is_first_workload_per_batch")
    launcher_call_args += [
        "ctx.winfo_kv_offsets",
        "ctx.winfo_kv_lens",
        "ctx.winfo_num_workloads",
    ]
    fp8_rebind: List[str] = []

    for tid, _role in _iter_kernel_tensors(sub_graph):
        name = f"tensor_{tid}"
        t = sub_graph.tensor_list[tid]
        wrapper_args.append(name)
        launcher_call_args.extend([name, f"{name}.shape[0]"])
        if _is_fp8(t):
            fp8_rebind.append(f"{name} = {name}.view(torch.uint8)")

    wrapper_args.append("ctx")
    return wrapper_args, launcher_call_args, fp8_rebind


def _generate_w_impl(sub_graph: Graph, sub_graph_id: int, ctx: Context) -> str:
    """Generate a W-scheduled subgraph as a JIT-compiled CUDA extension.

    Output structure (concatenated into the per-subgraph module):

        _<name>_subgraph_<id>_cuda_src = r\"\"\"
            <cuda kernel + host launcher source>
        \"\"\"
        _<name>_subgraph_<id>_cuda_mod = load_inline(
            name=...,
            cpp_sources="",
            cuda_sources=_<name>_subgraph_<id>_cuda_src,
            functions=["<name>_subgraph_<id>_launcher"],
        )

        def <name>_subgraph_<id>_impl(tensor_X, ..., ctx):
            <fp8 rebinds>
            _<name>_subgraph_<id>_cuda_mod.<launcher>(
                <forwarded args>
            )

    The ``load_inline`` call runs at module-import time; ``cpp_extension``
    caches by source hash, so repeat imports are fast.
    """
    kernel_src = generate_cuda_kernel(sub_graph, sub_graph_id, ctx)
    launcher_src = generate_cuda_launcher(sub_graph, sub_graph_id, ctx)
    wrapper_args, launcher_call_args, fp8_rebind = _build_launcher_args(sub_graph)

    sa_name = ctx.sparse_attention_name
    impl_name   = f"{sa_name}_subgraph_{sub_graph_id}_impl"
    launcher_fn = f"{sa_name}_subgraph_{sub_graph_id}_launcher"
    mod_name    = f"{sa_name}_subgraph_{sub_graph_id}_cuda_mod"
    src_var     = f"_{sa_name}_subgraph_{sub_graph_id}_cuda_src"
    decl_var    = f"_{sa_name}_subgraph_{sub_graph_id}_cpp_decl"
    mod_var     = f"_{sa_name}_subgraph_{sub_graph_id}_cuda_mod"

    # Concatenate the kernel and host-launcher sources into a single
    # CUDA TU. Both must be compiled by nvcc (the launcher uses the
    # ``<<<...>>>`` triple-chevron syntax), so both go in
    # ``cuda_sources``.
    cuda_source = "\n\n".join(s for s in (kernel_src, launcher_src) if s)

    # ``cpp_sources`` carries a forward declaration of the launcher.
    # ``load_inline`` generates the pybind11 ``m.def`` binding for each
    # name in ``functions=[...]`` from the host-compiled TU, so the
    # symbol has to be declared *there* (the definition lives in the
    # nvcc-compiled TU and gets linked in via ninja).
    params_str = ",\n".join(f"{INDENT}{p}" for p in _build_cuda_launcher_params(sub_graph))
    cpp_decl = (
        f"#include <torch/extension.h>\n"
        f"#include <cstdint>\n"
        f"void {launcher_fn}(\n"
        f"{params_str}\n"
        f");"
    )

    args_def = ",\n".join(f"{INDENT}{a}" for a in wrapper_args)
    call_args = ",\n".join(f"{INDENT * 2}{a}" for a in launcher_call_args)
    fp8_block = indent_block("\n".join(fp8_rebind), 1) if fp8_rebind else ""

    impl_str = f'''
{src_var} = r"""
{cuda_source}
"""

{decl_var} = r"""
{cpp_decl}
"""

{mod_var} = load_inline(
    name={mod_name!r},
    cpp_sources={decl_var},
    cuda_sources={src_var},
    functions=[{launcher_fn!r}],
    verbose=False,
)


def {impl_name}(
{args_def}
):
{fp8_block}
    {mod_var}.{launcher_fn}(
{call_args}
    )
'''
    return impl_str.strip()


def _generate_non_w_impl(sub_graph: Graph, sub_graph_id: int, ctx: Context) -> str:
    """Direct op-impl wrapper for non-workload-scheduled subgraphs.

    Mirrors the Triton backend: the single op carries all of its own
    codegen via :func:`get_impl_func`; this layer just stitches the
    function signature and indents the op-impl body.
    """
    assert len(sub_graph.op_list) == 1, (
        "Expected exactly one operation in non-workload-scheduled "
        "sub-graph for direct implementation."
    )

    arg_list = [f"tensor_{tid}" for tid in sub_graph.input_tensor_ids]
    arg_list += [f"tensor_{tid}" for tid in sub_graph.output_tensor_ids]
    arg_list.append("ctx")
    args_def = ",\n".join(f"{INDENT}{arg}" for arg in arg_list)

    op_impl_str = indent_block(
        get_impl_func(sub_graph.op_list[0])(sub_graph, 0, ctx), 1,
    )

    impl_str = f"""
def {ctx.sparse_attention_name}_subgraph_{sub_graph_id}_impl(
{args_def}
):
{op_impl_str}
"""
    return impl_str.strip()


def generate_cuda_impl(
    sub_graph: Graph,
    sub_graph_id: int,
    ctx: Context,
) -> str:
    """Generate a CUDA kernel and its Python wrapper implementation.

    Dispatches on :attr:`Graph.schedule`:

      - ``Schedule.W`` → :func:`_generate_w_impl` (CUDA kernel + host
        launcher embedded as a JIT-compiled extension).
      - anything else → :func:`_generate_non_w_impl` (single-op direct
        Python wrapper).

    Also registers the headers needed by the generated module —
    ``interface.py`` dedups them, so re-extending across subgraphs is
    safe.
    """
    ctx.compilation_header_lines.extend([
        "import torch",
        "from torch.utils.cpp_extension import load_inline",
        # Needed by the Schedule.S codegens reused from triton_impl
        # (see :mod:`cuda_impl.register`). Harmless on pure-W flows —
        # ``interface.py`` dedups headers, and unused imports cost
        # nothing at runtime.
        "import triton",
        "import triton.language as tl",
    ])
    if sub_graph.schedule == Schedule.W:
        return _generate_w_impl(sub_graph, sub_graph_id, ctx)
    return _generate_non_w_impl(sub_graph, sub_graph_id, ctx)
