"""Backend-specific dispatch traits for the Triton indexer compiler.

The indexer compiler emits the same op graph for both attention backends,
but the per-row addressing and the launcher bindings differ in well-defined
ways. Rather than scatter ``if backend == "trtllm":`` branches across every
codegen, we collect every backend-specific snippet into a single
:class:`IndexerBackend` traits object selected once at the top of each
codegen call via :func:`get_backend`.

Two backends live here today:

  * ``flashinfer`` — **CSR** layout. The planner emits flat
    ``dense_kv_indices`` / ``sparse_kv_indices`` keyed by
    ``dense_kv_indptr`` / ``sparse_kv_indptr``. Per-row block count comes
    from the indptr difference. Schedule.W's ``indices`` parameter is bound
    to ``ctx.dense_kv_indices``.

  * ``trtllm`` — **block-table** layout. The planner emits
    ``dense_block_tables`` / ``sparse_block_tables`` (shape
    ``[eff_bs, max_blocks_per_seq]``) and ``dense_seqlens`` /
    ``sparse_seqlens`` (length ``eff_bs``, in tokens); **no indptr or
    flat indices are written**. Per-row block count is
    ``ceil(seqlens[pid] / block_size)``. Schedule.W's ``indices`` parameter
    is bound to ``ctx.dense_block_tables.view(-1)`` with
    ``winfo_kv_offsets[j] = row * max_blocks_per_seq + col`` baked by the
    planner.

Adding a third backend means adding one more :class:`IndexerBackend` entry
to :data:`_BACKENDS` — no codegen changes elsewhere.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ....utils import INDENT
from ...context import Context


@dataclass(frozen=True)
class IndexerBackend:
    """Snippet bundle that fully describes the consumer-side
    backend-specific surface of the Triton indexer compiler.

    All ``*_snippet`` / ``*_arg`` / ``*_value`` fields are raw strings
    splice-substituted into generated source — no Python objects leak into
    the kernel.
    """

    # ----- identity -----
    name: str

    # ----- Schedule.W kernel launcher -----
    # Tensor bound to the ``indices`` parameter of every Schedule.W kernel
    # so the preamble's
    #   ``page_idx_i32 = tl.load(indices + ragged_idx_i32)``
    # resolves to a page id. The planner's ``winfo_kv_offsets`` encoding
    # must match this binding (see ``planner_sglang.py``).
    indices_src: str

    # ----- Schedule.S per-row kernels (softmax / normalize / conv1d / reduce_dim0) -----
    # Name of the per-row tensor kernel parameter (third positional slot).
    per_row_kernel_param: str

    # Launcher value passed for ``per_row_kernel_param`` (a ``ctx.<field>``
    # reference).
    per_row_launcher_value: str

    # Extra ``: tl.constexpr,`` kernel parameters appended after the
    # canonical ``x_D0`` / ``x_D1`` block. Multi-line, each line already
    # indented and comma-terminated; empty string when none.
    extra_kernel_constexpr_args: str

    # Extra positional launcher arguments matching ``extra_kernel_constexpr_args``.
    # Multi-line, each line already indented and comma-terminated; empty
    # when none.
    extra_launcher_args: str

    # Snippet that runs at the top of every per-row Schedule.S kernel and
    # produces two locals:
    #   * ``start``                — block-offset of this row's first valid
    #                                block in the RAGGED storage buffer.
    #   * ``num_blocks_this_seq``  — number of valid blocks for this row.
    # The snippet may reference ``pid`` and ``per_row_kernel_param`` (and,
    # in trtllm mode, the ``block_size`` / ``max_blocks_per_seq``
    # constexpr params). Indented as kernel body (one INDENT per line).
    start_and_count_snippet: str

    # ----- topK / approxTopK CUDA dispatch -----
    # Per-row args passed in slots 2 and 3 of the topk C ABI.
    topk_per_row_args: Tuple[str, str]
    # Dense index/block-table source (slot 4).
    topk_dense_src: str
    # Trailing args after the bos/eos pair. flashinfer passes one
    # (``max_num_pages``); trtllm passes two (``max_blocks_per_seq``,
    # ``block_size``).
    topk_trailing_args: Tuple[str, ...]


# Indented helpers ---------------------------------------------------------

def _indent_lines(text: str, level: int = 1) -> str:
    """Indent every non-blank line in ``text`` by ``level`` ``INDENT`` units."""
    prefix = INDENT * level
    return "\n".join(prefix + line if line else line for line in text.splitlines())


# Every dynamic (per-forward-batch) buffer lives on ``ctx.metadata`` now —
# ``Context`` keeps only static config. Static values such as
# ``ctx.block_size`` are still read straight off ``ctx``.

# ---------- flashinfer / CSR ----------
_FLASHINFER = IndexerBackend(
    name="flashinfer",
    indices_src="ctx.metadata.dense_kv_indices",
    per_row_kernel_param="indptr",
    per_row_launcher_value="ctx.metadata.dense_kv_indptr",
    extra_kernel_constexpr_args="",
    extra_launcher_args="",
    start_and_count_snippet=_indent_lines(
        "start = tl.load(indptr + pid)\n"
        "_end = tl.load(indptr + pid + 1)\n"
        "num_blocks_this_seq = _end - start"
    ),
    topk_per_row_args=(
        "ctx.metadata.dense_kv_indptr",
        "ctx.metadata.sparse_kv_indptr",
    ),
    topk_dense_src="ctx.metadata.dense_kv_indices",
    topk_trailing_args=("ctx.max_num_blocks_per_request",),
)


# ---------- trtllm / block-tables ----------
_TRTLLM = IndexerBackend(
    name="trtllm",
    indices_src="ctx.metadata.dense_block_tables.view(-1)",
    per_row_kernel_param="seqlens",
    per_row_launcher_value="ctx.metadata.dense_seqlens",
    extra_kernel_constexpr_args=(
        f"{INDENT}block_size: tl.constexpr,\n"
        f"{INDENT}max_blocks_per_seq: tl.constexpr,"
    ),
    extra_launcher_args=(
        f"{INDENT*2}ctx.block_size,\n"
        f"{INDENT*2}ctx.metadata.dense_block_tables.shape[1],"
    ),
    start_and_count_snippet=_indent_lines(
        "_tokens = tl.load(seqlens + pid)\n"
        "num_blocks_this_seq = (_tokens + block_size - 1) // block_size\n"
        "start = pid * max_blocks_per_seq"
    ),
    topk_per_row_args=(
        "ctx.metadata.dense_seqlens",
        "ctx.metadata.sparse_seqlens",
    ),
    topk_dense_src="ctx.metadata.dense_block_tables",
    topk_trailing_args=(
        "ctx.metadata.dense_block_tables.shape[1]",
        "ctx.block_size",
    ),
)


_BACKENDS = {
    "flashinfer": _FLASHINFER,
    "trtllm":     _TRTLLM,
}


def get_backend(ctx: Context) -> IndexerBackend:
    """Pick the :class:`IndexerBackend` traits for ``ctx``.

    Falls back to flashinfer/CSR when ``ctx.vortex_attention_backend`` is
    unset or empty — this matches the historical default for callers
    (notably ``vortex_torch.flow.verify``) that don't pin a backend.
    """
    name = (getattr(ctx, "vortex_attention_backend", None) or "flashinfer").lower()
    try:
        return _BACKENDS[name]
    except KeyError:
        raise ValueError(
            f"unknown vortex_attention_backend {name!r}; "
            f"expected one of {sorted(_BACKENDS)}"
        ) from None


__all__ = ["IndexerBackend", "get_backend"]
