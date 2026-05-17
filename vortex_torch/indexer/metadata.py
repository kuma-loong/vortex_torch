"""Per-forward-batch metadata for the indexer.

The :class:`MetaData` object owns every tensor / scalar whose value depends
on the *current* forward batch — what the planner writes during
``init_forward_metadata`` and what the compiled indexer kernels read on
the hot path. It is *separate* from :class:`Context`, which holds purely
static configuration (page/block sizes, head counts, allocation budgets,
graph metadata, codegen scratch).

Pre-allocation is done once at attention-backend ``__init__`` via
:func:`MetaData.preallocate` — the same buffer objects are reused across
every forward batch (CUDA-graph friendly: pointers never move, only the
contents change).

Layout summary:

  * ``winfo_*`` — workload-scheduler outputs (per-workload, length
    ``ctx.max_num_workloads``).
  * ``dense_kv_indptr`` / ``sparse_kv_indptr`` — CSR prefix sums
    (length ``max_bs * num_kv_heads + 1``). ``None`` in trtllm mode.
  * ``dense_kv_indices`` / ``sparse_kv_indices`` — flat CSR block ids
    (length ``max_bs * num_kv_heads * max_blocks_per_seq``). ``None`` in
    trtllm mode.
  * ``dense_seqlens`` / ``sparse_seqlens`` — per-row token counts
    (length ``max_bs * num_kv_heads``). ``None`` in flashinfer mode.
  * ``dense_block_tables`` / ``sparse_block_tables`` —
    ``[max_bs * num_kv_heads, max_blocks_per_seq]`` block ids. ``None``
    in flashinfer mode.
  * ``kv_last_page_len`` — per-row last-page token count
    (length ``max_bs * num_kv_heads``).
  * ``batch_size`` — current request count (set by every planner call).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from .context import Context


class MetaData:
    """Per-forward-batch state for the indexer. See module docstring."""

    __slots__ = (
        # workload scheduler outputs
        "winfo_q_indices",
        "winfo_is_first_workload_per_batch",
        "winfo_kv_offsets",
        "winfo_kv_lens",
        "winfo_num_workloads",
        "winfo_chunk_size",
        # CSR (flashinfer) buffers — None when backend == "trtllm"
        "dense_kv_indptr",
        "sparse_kv_indptr",
        "dense_kv_indices",
        "sparse_kv_indices",
        # block-table (trtllm) buffers — None when backend == "flashinfer"
        "dense_seqlens",
        "sparse_seqlens",
        "dense_block_tables",
        "sparse_block_tables",
        # both backends
        "kv_last_page_len",
        "batch_size",
    )

    # Type hints (informational only; values populated by ``preallocate``).
    winfo_q_indices: torch.Tensor
    winfo_is_first_workload_per_batch: torch.Tensor
    winfo_kv_offsets: torch.Tensor
    winfo_kv_lens: torch.Tensor
    winfo_num_workloads: torch.Tensor
    winfo_chunk_size: torch.Tensor

    dense_kv_indptr: Optional[torch.Tensor]
    sparse_kv_indptr: Optional[torch.Tensor]
    dense_kv_indices: Optional[torch.Tensor]
    sparse_kv_indices: Optional[torch.Tensor]

    dense_seqlens: Optional[torch.Tensor]
    sparse_seqlens: Optional[torch.Tensor]
    dense_block_tables: Optional[torch.Tensor]
    sparse_block_tables: Optional[torch.Tensor]

    kv_last_page_len: torch.Tensor
    batch_size: int

    def __init__(self) -> None:
        for name in self.__slots__:
            object.__setattr__(
                self, name, 0 if name == "batch_size" else None
            )

    # ------------------------------------------------------------------
    # Pre-allocation entry points (one per attention backend)
    # ------------------------------------------------------------------
    @classmethod
    def preallocate(
        cls,
        ctx: "Context",
        *,
        device: torch.device | str,
    ) -> "MetaData":
        """Build a backend-appropriate ``MetaData`` from a populated
        :class:`Context`.

        Picks the buffer set from ``ctx.vortex_attention_backend``:
          * ``"flashinfer"`` → allocates CSR buffers
            (``dense/sparse_kv_indptr``, ``dense/sparse_kv_indices``);
            leaves block-table buffers as ``None``.
          * ``"trtllm"``     → allocates 2D block_tables + per-row seqlens;
            leaves CSR buffers as ``None``.

        ``winfo_*`` and ``kv_last_page_len`` are always allocated.
        """
        md = cls()
        md._alloc_common(ctx, device=device)

        backend = (ctx.vortex_attention_backend or "flashinfer").lower()
        if backend == "trtllm":
            md._alloc_trtllm(ctx, device=device)
        elif backend == "flashinfer":
            md._alloc_flashinfer(ctx, device=device)
        else:
            raise ValueError(
                f"MetaData.preallocate: unknown vortex_attention_backend "
                f"{ctx.vortex_attention_backend!r}; expected 'flashinfer' or 'trtllm'"
            )
        return md

    # ------------------------------------------------------------------
    def _alloc_common(self, ctx: "Context", *, device) -> None:
        eff_bs = ctx.max_bs * ctx.num_kv_heads
        i32 = torch.int32
        u8 = torch.uint8

        self.winfo_q_indices = torch.zeros(
            (ctx.max_num_workloads,), dtype=i32, device=device,
        )
        self.winfo_is_first_workload_per_batch = torch.zeros(
            (ctx.max_num_workloads,), dtype=u8, device=device,
        )
        self.winfo_kv_offsets = torch.zeros(
            (ctx.max_num_workloads,), dtype=i32, device=device,
        )
        self.winfo_kv_lens = torch.zeros(
            (ctx.max_num_workloads,), dtype=i32, device=device,
        )
        self.winfo_num_workloads = torch.zeros((1,), dtype=i32, device=device)
        self.winfo_chunk_size = torch.zeros((1,), dtype=i32, device=device)

        # Every backend's planner writes the per-row last-block (token) length.
        self.kv_last_page_len = torch.ones((eff_bs,), dtype=i32, device=device)

    def _alloc_flashinfer(self, ctx: "Context", *, device) -> None:
        eff_bs = ctx.max_bs * ctx.num_kv_heads
        # CSR ``kv_indices`` length: one int32 per cached block across the
        # full request budget. Use ``max_num_blocks`` (already the planner
        # allocation budget) for parity with the prior backend wiring.
        indices_len = ctx.max_num_blocks
        i32 = torch.int32

        self.dense_kv_indptr = torch.zeros((eff_bs + 1,), dtype=i32, device=device)
        self.sparse_kv_indptr = torch.zeros((eff_bs + 1,), dtype=i32, device=device)
        self.dense_kv_indices = torch.zeros((indices_len,), dtype=i32, device=device)
        self.sparse_kv_indices = torch.zeros((indices_len,), dtype=i32, device=device)
        # block-table buffers stay ``None`` in flashinfer mode.

    def _alloc_trtllm(self, ctx: "Context", *, device) -> None:
        eff_bs = ctx.max_bs * ctx.num_kv_heads
        max_blocks_per_seq = ctx.max_num_blocks_per_request
        i32 = torch.int32

        self.dense_seqlens = torch.zeros((eff_bs,), dtype=i32, device=device)
        self.sparse_seqlens = torch.zeros((eff_bs,), dtype=i32, device=device)
        self.dense_block_tables = torch.zeros(
            (eff_bs, max_blocks_per_seq), dtype=i32, device=device,
        )
        self.sparse_block_tables = torch.zeros(
            (eff_bs, max_blocks_per_seq), dtype=i32, device=device,
        )
        # CSR buffers stay ``None`` in trtllm mode.

    # ------------------------------------------------------------------
    def set_batch_size(self, n: int) -> None:
        self.batch_size = n


__all__ = ["MetaData"]
