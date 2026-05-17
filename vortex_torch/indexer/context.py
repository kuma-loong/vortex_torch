from __future__ import annotations

from typing import Any, Final, Optional, Union
import uuid

import torch

from ..abs import ContextBase
from ..utils import UNSET, Mode, resolve_dtype
from .metadata import MetaData


class Context(ContextBase):
    """Static, single-instance indexer context.

    Holds **only** configuration that's fixed for the lifetime of the
    compiled indexer — shapes, page/block sizes, head counts, allocation
    budgets, codegen scratch (op/tensor lists), backend identity, …

    Per-forward-batch buffers (winfo_*, dense/sparse_kv_indptr+indices,
    dense/sparse_seqlens, dense/sparse_block_tables, kv_last_page_len)
    and ``batch_size`` live on a separate :class:`MetaData` object,
    pre-allocated at attention-backend ``__init__`` and exposed as
    ``ctx.metadata``. Codegen emits ``ctx.metadata.<field>`` for any
    value that varies between forward batches.

    Build pattern (from each attention backend's ``_compile``):
        ctx = Context()
        ctx.create(self, model_runner)              # static fields
        ctx.metadata = MetaData.preallocate(ctx, device=...)
    """

    __slots__ = ContextBase.__slots__ + (
        # ---- per-forward-batch state (pre-allocated MetaData) ----
        "metadata",
        # ---- batch budget ----
        "max_bs",
        # ---- active attention backend ----
        # "flashinfer" (CSR indices, default) or "trtllm" (2D block_tables).
        # Set by Context.create() from server_args.vortex_attention_backend;
        # consumed by the indexer compiler's IndexerBackend traits.
        "vortex_attention_backend",
        # ---- workload-scheduler shape ----
        "max_num_workloads",
        "workload_chunk_size",
        # ---- head / shape ----
        "group_size", "num_kv_heads", "num_qo_heads", "head_dim",
        # ---- hardware / paging ----
        "num_sms", "page_size", "max_num_pages", "max_num_pages_per_request",
        "block_size", "max_num_blocks", "max_num_blocks_per_request",
        "num_blocks_per_page", "num_pages_per_workload",
        # ---- topk / reserved-slot policy ----
        "topk_val", "topk_ratio", "block_reserved_bos", "block_reserved_eos",
        "max_topk_val",
        # ---- auxiliary accounting ----
        "_aux_total_bytes", "_aux_total_flops",
        # ---- codegen / graph scratch ----
        "tensor_list", "op_list", "output_tensor_to_op_list",
        "op_to_input_tensor_list", "op_to_output_tensor_list",
        "side_effect_op_ids",
        "sparse_attention_name", "impl_backend", "tensor_id_to_tensor_name_map",
        "compilation_header_lines", "auxilary_func_def_lines",
        "compilation_cache_dir",
    )

    # ---- type hints (declarations only) ----
    metadata: Optional[MetaData]
    max_bs: int
    vortex_attention_backend: str
    max_num_workloads: int
    workload_chunk_size: int
    group_size: int
    num_kv_heads: int
    num_qo_heads: int
    head_dim: int
    num_sms: int
    page_size: int
    max_num_pages: int
    max_num_pages_per_request: int
    block_size: int
    max_num_blocks: int
    max_num_blocks_per_request: int
    num_blocks_per_page: int
    num_pages_per_workload: int
    topk_val: int
    topk_ratio: float
    block_reserved_bos: int
    block_reserved_eos: int
    max_topk_val: Union[int, None]
    _aux_total_bytes: int
    _aux_total_flops: int
    tensor_list: list
    op_list: list
    output_tensor_to_op_list: list
    op_to_input_tensor_list: list
    op_to_output_tensor_list: list
    side_effect_op_ids: list
    sparse_attention_name: str
    impl_backend: str
    tensor_id_to_tensor_name_map: dict
    compilation_header_lines: list
    auxilary_func_def_lines: list
    compilation_cache_dir: str

    def __init__(self) -> None:
        for name in self.__slots__:
            if name == "_created":
                object.__setattr__(self, name, False)
            elif name == "name":
                object.__setattr__(self, name, "Indexer")
            elif name == "_aux_total_bytes":
                object.__setattr__(self, name, 0)
            elif name == "_aux_total_flops":
                object.__setattr__(self, name, 0)
            elif name == "mode":
                object.__setattr__(self, name, Mode.profile)
            elif name == "metadata":
                object.__setattr__(self, name, None)
            else:
                object.__setattr__(self, name, UNSET)

    # ------------------------------------------------------------------
    # Per-batch convenience accessors — forward to ``self.metadata``.
    # ------------------------------------------------------------------
    @property
    def batch_size(self) -> int:
        """Current batch size — proxied from ``self.metadata``.

        Kept as a property (not a slot) so the only writable copy lives on
        ``self.metadata``; ``ctx.batch_size`` is now read-only.
        """
        return 0 if self.metadata is None else self.metadata.batch_size

    def set_batch_size(self, n: int) -> None:
        """Compatibility shim — forwards to ``self.metadata.set_batch_size``."""
        if self.metadata is None:
            raise RuntimeError(
                "Context.set_batch_size called before MetaData was preallocated; "
                "did you forget `ctx.metadata = MetaData.preallocate(ctx, device=...)`?"
            )
        self.metadata.set_batch_size(n)

    # ------------------------------------------------------------------
    def create(self, parent: Any, model_runner: Any, *, overwrite: bool = False) -> "Context":
        """Populate the static fields. Per-batch ``MetaData`` is allocated
        separately by the caller via ``MetaData.preallocate(ctx, device=...)``
        — see this class's docstring.
        """
        if self._created and not overwrite:
            raise RuntimeError("Context.create() already called; pass overwrite=True to reinitialize.")

        sa = model_runner.server_args
        max_pages_per_req = (
            (model_runner.model_config.context_len + sa.page_size - 1) // sa.page_size
            if sa.vortex_max_seq_lens < 0
            else (sa.vortex_max_seq_lens + sa.page_size - 1) // sa.page_size
        )
        max_bs = int(model_runner.req_to_token_pool.size)
        self.max_bs = max_bs

        self.workload_chunk_size = sa.vortex_workload_chunk_size

        self.group_size = parent.group_size
        self.num_kv_heads = parent.num_kv_heads
        self.num_qo_heads = parent.num_qo_heads
        self.head_dim = parent.head_dim

        self.num_sms = torch.cuda.get_device_properties(0).multi_processor_count
        self.page_size = sa.page_size
        self.block_size = sa.vortex_block_size
        self.num_blocks_per_page = self.page_size // self.block_size
        assert self.page_size % self.block_size == 0, "Page size must be a multiple of block size."
        assert self.workload_chunk_size % self.num_blocks_per_page == 0, "Workload chunk size must be a multiple of blocks per page."
        # Capacity model (adjust as needed)
        self.max_num_pages = max_pages_per_req * max_bs * self.num_kv_heads
        self.max_num_pages_per_request = max_pages_per_req
        self.max_num_blocks = self.max_num_pages * self.num_blocks_per_page
        self.max_num_blocks_per_request = self.max_num_pages_per_request * self.num_blocks_per_page
        self.num_pages_per_workload = self.workload_chunk_size // self.num_blocks_per_page
        self.topk_val = sa.vortex_topk_val
        self.max_topk_val = sa.vortex_max_topk_val
        self.topk_ratio = sa.vortex_topk_ratio
        self.vortex_dtype = resolve_dtype(
            getattr(sa, "vortex_dtype", "bfloat16"), default=torch.bfloat16
        )

        self.block_reserved_bos = sa.vortex_block_reserved_bos
        self.block_reserved_eos = sa.vortex_block_reserved_eos

        self.max_num_workloads = (
            (self.max_num_blocks // max(1, self.workload_chunk_size)) + max_bs * self.num_kv_heads
        )

        self.tensor_list = []
        self.op_list = []
        self.output_tensor_to_op_list = []
        self.op_to_input_tensor_list = []
        self.op_to_output_tensor_list = []
        self.side_effect_op_ids = []
        self.tensor_id_to_tensor_name_map = {}
        self.compilation_header_lines = []
        self.auxilary_func_def_lines = []
        self.compilation_cache_dir = sa.vortex_compilation_cache_dir
        self.sparse_attention_name = parent.sparse_attention.__class__.__name__.lower() + f"_{uuid.uuid4().hex[:8]}"  # unique name for this attention instance
        self.impl_backend = "triton"  # default to triton; can be overridden by user
        self.vortex_attention_backend = getattr(
            sa, "vortex_attention_backend", "flashinfer"
        ) or "flashinfer"
        self._created = True
        return self


# Module-level singleton (part of the public package API)
ctx: Final[Context] = Context()


def get_ctx() -> Context:
    return ctx


__all__ = ["Context", "MetaData", "ctx", "get_ctx"]
