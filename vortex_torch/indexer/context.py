from __future__ import annotations
from typing import Any, Final, Union
import torch
from ..context_base import ContextBase
from ..utils import UNSET, Mode


class Context(ContextBase):
    """Mutable, single-instance context; populate later via .create(...)."""
    __slots__ =  ContextBase.__slots__ + (
        # indices / indptr
        "dense_kv_indices", "sparse_kv_indices", "dense_kv_indptr", "sparse_kv_indptr", "kv_last_page_len", "batch_size",
        # winfo
        "winfo_q_indices", "winfo_kv_offsets", "winfo_kv_lens", "winfo_num_workloads", "winfo_chunk_size", "max_num_workloads",
        # chunk limits
        "max_chunk_size", "min_chunk_size",
        # head / shape
        "group_size", "num_kv_heads", "num_qo_heads", "head_dim",
        # hardware / paging
        "num_sms", "page_size", "max_num_pages", "max_num_pages_per_request",
        # misc
        "indexer_dtype", "topk_val", "page_reserved_bos", "page_reserved_eos",
        
        # auxilary memory in graph
        "_aux_total_bytes",
    )

    def __init__(self) -> None:
        # Start as an empty shell (no big allocations).
        for name in self.__slots__:
            if name == "_created":
                object.__setattr__(self, name, False)
            elif name == "name":
                object.__setattr__(self, name, "Indexer")
            elif name == "_aux_total_bytes":
                object.__setattr__(self, name, 0)  # start from 0 bytes
            elif name == "batch_size":
                object.__setattr__(self, name, 0)
            elif name == "mode":
                object.__setattr__(self, name, Mode.profile) 
            else:
                object.__setattr__(self, name, UNSET)

    
    def set_batch_size(self, n: int) -> None:
        
        self.batch_size = n
        
    
    def create(self, parent: Any, model_runner: Any, *, overwrite: bool = False) -> "Context":
        """
        Populate this instance once (no locking). Set overwrite=True to allow re-init.
        NOTE: Without locking, concurrent callers may race; call from a single thread.
        """
        if self._created and not overwrite:
            raise RuntimeError("Context.create() already called; pass overwrite=True to reinitialize.")

        sa = model_runner.server_args
        max_pages_per_req = (
            (model_runner.model_config.context_len + sa.page_size - 1) // sa.page_size
            if sa.vortex_max_seq_lens < 0
            else (sa.vortex_max_seq_lens + sa.page_size - 1) // sa.page_size
        )

        max_seq_lengths = int(model_runner.model_config.context_len)
        max_bs = int(model_runner.req_to_token_pool.size)

        # Backend-known fields
        self.dense_kv_indices = parent.kv_indices_decode[0]
        self.sparse_kv_indices = parent.kv_indices_decode[1]
        self.dense_kv_indptr = parent.kv_indptr_decode[0]
        self.sparse_kv_indptr = parent.kv_indptr_decode[1]
        self.kv_last_page_len = parent.kv_last_page_len_decode

        self.max_chunk_size = sa.vortex_lb_max_chunk_size
        self.min_chunk_size = sa.vortex_lb_min_chunk_size

        self.group_size = parent.group_size
        self.num_kv_heads = parent.num_kv_heads
        self.num_qo_heads = parent.num_qo_heads
        self.head_dim = parent.head_dim

        self.num_sms = torch.cuda.get_device_properties(0).multi_processor_count
        self.page_size = sa.page_size

        # Capacity model (adjust as needed)
        pages_per_seq = (max_seq_lengths + self.page_size - 1) // self.page_size  # ceil
        self.max_num_pages = pages_per_seq * max_bs * self.num_kv_heads
        self.max_num_pages_per_request = max_pages_per_req

        self.topk_val = sa.vortex_topk_val
        dtype_str = getattr(sa, "indexer_dtype", "float32")
        if isinstance(dtype_str, str):
            self.indexer_dtype = getattr(torch, dtype_str, torch.float32)
        else:
            self.indexer_dtype = dtype_str
        
        self.page_reserved_bos = sa.vortex_page_reserved_bos
        self.page_reserved_eos = sa.vortex_page_reserved_eos

        self.max_num_workloads = (
            (self.max_num_pages // max(1, sa.vortex_lb_min_chunk_size)) + max_bs * self.num_kv_heads
        )

        device = getattr(model_runner, "device", "cpu")
        self.winfo_q_indices = torch.zeros((self.max_num_workloads,), dtype=torch.int32, device=device)
        self.winfo_kv_offsets = torch.zeros((self.max_num_workloads,), dtype=torch.int32, device=device)
        self.winfo_kv_lens = torch.zeros((self.max_num_workloads,), dtype=torch.int32, device=device)
        self.winfo_num_workloads = torch.zeros((1,), dtype=torch.int32, device=device)
        self.winfo_chunk_size = torch.zeros((1,), dtype=torch.int32, device=device)

        self._created = True
        return self



# Module-level singleton (part of the public package API)
ctx: Final[Context] = Context()

def get_ctx() -> Context:
    return ctx

__all__ = ["Context", "ctx", "get_ctx"]
