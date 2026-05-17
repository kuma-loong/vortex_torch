import torch
from typing import Tuple
from .context import Context
from .planner_sglang import get_sglang_plan_decode_v2_module
from .prefill_sglang import get_sglang_prefill_module


def get_decode_planner(policy: str = None):

    module = get_sglang_plan_decode_v2_module(
        policy_body=policy,
        verbose=True,
        fallback_to_default=True,
    )
    def plan_decode(
        cached_seq_lens: torch.Tensor,
        req_to_token: torch.Tensor,
        req_indices: torch.Tensor,
        ctx: Context
    ):
        module.sglang_plan_decode_v2(
            cached_seq_lens,
            ctx.dense_kv_indptr,
            ctx.dense_kv_indices,
            ctx.sparse_kv_indptr,
            ctx.sparse_kv_indices,
            ctx.kv_last_page_len,
            req_to_token,
            req_indices,
            ctx.winfo_q_indices,
            ctx.winfo_is_first_workload_per_batch,
            ctx.winfo_kv_offsets,
            ctx.winfo_kv_lens,
            ctx.winfo_num_workloads,
            ctx.winfo_chunk_size,
            ctx.page_size,
            ctx.block_size,
            ctx.num_kv_heads,
            ctx.topk_val,
            ctx.topk_ratio,
            ctx.block_reserved_bos,
            ctx.block_reserved_eos,
            ctx.workload_chunk_size
        )

        ctx.set_batch_size(cached_seq_lens.shape[0])

    return plan_decode


def get_decode_planner_trtllm(policy: str = None):
    """Decode planner variant that emits trtllm-ready outputs directly.

    Outputs filled by the underlying CUDA kernel:
      * ``ctx.dense_block_tables``  — every selected page for the dense path
      * ``ctx.sparse_block_tables`` — only the BOS+EOS slots; the middle is
        filled by the topk kernel later
      * ``ctx.dense_seqlens`` / ``ctx.sparse_seqlens`` — int32 token counts
      * ``ctx.dense_kv_indptr`` / ``ctx.sparse_kv_indptr`` — still needed by
        the workload scheduler and by topk
      * ``ctx.dense_kv_indices`` — written in addition to
        ``dense_block_tables`` because the indexer's score-gather codegen
        still consults the CSR form (see TODO below)
      * ``ctx.kv_last_page_len`` — same semantics as before

    TODO(opt-b): teach the score-gather codegen (and any other indexer op
    that still reads ``ctx.dense_kv_indices`` in trtllm mode) to consume
    ``ctx.dense_block_tables`` instead. Once that's done, drop the
    ``dense_kv_indices`` write from the CUDA kernel and the argument from
    this Python wrapper — saves one int32-per-block store per plan call.
    """
    module = get_sglang_plan_decode_v2_module(
        policy_body=policy,
        verbose=True,
        fallback_to_default=True,
    )

    def plan_decode_trtllm(
        cached_seq_lens: torch.Tensor,
        req_to_token: torch.Tensor,
        req_indices: torch.Tensor,
        ctx: Context,
    ):
        module.sglang_plan_decode_v2_trtllm(
            cached_seq_lens,
            ctx.dense_kv_indptr,
            ctx.sparse_kv_indptr,
            ctx.dense_kv_indices,  # TODO(opt-b): drop with codegen migration.
            ctx.kv_last_page_len,
            ctx.dense_block_tables,
            ctx.sparse_block_tables,
            ctx.dense_seqlens,
            ctx.sparse_seqlens,
            req_to_token,
            req_indices,
            ctx.winfo_q_indices,
            ctx.winfo_is_first_workload_per_batch,
            ctx.winfo_kv_offsets,
            ctx.winfo_kv_lens,
            ctx.winfo_num_workloads,
            ctx.winfo_chunk_size,
            ctx.page_size,
            ctx.block_size,
            ctx.num_kv_heads,
            ctx.topk_val,
            ctx.topk_ratio,
            ctx.block_reserved_bos,
            ctx.block_reserved_eos,
            ctx.workload_chunk_size,
        )
        ctx.set_batch_size(cached_seq_lens.shape[0])

    return plan_decode_trtllm


def get_prefill_planner():
    """Mirror of :func:`get_decode_planner` for the prefill path.

    Triggers a one-time JIT compile of the prefill module on first call,
    then returns a closure that calls ``sglang_plan_prefill`` with the
    module reference baked in (no per-call lookup).
    """
    module = get_sglang_prefill_module()

    def plan_prefill(
        cached_seq_lens: torch.Tensor,
        dense_kv_indptr: torch.Tensor,
        dense_kv_indices: torch.Tensor,
        input_seq_lens: torch.Tensor,
        qo_indptr_ragged: torch.Tensor,
        qo_indptr_paged: torch.Tensor,
        kv_last_page_len: torch.Tensor,
        req_to_token: torch.Tensor,
        req_indices: torch.Tensor,
        batch_table: torch.Tensor,
        page_size: int,
        num_kv_heads: int,
    ):
        module.sglang_plan_prefill(
            cached_seq_lens,
            dense_kv_indptr,
            dense_kv_indices,
            input_seq_lens,
            qo_indptr_ragged,
            qo_indptr_paged,
            kv_last_page_len,
            req_to_token,
            req_indices,
            batch_table,
            page_size,
            num_kv_heads,
        )

    return plan_prefill


def get_chunkwise_nh2hn_transpose():
    """Factory for the ``Chunkwise_NH2HN_Transpose`` kernel; mirrors
    :func:`get_decode_planner`'s closure pattern so the module reference
    is bound once at backend init."""
    module = get_sglang_prefill_module()

    def chunkwise_nh2hn_transpose(
        x: torch.Tensor,
        indptr: torch.Tensor,
        batch_table: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> torch.Tensor:
        return module.Chunkwise_NH2HN_Transpose(
            x, indptr, batch_table, num_qo_heads, num_kv_heads, head_dim,
        )

    return chunkwise_nh2hn_transpose


def get_chunkwise_hn2nh_transpose():
    """Factory for the ``Chunkwise_HN2NH_Transpose`` kernel; mirrors
    :func:`get_decode_planner`'s closure pattern."""
    module = get_sglang_prefill_module()

    def chunkwise_hn2nh_transpose(
        x: torch.Tensor,
        y: torch.Tensor,
        indptr: torch.Tensor,
        batch_table: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return module.Chunkwise_HN2NH_Transpose(
            x, y, indptr, batch_table, num_qo_heads, num_kv_heads, head_dim,
        )

    return chunkwise_hn2nh_transpose

