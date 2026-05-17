"""C ABI for the ``Union()`` output op (trtllm-only).

Merges two ``(block_table, seqlens)`` pairs into the final
``(sparse_block_tables, sparse_seqlens)`` pair. The exported symbol is
still named ``topk`` (matches the load_inline functions= entry); see
the per-leaf ``kernel.cu`` for the dedup + tail-placement algorithm.
"""

CPP_SOURCE = r"""
#include <torch/extension.h>

void topk(
    const at::Tensor& dense_seqlens,
    at::Tensor&       sparse_seqlens,
    const at::Tensor& dense_block_tables,
    const at::Tensor& block_tables_0,
    const at::Tensor& seqlens_0,
    const at::Tensor& block_tables_1,
    const at::Tensor& seqlens_1,
    at::Tensor&       sparse_block_tables,
    const int64_t     eff_batch_size,
    const int64_t     max_blocks_per_seq,
    const int64_t     block_size);
"""
