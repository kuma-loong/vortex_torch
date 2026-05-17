"""C ABI for topk_output / trtllm (block-table layout).

Indptr-free: per-row token counts come in via ``dense_seqlens`` /
``sparse_seqlens`` (length ``eff_bs``); the block stride is the
explicit ``block_size``. The output index buffer is a 2D
``[eff_bs, max_blocks_per_seq]`` block-table.
"""

CPP_SOURCE = r"""
#include <torch/extension.h>

void topk(
    const at::Tensor& x,
    const at::Tensor& dense_seqlens,
    const at::Tensor& sparse_seqlens,
    const at::Tensor& dense_block_tables,
    at::Tensor&       sparse_block_tables,
    const int64_t     eff_batch_size,
    const int64_t     reserved_bos,
    const int64_t     reserved_eos,
    const int64_t     max_blocks_per_seq,
    const int64_t     block_size);
"""
