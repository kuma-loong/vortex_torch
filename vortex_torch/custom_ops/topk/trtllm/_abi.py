"""C ABI for the ``TopK(k)`` indexer op (trtllm-only).

Like :mod:`topk_output.trtllm._abi` but
  * ``sparse_seqlens`` is **mutable** (the kernel writes it);
  * ``topk_val`` is an explicit runtime argument;
  * row-too-small fallback (copy entire dense row) lives in the kernel.
"""

CPP_SOURCE = r"""
#include <torch/extension.h>

void topk(
    const at::Tensor& x,
    const at::Tensor& dense_seqlens,
    at::Tensor&       sparse_seqlens,
    const at::Tensor& dense_block_tables,
    at::Tensor&       sparse_block_tables,
    const int64_t     eff_batch_size,
    const int64_t     reserved_bos,
    const int64_t     reserved_eos,
    const int64_t     max_blocks_per_seq,
    const int64_t     block_size,
    const int64_t     topk_val);
"""
