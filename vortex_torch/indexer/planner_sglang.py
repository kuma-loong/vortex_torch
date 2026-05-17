import hashlib
import textwrap
import torch
from torch.utils.cpp_extension import load_inline

DEFAULT_POLICY_BODY = r"""
const int static_kv_budget = topk_val + block_reserved_bos + block_reserved_eos;
const int dynamic_kv_budget = int(cached_block_len * topk_ratio);
return max(static_kv_budget, dynamic_kv_budget);
"""

CPP_SOURCE = r"""
#include <torch/extension.h>

void sglang_plan_decode_v2(
    const at::Tensor& cached_seq_lens,
    at::Tensor& dense_kv_indptr,
    at::Tensor& dense_kv_indices,
    at::Tensor& sparse_kv_indptr,
    at::Tensor& sparse_kv_indices,
    at::Tensor& kv_last_page_len,
    const at::Tensor& req_to_token,
    const at::Tensor& req_indices,
    at::Tensor& winfo_q_indices,
    at::Tensor& winfo_is_first_workload_per_batch,
    at::Tensor& winfo_kv_offsets,
    at::Tensor& winfo_kv_lens,
    at::Tensor& winfo_num_workloads,
    at::Tensor& winfo_chunk_size,
    const int64_t page_size,
    const int64_t block_size,
    const int64_t num_kv_heads,
    const int64_t topk_val,
    const double topk_ratio,
    const int64_t block_reserved_bos,
    const int64_t block_reserved_eos,
    const int64_t workload_chunk_size
);

void sglang_plan_decode_v2_trtllm(
    const at::Tensor& cached_seq_lens,
    at::Tensor& kv_last_page_len,
    at::Tensor& dense_block_tables,
    at::Tensor& sparse_block_tables,
    at::Tensor& dense_seqlens,
    at::Tensor& sparse_seqlens,
    const at::Tensor& req_to_token,
    const at::Tensor& req_indices,
    at::Tensor& winfo_q_indices,
    at::Tensor& winfo_is_first_workload_per_batch,
    at::Tensor& winfo_kv_offsets,
    at::Tensor& winfo_kv_lens,
    at::Tensor& winfo_num_workloads,
    at::Tensor& winfo_chunk_size,
    const int64_t page_size,
    const int64_t block_size,
    const int64_t num_kv_heads,
    const int64_t topk_val,
    const double topk_ratio,
    const int64_t block_reserved_bos,
    const int64_t block_reserved_eos,
    const int64_t workload_chunk_size
);
"""

CUDA_TEMPLATE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <stdint.h>

struct PairSum {
  __device__ __forceinline__ int2 operator()(const int2& a, const int2& b) const {
    return make_int2(a.x + b.x, a.y + b.y);
  }
};

__device__ __forceinline__ int ComputeKvBudget(
    const int cached_block_len,
    const int topk_val,
    const float topk_ratio,
    const int block_reserved_bos,
    const int block_reserved_eos
) {
__POLICY_BODY__
}

__launch_bounds__(1024, 1)
__global__ void Sgl_Decode_Plan_Indptr_V2_Kernel(
    const int* __restrict__ cached_seq_lens,
    int* __restrict__ dense_kv_indptr,
    int* __restrict__ sparse_kv_indptr,
    const int batch_size,
    const int num_kv_heads,
    const int page_size,
    const int block_size,
    const int topk_val,
    const float topk_ratio,
    const int block_reserved_bos,
    const int block_reserved_eos
) {
    const int tx = threadIdx.x;
    const int cached_seq_len = (tx < batch_size) ? cached_seq_lens[tx] : 0;
    const int cached_block_len = (cached_seq_len + block_size - 1) / block_size;

    const int kv_budget = ComputeKvBudget(
        cached_block_len,
        topk_val,
        topk_ratio,
        block_reserved_bos,
        block_reserved_eos
    );

    using BlockScanInt2 = cub::BlockScan<int2, 1024>;

    __shared__ union {
        typename BlockScanInt2::TempStorage scan_int2;
    } temp;

    const int block_cnt = (tx < batch_size) ? cached_block_len : 0;
    const int sparse_cnt = (tx < batch_size) ? min(kv_budget, cached_block_len) : 0;

    int2 in = make_int2(block_cnt, sparse_cnt);
    int2 out;
    BlockScanInt2(temp.scan_int2).InclusiveScan(in, out, PairSum{});

    const int dense_cumsum = out.x;
    const int sparse_cumsum = out.y;

    if (tx < batch_size) {
        #pragma unroll
        for (int i = 0; i < num_kv_heads; ++i) {
            dense_kv_indptr[num_kv_heads * (tx + 1) - i] =
                dense_cumsum * num_kv_heads - i * cached_block_len;

            sparse_kv_indptr[num_kv_heads * (tx + 1) - i] =
                sparse_cumsum * num_kv_heads - i * min(kv_budget, cached_block_len);
        }
    }

    if (tx == 0) {
        dense_kv_indptr[0] = 0;
        sparse_kv_indptr[0] = 0;
    }
}

// IS_TRTLLM controls both per-row data source and ``winfo_kv_offsets[j]``
// encoding:
//   * false (flashinfer / CSR) — reads ``dense_kv_indptr`` / ``sparse_kv_indptr``
//                                (length ``eff_bs+1``) for per-row block
//                                counts; emits flat CSR offsets
//                                (``dense_kv_indptr[row] + col``) into
//                                ``dense_kv_indices``.
//   * true  (trtllm / BT)      — reads ``dense_seqlens`` / ``sparse_seqlens``
//                                (length ``eff_bs``, in **tokens**) and
//                                converts to block counts via
//                                ``ceil(tokens / block_size)``; emits
//                                linear BT offsets
//                                (``row * max_blocks_per_seq + col``) into
//                                the flattened ``dense_block_tables``.
// The Schedule.W kernel preamble does
//   ``page_idx_i32 = tl.load(indices + ragged_idx_i32)``
// and ``ragged_idx_i32 * shape[1]`` for RAGGED store/load addresses, so
// matching ``indices`` to the offset encoding is the *only* knob the
// consumer side needs to flip — see the matching launcher branch in
// ``vortex_torch/indexer/compiler/triton_impl/kernel_gen.py``.
template <int ITEM_PER_THREAD, bool IS_TRTLLM>
__launch_bounds__(1024, 1)
__global__ void Sgl_Decode_Plan_Workload_V2_Kernel(
    // CSR path consumes the first two as indptr (length eff_bs+1);
    // BT  path consumes the same slots as seqlens (length eff_bs, tokens).
    const int* __restrict__ row_a,
    const int* __restrict__ row_b,
    int* __restrict__ winfo_q_indices,
    uint8_t* __restrict__ winfo_is_first_workload_per_batch,
    int* __restrict__ winfo_kv_offsets,
    int* __restrict__ winfo_kv_lens,
    int* __restrict__ winfo_num_workloads,
    int* __restrict__ winfo_chunk_size,
    const int workload_chunk_size,
    const int eff_batch_size,
    const int block_reserved_bos,
    const int block_reserved_eos,
    const int max_blocks_per_seq,   // only used when IS_TRTLLM
    const int block_size            // only used when IS_TRTLLM
) {
    const int tx = threadIdx.x;
    using BlockScanInt = cub::BlockScan<int, 1024>;

    __shared__ union {
        typename BlockScanInt::TempStorage scan_int;
    } temp;

    uint16_t block_count[ITEM_PER_THREAD];
    int chunked_block_count_prefix_sum[ITEM_PER_THREAD + 1];
    const int tx_offset = tx * ITEM_PER_THREAD;

    chunked_block_count_prefix_sum[0] = 0;

    #pragma unroll
    for (int i = 0; i < ITEM_PER_THREAD; ++i) {
        int16_t w = 0;
        if ((tx_offset + i) < eff_batch_size) {
            int16_t dense_blocks_i;
            int16_t sparse_blocks_i;
            if constexpr (IS_TRTLLM) {
                // row_a = dense_seqlens (tokens), row_b = sparse_seqlens (tokens)
                const int dense_tokens  = row_a[tx_offset + i];
                const int sparse_tokens = row_b[tx_offset + i];
                dense_blocks_i  = int16_t((dense_tokens  + block_size - 1) / block_size);
                sparse_blocks_i = int16_t((sparse_tokens + block_size - 1) / block_size);
            } else {
                // row_a = dense_kv_indptr, row_b = sparse_kv_indptr (block-granular indptr)
                dense_blocks_i  = int16_t(row_a[tx_offset + i + 1] - row_a[tx_offset + i]);
                sparse_blocks_i = int16_t(row_b[tx_offset + i + 1] - row_b[tx_offset + i]);
            }

            if (dense_blocks_i > sparse_blocks_i) {
                w = dense_blocks_i - block_reserved_eos;
            } else {
                w = 0;
            }
        }
        block_count[i] = w;
        chunked_block_count_prefix_sum[i + 1] =
            int((block_count[i] + workload_chunk_size - 1) / workload_chunk_size);
    }

    BlockScanInt(temp.scan_int).InclusiveSum(
        chunked_block_count_prefix_sum,
        chunked_block_count_prefix_sum
    );

    if (tx == 1023) {
        *winfo_num_workloads = chunked_block_count_prefix_sum[ITEM_PER_THREAD];
        *winfo_chunk_size = workload_chunk_size;
    }

    #pragma unroll
    for (int i = 0; i < ITEM_PER_THREAD; ++i) {
        if ((tx_offset + i) < eff_batch_size) {
            const int start = chunked_block_count_prefix_sum[i];
            const int end = chunked_block_count_prefix_sum[i + 1];
            int last_len = int(block_count[i] % workload_chunk_size);
            if (last_len == 0) {
                last_len = workload_chunk_size;
            }

            const int row = tx_offset + i;
            // CSR base = flat prefix-sum into dense_kv_indices (row_a is indptr).
            // BT  base = row * max_blocks_per_seq (linear into flattened
            //            dense_block_tables, see Schedule.W launcher).
            int row_base;
            if constexpr (IS_TRTLLM) {
                row_base = row * max_blocks_per_seq;
            } else {
                row_base = row_a[row];   // row_a == dense_kv_indptr in CSR mode
            }
            for (int j = start; j < end; ++j) {
                winfo_q_indices[j] = row;
                winfo_kv_lens[j] = (j != end - 1) ? workload_chunk_size : last_len;
                winfo_kv_offsets[j] =
                    row_base + (j - start) * workload_chunk_size;
                winfo_is_first_workload_per_batch[j] = (j == start) ? 1 : 0;
            }
        }
    }
}

__global__ void Sgl_Decode_Plan_Indices_V2_Kernel(
    const int* __restrict__ req_to_token,
    const int64_t* __restrict__ req_indices,
    const int* __restrict__ cache_seq_lens,
    const int* __restrict__ dense_kv_indptr,
    const int* __restrict__ sparse_kv_indptr,
    int* __restrict__ dense_kv_indices,
    int* __restrict__ sparse_kv_indices,
    int* __restrict__ kv_last_block_len,
    const int req_to_token_stride,
    const int page_size,
    const int block_size,
    const int num_kv_heads,
    const int block_reserved_bos,
    const int block_reserved_eos
) {
    const int nblk = blockDim.x;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;

    const int64_t req_idx = req_indices[bx];
    const int* token_indices = req_to_token + req_idx * req_to_token_stride;

    const int kv_len = cache_seq_lens[bx];
    const int block_len = (kv_len + block_size - 1) / block_size;
    const int last_len = kv_len % block_size;
    const int num_blocks_per_page = page_size / block_size;

    if (tx == 0) {
        kv_last_block_len[bx * num_kv_heads + by] = (last_len == 0) ? block_size : last_len;
    }

    int* dense_output = dense_kv_indices + dense_kv_indptr[bx * num_kv_heads + by];
    int* sparse_output = sparse_kv_indices + sparse_kv_indptr[bx * num_kv_heads + by];

    const int kv_budget =
        sparse_kv_indptr[bx * num_kv_heads + by + 1] -
        sparse_kv_indptr[bx * num_kv_heads + by];

    int pos = tx;
    while (pos < block_len) {
        const int data = token_indices[pos * block_size];
        const int page_id = (data / page_size) * num_kv_heads + by;
        dense_output[pos] =
            page_id * num_blocks_per_page + (data % page_size) / block_size;
        pos += nblk;
    }

    if (block_len <= kv_budget) {
        int p = tx;
        while (p < block_len) {
            sparse_output[p] = dense_output[p];
            p += nblk;
        }
    } else {
        int p = tx;
        while (p < block_reserved_bos) {
            sparse_output[p] = dense_output[p];
            p += nblk;
        }

        p = tx;
        while (p < block_reserved_eos) {
            const int data = token_indices[(block_len - p - 1) * block_size];
            const int page_id = (data / page_size) * num_kv_heads + by;
            sparse_output[kv_budget - p - 1] =
                page_id * num_blocks_per_page + (data % page_size) / block_size;
            p += nblk;
        }
    }
}

void sglang_plan_decode_v2(
    const at::Tensor& cached_seq_lens,
    at::Tensor& dense_kv_indptr,
    at::Tensor& dense_kv_indices,
    at::Tensor& sparse_kv_indptr,
    at::Tensor& sparse_kv_indices,
    at::Tensor& kv_last_page_len,
    const at::Tensor& req_to_token,
    const at::Tensor& req_indices,
    at::Tensor& winfo_q_indices,
    at::Tensor& winfo_is_first_workload_per_batch,
    at::Tensor& winfo_kv_offsets,
    at::Tensor& winfo_kv_lens,
    at::Tensor& winfo_num_workloads,
    at::Tensor& winfo_chunk_size,
    const int64_t page_size,
    const int64_t block_size,
    const int64_t num_kv_heads,
    const int64_t topk_val,
    const double topk_ratio,
    const int64_t block_reserved_bos,
    const int64_t block_reserved_eos,
    const int64_t workload_chunk_size
) {
    TORCH_CHECK(cached_seq_lens.is_cuda(), "cached_seq_lens must be a CUDA tensor");
    TORCH_CHECK(dense_kv_indptr.is_cuda(), "dense_kv_indptr must be a CUDA tensor");
    TORCH_CHECK(dense_kv_indices.is_cuda(), "dense_kv_indices must be a CUDA tensor");
    TORCH_CHECK(sparse_kv_indptr.is_cuda(), "sparse_kv_indptr must be a CUDA tensor");
    TORCH_CHECK(sparse_kv_indices.is_cuda(), "sparse_kv_indices must be a CUDA tensor");
    TORCH_CHECK(kv_last_page_len.is_cuda(), "kv_last_page_len must be a CUDA tensor");
    TORCH_CHECK(req_to_token.is_cuda(), "req_to_token must be a CUDA tensor");
    TORCH_CHECK(req_indices.is_cuda(), "req_indices must be a CUDA tensor");
    TORCH_CHECK(winfo_q_indices.is_cuda(), "winfo_q_indices must be a CUDA tensor");
    TORCH_CHECK(winfo_is_first_workload_per_batch.is_cuda(), "winfo_is_first_workload_per_batch must be a CUDA tensor");
    TORCH_CHECK(winfo_kv_offsets.is_cuda(), "winfo_kv_offsets must be a CUDA tensor");
    TORCH_CHECK(winfo_kv_lens.is_cuda(), "winfo_kv_lens must be a CUDA tensor");
    TORCH_CHECK(winfo_num_workloads.is_cuda(), "winfo_num_workloads must be a CUDA tensor");
    TORCH_CHECK(winfo_chunk_size.is_cuda(), "winfo_chunk_size must be a CUDA tensor");

    TORCH_CHECK(req_to_token.dtype() == torch::kInt32, "req_to_token must be int32");
    TORCH_CHECK(req_indices.dtype() == torch::kInt64, "req_indices must be int64");
    TORCH_CHECK(cached_seq_lens.dtype() == torch::kInt32, "cached_seq_lens must be int32");
    TORCH_CHECK(dense_kv_indptr.dtype() == torch::kInt32, "dense_kv_indptr must be int32");
    TORCH_CHECK(dense_kv_indices.dtype() == torch::kInt32, "dense_kv_indices must be int32");
    TORCH_CHECK(sparse_kv_indptr.dtype() == torch::kInt32, "sparse_kv_indptr must be int32");
    TORCH_CHECK(sparse_kv_indices.dtype() == torch::kInt32, "sparse_kv_indices must be int32");
    TORCH_CHECK(kv_last_page_len.dtype() == torch::kInt32, "kv_last_page_len must be int32");
    TORCH_CHECK(winfo_q_indices.dtype() == torch::kInt32, "winfo_q_indices must be int32");
    TORCH_CHECK(winfo_is_first_workload_per_batch.dtype() == torch::kUInt8, "winfo_is_first_workload_per_batch must be uint8");
    TORCH_CHECK(winfo_kv_offsets.dtype() == torch::kInt32, "winfo_kv_offsets must be int32");
    TORCH_CHECK(winfo_kv_lens.dtype() == torch::kInt32, "winfo_kv_lens must be int32");
    TORCH_CHECK(winfo_num_workloads.dtype() == torch::kInt32, "winfo_num_workloads must be int32");
    TORCH_CHECK(winfo_chunk_size.dtype() == torch::kInt32, "winfo_chunk_size must be int32");

    TORCH_CHECK(page_size >= 1, "page_size must be >= 1");
    TORCH_CHECK(block_size >= 1, "block_size must be >= 1");
    TORCH_CHECK(page_size % block_size == 0, "page_size must be divisible by block_size");
    TORCH_CHECK(num_kv_heads >= 1, "num_kv_heads must be >= 1");
    TORCH_CHECK(topk_val >= 1, "topk_val must be >= 1");
    TORCH_CHECK(topk_ratio >= 0.0 && topk_ratio <= 1.0, "topk_ratio must be in [0, 1]");
    TORCH_CHECK(block_reserved_bos >= 0, "block_reserved_bos must be >= 0");
    TORCH_CHECK(block_reserved_eos >= 1, "block_reserved_eos must be >= 1");
    TORCH_CHECK(workload_chunk_size >= 1, "workload_chunk_size must be >= 1");

    const int batch_size = static_cast<int>(cached_seq_lens.size(0));
    const int eff_batch_size = batch_size * static_cast<int>(num_kv_heads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK(batch_size <= 1024,
        "Sgl_Decode_Plan_Indptr_V2_Kernel assumes batch_size <= 1024");

    Sgl_Decode_Plan_Indptr_V2_Kernel<<<1, 1024, 0, stream>>>(
        cached_seq_lens.data_ptr<int>(),
        dense_kv_indptr.data_ptr<int>(),
        sparse_kv_indptr.data_ptr<int>(),
        batch_size,
        static_cast<int>(num_kv_heads),
        static_cast<int>(page_size),
        static_cast<int>(block_size),
        static_cast<int>(topk_val),
        static_cast<float>(topk_ratio),
        static_cast<int>(block_reserved_bos),
        static_cast<int>(block_reserved_eos)
    );

    // CSR path: pass IS_TRTLLM=false and 0 for max_blocks_per_seq (unused).
    if (eff_batch_size <= 1024) {
        Sgl_Decode_Plan_Workload_V2_Kernel<1, false><<<1, 1024, 0, stream>>>(
            dense_kv_indptr.data_ptr<int>(),
            sparse_kv_indptr.data_ptr<int>(),
            winfo_q_indices.data_ptr<int>(),
            winfo_is_first_workload_per_batch.data_ptr<uint8_t>(),
            winfo_kv_offsets.data_ptr<int>(),
            winfo_kv_lens.data_ptr<int>(),
            winfo_num_workloads.data_ptr<int>(),
            winfo_chunk_size.data_ptr<int>(),
            static_cast<int>(workload_chunk_size),
            eff_batch_size,
            static_cast<int>(block_reserved_bos),
            static_cast<int>(block_reserved_eos),
            0,
            0
        );
    } else if (eff_batch_size <= 2048) {
        Sgl_Decode_Plan_Workload_V2_Kernel<2, false><<<1, 1024, 0, stream>>>(
            dense_kv_indptr.data_ptr<int>(),
            sparse_kv_indptr.data_ptr<int>(),
            winfo_q_indices.data_ptr<int>(),
            winfo_is_first_workload_per_batch.data_ptr<uint8_t>(),
            winfo_kv_offsets.data_ptr<int>(),
            winfo_kv_lens.data_ptr<int>(),
            winfo_num_workloads.data_ptr<int>(),
            winfo_chunk_size.data_ptr<int>(),
            static_cast<int>(workload_chunk_size),
            eff_batch_size,
            static_cast<int>(block_reserved_bos),
            static_cast<int>(block_reserved_eos),
            0,
            0
        );
    } else if (eff_batch_size <= 4096) {
        Sgl_Decode_Plan_Workload_V2_Kernel<4, false><<<1, 1024, 0, stream>>>(
            dense_kv_indptr.data_ptr<int>(),
            sparse_kv_indptr.data_ptr<int>(),
            winfo_q_indices.data_ptr<int>(),
            winfo_is_first_workload_per_batch.data_ptr<uint8_t>(),
            winfo_kv_offsets.data_ptr<int>(),
            winfo_kv_lens.data_ptr<int>(),
            winfo_num_workloads.data_ptr<int>(),
            winfo_chunk_size.data_ptr<int>(),
            static_cast<int>(workload_chunk_size),
            eff_batch_size,
            static_cast<int>(block_reserved_bos),
            static_cast<int>(block_reserved_eos),
            0,
            0
        );
    } else if (eff_batch_size <= 8192) {
        Sgl_Decode_Plan_Workload_V2_Kernel<8, false><<<1, 1024, 0, stream>>>(
            dense_kv_indptr.data_ptr<int>(),
            sparse_kv_indptr.data_ptr<int>(),
            winfo_q_indices.data_ptr<int>(),
            winfo_is_first_workload_per_batch.data_ptr<uint8_t>(),
            winfo_kv_offsets.data_ptr<int>(),
            winfo_kv_lens.data_ptr<int>(),
            winfo_num_workloads.data_ptr<int>(),
            winfo_chunk_size.data_ptr<int>(),
            static_cast<int>(workload_chunk_size),
            eff_batch_size,
            static_cast<int>(block_reserved_bos),
            static_cast<int>(block_reserved_eos),
            0,
            0
        );
    } else {
        TORCH_CHECK(false, "eff_batch_size > 8192 is not supported");
    }

    const int req_to_token_stride = static_cast<int>(req_to_token.size(1));
    dim3 nblks(batch_size, static_cast<unsigned int>(num_kv_heads));
    dim3 nthrs(512);

    Sgl_Decode_Plan_Indices_V2_Kernel<<<nblks, nthrs, 0, stream>>>(
        req_to_token.data_ptr<int>(),
        req_indices.data_ptr<int64_t>(),
        cached_seq_lens.data_ptr<int>(),
        dense_kv_indptr.data_ptr<int>(),
        sparse_kv_indptr.data_ptr<int>(),
        dense_kv_indices.data_ptr<int>(),
        sparse_kv_indices.data_ptr<int>(),
        kv_last_page_len.data_ptr<int>(),
        req_to_token_stride,
        static_cast<int>(page_size),
        static_cast<int>(block_size),
        static_cast<int>(num_kv_heads),
        static_cast<int>(block_reserved_bos),
        static_cast<int>(block_reserved_eos)
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// trtllm-flavoured planner (indptr-free): writes
//   - dense_block_tables  : [eff_bs, max_blocks_per_seq] int32
//   - sparse_block_tables : [eff_bs, max_blocks_per_seq] int32  (BOS+EOS only)
//   - dense_seqlens       : [eff_bs] int32   (tokens)
//   - sparse_seqlens      : [eff_bs] int32   (tokens)
//   - kv_last_page_len    : [eff_bs] int32
// where eff_bs == batch_size * num_kv_heads. The middle slots of every
// sparse_block_tables row (between BOS and EOS) are intentionally NOT touched
// here — they are filled by the topk kernel.
// ---------------------------------------------------------------------------

// (1) Seqlens kernel: emits ``dense_seqlens`` / ``sparse_seqlens`` (tokens)
// for every (req, kv_head) row. Must run BEFORE the workload scheduler in
// trtllm mode, since the trtllm workload scheduler reads seqlens to derive
// per-row block counts (replacing the CSR-mode indptr diff).
__launch_bounds__(1024, 1)
__global__ void Sgl_Decode_Plan_Seqlens_V2_Trtllm_Kernel(
    const int* __restrict__ cached_seq_lens,
    int* __restrict__ dense_seqlens,
    int* __restrict__ sparse_seqlens,
    const int batch_size,
    const int num_kv_heads,
    const int block_size,
    const int topk_val,
    const float topk_ratio,
    const int block_reserved_bos,
    const int block_reserved_eos
) {
    const int tx = threadIdx.x;
    if (tx >= batch_size) return;

    const int cached_seq_len = cached_seq_lens[tx];
    const int block_len = (cached_seq_len + block_size - 1) / block_size;
    const int kv_budget_raw = ComputeKvBudget(
        block_len, topk_val, topk_ratio,
        block_reserved_bos, block_reserved_eos);
    const int sparse_block_len = (kv_budget_raw < block_len) ? kv_budget_raw : block_len;

    const int last_len_mod = cached_seq_len % block_size;
    const int last_block_len = (last_len_mod == 0) ? block_size : last_len_mod;

    const int dense_tokens  = cached_seq_len;
    const int sparse_tokens = (sparse_block_len > 0)
        ? ((sparse_block_len - 1) * block_size + last_block_len)
        : 0;

    // Replicate per (req, kv_head). Layout matches ``row = bx*kv_heads + by``.
    #pragma unroll
    for (int i = 0; i < num_kv_heads; ++i) {
        const int row = tx * num_kv_heads + i;
        dense_seqlens[row]  = dense_tokens;
        sparse_seqlens[row] = sparse_tokens;
    }
}

// (3) Indices kernel: emits block_tables + kv_last_page_len. Indptr-free —
// re-derives ``kv_budget`` from the policy locally rather than reading from
// a planner-emitted indptr buffer.
__global__ void Sgl_Decode_Plan_Indices_V2_Trtllm_Kernel(
    const int* __restrict__ req_to_token,
    const int64_t* __restrict__ req_indices,
    const int* __restrict__ cache_seq_lens,
    int* __restrict__ dense_block_tables,
    int* __restrict__ sparse_block_tables,
    int* __restrict__ kv_last_block_len,
    const int max_blocks_per_seq,
    const int req_to_token_stride,
    const int page_size,
    const int block_size,
    const int num_kv_heads,
    const int topk_val,
    const float topk_ratio,
    const int block_reserved_bos,
    const int block_reserved_eos
) {
    const int nblk = blockDim.x;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int row = bx * num_kv_heads + by;

    const int64_t req_idx = req_indices[bx];
    const int* token_indices = req_to_token + req_idx * req_to_token_stride;

    const int kv_len = cache_seq_lens[bx];
    const int block_len = (kv_len + block_size - 1) / block_size;
    const int last_len_mod = kv_len % block_size;
    const int last_block_len = (last_len_mod == 0) ? block_size : last_len_mod;
    const int num_blocks_per_page = page_size / block_size;

    const int kv_budget_raw = ComputeKvBudget(
        block_len, topk_val, topk_ratio,
        block_reserved_bos, block_reserved_eos);
    const int kv_budget = (kv_budget_raw < block_len) ? kv_budget_raw : block_len;

    if (tx == 0) {
        kv_last_block_len[row] = last_block_len;
    }

    int* dense_output  = dense_block_tables  + row * max_blocks_per_seq;
    int* sparse_output = sparse_block_tables + row * max_blocks_per_seq;

    // ---- Dense: write every block_len entry. ----------------------------
    int pos = tx;
    while (pos < block_len) {
        const int data = token_indices[pos * block_size];
        const int page_id = (data / page_size) * num_kv_heads + by;
        const int block_id =
            page_id * num_blocks_per_page + (data % page_size) / block_size;
        dense_output[pos] = block_id;
        pos += nblk;
    }

    // ---- Sparse: BOS + EOS (or full copy if block_len <= kv_budget). -----
    if (block_len <= kv_budget) {
        int p = tx;
        while (p < block_len) {
            sparse_output[p] = dense_output[p];
            p += nblk;
        }
    } else {
        // BOS at the front of the row.
        int p = tx;
        while (p < block_reserved_bos) {
            sparse_output[p] = dense_output[p];
            p += nblk;
        }

        // EOS at the tail of the row, written in reverse so slot
        // (kv_budget - 1) holds the last real KV block.
        p = tx;
        while (p < block_reserved_eos) {
            const int data = token_indices[(block_len - p - 1) * block_size];
            const int page_id = (data / page_size) * num_kv_heads + by;
            sparse_output[kv_budget - p - 1] =
                page_id * num_blocks_per_page + (data % page_size) / block_size;
            p += nblk;
        }
        // Middle slots [block_reserved_bos, kv_budget - block_reserved_eos)
        // are deliberately left untouched — topk fills them later.
    }
}

void sglang_plan_decode_v2_trtllm(
    const at::Tensor& cached_seq_lens,
    at::Tensor& kv_last_page_len,
    at::Tensor& dense_block_tables,
    at::Tensor& sparse_block_tables,
    at::Tensor& dense_seqlens,
    at::Tensor& sparse_seqlens,
    const at::Tensor& req_to_token,
    const at::Tensor& req_indices,
    at::Tensor& winfo_q_indices,
    at::Tensor& winfo_is_first_workload_per_batch,
    at::Tensor& winfo_kv_offsets,
    at::Tensor& winfo_kv_lens,
    at::Tensor& winfo_num_workloads,
    at::Tensor& winfo_chunk_size,
    const int64_t page_size,
    const int64_t block_size,
    const int64_t num_kv_heads,
    const int64_t topk_val,
    const double topk_ratio,
    const int64_t block_reserved_bos,
    const int64_t block_reserved_eos,
    const int64_t workload_chunk_size
) {
    TORCH_CHECK(cached_seq_lens.is_cuda(), "cached_seq_lens must be a CUDA tensor");
    TORCH_CHECK(kv_last_page_len.is_cuda(), "kv_last_page_len must be a CUDA tensor");
    TORCH_CHECK(dense_block_tables.is_cuda(), "dense_block_tables must be a CUDA tensor");
    TORCH_CHECK(sparse_block_tables.is_cuda(), "sparse_block_tables must be a CUDA tensor");
    TORCH_CHECK(dense_seqlens.is_cuda(), "dense_seqlens must be a CUDA tensor");
    TORCH_CHECK(sparse_seqlens.is_cuda(), "sparse_seqlens must be a CUDA tensor");
    TORCH_CHECK(req_to_token.is_cuda(), "req_to_token must be a CUDA tensor");
    TORCH_CHECK(req_indices.is_cuda(), "req_indices must be a CUDA tensor");
    TORCH_CHECK(winfo_q_indices.is_cuda(), "winfo_q_indices must be a CUDA tensor");
    TORCH_CHECK(winfo_is_first_workload_per_batch.is_cuda(), "winfo_is_first_workload_per_batch must be a CUDA tensor");
    TORCH_CHECK(winfo_kv_offsets.is_cuda(), "winfo_kv_offsets must be a CUDA tensor");
    TORCH_CHECK(winfo_kv_lens.is_cuda(), "winfo_kv_lens must be a CUDA tensor");
    TORCH_CHECK(winfo_num_workloads.is_cuda(), "winfo_num_workloads must be a CUDA tensor");
    TORCH_CHECK(winfo_chunk_size.is_cuda(), "winfo_chunk_size must be a CUDA tensor");

    TORCH_CHECK(req_to_token.dtype() == torch::kInt32, "req_to_token must be int32");
    TORCH_CHECK(req_indices.dtype() == torch::kInt64, "req_indices must be int64");
    TORCH_CHECK(cached_seq_lens.dtype() == torch::kInt32, "cached_seq_lens must be int32");
    TORCH_CHECK(kv_last_page_len.dtype() == torch::kInt32, "kv_last_page_len must be int32");
    TORCH_CHECK(dense_block_tables.dtype() == torch::kInt32, "dense_block_tables must be int32");
    TORCH_CHECK(sparse_block_tables.dtype() == torch::kInt32, "sparse_block_tables must be int32");
    TORCH_CHECK(dense_seqlens.dtype() == torch::kInt32, "dense_seqlens must be int32");
    TORCH_CHECK(sparse_seqlens.dtype() == torch::kInt32, "sparse_seqlens must be int32");
    TORCH_CHECK(winfo_q_indices.dtype() == torch::kInt32, "winfo_q_indices must be int32");
    TORCH_CHECK(winfo_is_first_workload_per_batch.dtype() == torch::kUInt8, "winfo_is_first_workload_per_batch must be uint8");
    TORCH_CHECK(winfo_kv_offsets.dtype() == torch::kInt32, "winfo_kv_offsets must be int32");
    TORCH_CHECK(winfo_kv_lens.dtype() == torch::kInt32, "winfo_kv_lens must be int32");
    TORCH_CHECK(winfo_num_workloads.dtype() == torch::kInt32, "winfo_num_workloads must be int32");
    TORCH_CHECK(winfo_chunk_size.dtype() == torch::kInt32, "winfo_chunk_size must be int32");

    TORCH_CHECK(dense_block_tables.dim() == 2, "dense_block_tables must be 2D");
    TORCH_CHECK(sparse_block_tables.dim() == 2, "sparse_block_tables must be 2D");
    TORCH_CHECK(dense_block_tables.size(1) == sparse_block_tables.size(1),
                "dense / sparse block_tables must share max_blocks_per_seq");

    TORCH_CHECK(page_size >= 1, "page_size must be >= 1");
    TORCH_CHECK(block_size >= 1, "block_size must be >= 1");
    TORCH_CHECK(page_size % block_size == 0, "page_size must be divisible by block_size");
    TORCH_CHECK(num_kv_heads >= 1, "num_kv_heads must be >= 1");
    TORCH_CHECK(topk_val >= 1, "topk_val must be >= 1");
    TORCH_CHECK(topk_ratio >= 0.0 && topk_ratio <= 1.0, "topk_ratio must be in [0, 1]");
    TORCH_CHECK(block_reserved_bos >= 0, "block_reserved_bos must be >= 0");
    TORCH_CHECK(block_reserved_eos >= 1, "block_reserved_eos must be >= 1");
    TORCH_CHECK(workload_chunk_size >= 1, "workload_chunk_size must be >= 1");

    const int batch_size = static_cast<int>(cached_seq_lens.size(0));
    const int eff_batch_size = batch_size * static_cast<int>(num_kv_heads);
    const int max_blocks_per_seq = static_cast<int>(dense_block_tables.size(1));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK(batch_size <= 1024,
        "Sgl_Decode_Plan_Seqlens_V2_Trtllm_Kernel assumes batch_size <= 1024");

    // (1) Seqlens — writes dense_seqlens / sparse_seqlens (tokens) for the
    // workload scheduler in (2) and the trtllm topk / decode kernels at
    // forward time. Replaces the CSR-mode Indptr kernel; no indptr is ever
    // emitted in trtllm mode.
    Sgl_Decode_Plan_Seqlens_V2_Trtllm_Kernel<<<1, 1024, 0, stream>>>(
        cached_seq_lens.data_ptr<int>(),
        dense_seqlens.data_ptr<int>(),
        sparse_seqlens.data_ptr<int>(),
        batch_size,
        static_cast<int>(num_kv_heads),
        static_cast<int>(block_size),
        static_cast<int>(topk_val),
        static_cast<float>(topk_ratio),
        static_cast<int>(block_reserved_bos),
        static_cast<int>(block_reserved_eos)
    );

    // (2) Workload — IS_TRTLLM=true reads seqlens (in tokens), converts
    // to blocks via ``ceil(tokens / block_size)``, and bakes
    // ``winfo_kv_offsets[j] = row*max_blocks_per_seq + col`` so the
    // Schedule.W kernel preamble (with ``indices = dense_block_tables.view(-1)``)
    // resolves page ids correctly.
    auto launch_workload = [&](auto items_per_thread) {
        constexpr int IPT = decltype(items_per_thread)::value;
        Sgl_Decode_Plan_Workload_V2_Kernel<IPT, true><<<1, 1024, 0, stream>>>(
            dense_seqlens.data_ptr<int>(),
            sparse_seqlens.data_ptr<int>(),
            winfo_q_indices.data_ptr<int>(),
            winfo_is_first_workload_per_batch.data_ptr<uint8_t>(),
            winfo_kv_offsets.data_ptr<int>(),
            winfo_kv_lens.data_ptr<int>(),
            winfo_num_workloads.data_ptr<int>(),
            winfo_chunk_size.data_ptr<int>(),
            static_cast<int>(workload_chunk_size),
            eff_batch_size,
            static_cast<int>(block_reserved_bos),
            static_cast<int>(block_reserved_eos),
            max_blocks_per_seq,
            static_cast<int>(block_size)
        );
    };
    if (eff_batch_size <= 1024) {
        launch_workload(std::integral_constant<int, 1>{});
    } else if (eff_batch_size <= 2048) {
        launch_workload(std::integral_constant<int, 2>{});
    } else if (eff_batch_size <= 4096) {
        launch_workload(std::integral_constant<int, 4>{});
    } else if (eff_batch_size <= 8192) {
        launch_workload(std::integral_constant<int, 8>{});
    } else {
        TORCH_CHECK(false, "eff_batch_size > 8192 is not supported");
    }

    // (3) Indices — block_tables + kv_last_page_len. No CSR side-output.
    const int req_to_token_stride = static_cast<int>(req_to_token.size(1));
    dim3 nblks(batch_size, static_cast<unsigned int>(num_kv_heads));
    dim3 nthrs(512);

    Sgl_Decode_Plan_Indices_V2_Trtllm_Kernel<<<nblks, nthrs, 0, stream>>>(
        req_to_token.data_ptr<int>(),
        req_indices.data_ptr<int64_t>(),
        cached_seq_lens.data_ptr<int>(),
        dense_block_tables.data_ptr<int>(),
        sparse_block_tables.data_ptr<int>(),
        kv_last_page_len.data_ptr<int>(),
        max_blocks_per_seq,
        req_to_token_stride,
        static_cast<int>(page_size),
        static_cast<int>(block_size),
        static_cast<int>(num_kv_heads),
        static_cast<int>(topk_val),
        static_cast<float>(topk_ratio),
        static_cast<int>(block_reserved_bos),
        static_cast<int>(block_reserved_eos)
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

_MODULE_CACHE = {}


def _normalize_policy_body(policy_body: str) -> str:
    body = textwrap.dedent(policy_body).strip()
    if not body:
        raise ValueError("policy_body must not be empty")
    return textwrap.indent(body, "    ")


def _make_cuda_source(policy_body: str) -> str:
    return CUDA_TEMPLATE.replace("__POLICY_BODY__", _normalize_policy_body(policy_body))


def _module_name_from_policy(policy_body: str) -> str:
    digest = hashlib.sha256(policy_body.encode("utf-8")).hexdigest()[:16]
    return f"sglang_plan_decode_v2_ext_{digest}"


def _compile_module(policy_body: str, verbose: bool = True):
    module_name = _module_name_from_policy(policy_body)

    if module_name in _MODULE_CACHE:
        return _MODULE_CACHE[module_name]

    cuda_source = _make_cuda_source(policy_body)

    module = load_inline(
        name=module_name,
        cpp_sources=CPP_SOURCE,
        cuda_sources=cuda_source,
        functions=["sglang_plan_decode_v2", "sglang_plan_decode_v2_trtllm"],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
        ],
        with_cuda=True,
        verbose=verbose,
    )
    _MODULE_CACHE[module_name] = module
    return module


def get_sglang_plan_decode_v2_module(
    policy_body: str | None = None,
    verbose: bool = True,
    fallback_to_default: bool = True,
):
    effective_policy = DEFAULT_POLICY_BODY if policy_body is None else policy_body

    try:
        return _compile_module(effective_policy, verbose=verbose)
    except Exception:
        if (policy_body is None) or (not fallback_to_default):
            raise
        return _compile_module(DEFAULT_POLICY_BODY, verbose=verbose)


