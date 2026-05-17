// Union of two (block_table, seqlens) pairs for the trtllm backend.
//
// Powers the ``Union()`` output op. Per row ``bx`` it:
//
//   1) Derives ``last_block_id = dense_block_tables[bx, dense_block_len-1]``
//      (the dense path's true last block) and the partial token count of
//      that block (``last_block_len = dense_seqlens[bx] mod block_size``,
//      or ``block_size`` when the remainder is zero).
//   2) Walks ``block_tables_0[bx, :blocks_0]`` then ``block_tables_1[bx, :blocks_1]``
//      where ``blocks_i = ceil(seqlens_i[bx] / block_size)``.
//      Each scanned block id (excluding ``last_block_id``) is added to a
//      shared-memory dedup buffer if not already present.
//   3) Writes the deduped ids into ``sparse_block_tables[bx, 0:u)``,
//      then places ``last_block_id`` at position ``u`` so trtllm decode
//      sees the partial block at the tail of the table.
//   4) Sets ``sparse_seqlens[bx] = u * block_size + last_block_len``.
//
// Edge case: if either input row's last entry would have been the dense
// last block (which TopK(k) always places at the tail), step (2)'s
// ``id == last_block_id`` filter drops the duplicate, then step (3)
// re-inserts a single copy at the end. So the union is "sequence-tail
// canonical": exactly one occurrence of the true last block, sitting at
// the last slot, with the matching partial ``last_block_len``.

#include <ATen/core/TensorBase.h>
#include <ATen/core/TensorBody.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Exception.h>
#include <cuda.h>

// Capacity of the per-row shared-memory dedup buffer. Must be ≥ the
// largest possible union size for any row in the workload. The strict
// upper bound is ``max_blocks_per_seq`` (every dense block selected
// uniquely), so this value caps the addressable context length:
// ``block_size * VORTEX_UNION_MAX``. 4096 int32 slots = 16KB of shared
// memory and covers contexts up to ``block_size * 4096`` tokens
// (≈ 64K with block_size=16). Bump and recompile for longer contexts.
#ifndef VORTEX_UNION_MAX
#define VORTEX_UNION_MAX 4096
#endif


template <int NUM_THREADS>
__global__ void Union_Kernel(
const int* __restrict__ dense_seqlens,        // [eff_bs] tokens
int*       __restrict__ sparse_seqlens,       // [eff_bs] tokens (OUTPUT)
const int* __restrict__ dense_block_tables,   // [eff_bs, row_stride]
const int* __restrict__ block_tables_0,       // [eff_bs, row_stride]
const int* __restrict__ seqlens_0,            // [eff_bs] tokens
const int* __restrict__ block_tables_1,       // [eff_bs, row_stride]
const int* __restrict__ seqlens_1,            // [eff_bs] tokens
int*       __restrict__ sparse_block_tables,  // [eff_bs, row_stride] (OUTPUT)
const int  row_stride,                        // = max_blocks_per_seq
const int  block_size)
{
    const int bx = blockIdx.x;
    const int tx = threadIdx.x;

    const int dense_tokens = dense_seqlens[bx];
    if (dense_tokens <= 0) {
        if (tx == 0) sparse_seqlens[bx] = 0;
        return;
    }
    const int dense_block_len = (dense_tokens + block_size - 1) / block_size;
    const int last_mod = dense_tokens % block_size;
    const int last_block_len = (last_mod == 0) ? block_size : last_mod;

    const int row_offset = bx * row_stride;
    const int* __restrict__ in0 = block_tables_0 + row_offset;
    const int* __restrict__ in1 = block_tables_1 + row_offset;
    int*       __restrict__ out = sparse_block_tables + row_offset;
    const int last_block_id =
        dense_block_tables[row_offset + dense_block_len - 1];

    const int tokens0 = seqlens_0[bx];
    const int tokens1 = seqlens_1[bx];
    const int blocks0 = (tokens0 + block_size - 1) / block_size;
    const int blocks1 = (tokens1 + block_size - 1) / block_size;

    __shared__ int s_union[VORTEX_UNION_MAX];
    __shared__ int s_count;
    if (tx == 0) s_count = 0;
    __syncthreads();

    // Single-threaded dedup. The per-row workload is small (typically
    // bos + 2*k + eos with k <= 256, so well under 600 candidates and
    // a quadratic walk is still trivially fast on-device); switch to a
    // parallel dedup if you start pushing k past a few hundred.
    if (tx == 0) {
        for (int side = 0; side < 2; ++side) {
            const int* in = (side == 0) ? in0 : in1;
            const int  n  = (side == 0) ? blocks0 : blocks1;
            for (int i = 0; i < n; ++i) {
                const int id = in[i];
                if (id == last_block_id) continue;  // re-added at the tail
                bool seen = false;
                #pragma unroll 1
                for (int j = 0; j < s_count; ++j) {
                    if (s_union[j] == id) { seen = true; break; }
                }
                if (!seen && s_count < VORTEX_UNION_MAX) {
                    s_union[s_count++] = id;
                }
            }
        }
    }
    __syncthreads();

    const int u = s_count;
    for (int i = tx; i < u; i += NUM_THREADS) {
        out[i] = s_union[i];
    }
    if (tx == 0) {
        // Place the true last block at the end so the partial
        // ``last_block_len`` covers the right slot for trtllm decode.
        out[u] = last_block_id;
        sparse_seqlens[bx] = u * block_size + last_block_len;
    }
}


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
const int64_t     block_size)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    constexpr int kThreads = 128;
    TORCH_CHECK(
        max_blocks_per_seq <= VORTEX_UNION_MAX,
        "Union: max_blocks_per_seq > VORTEX_UNION_MAX (",
        VORTEX_UNION_MAX, ") not supported — rebuild with a larger "
        "-DVORTEX_UNION_MAX.");

    dim3 nblks(eff_batch_size);
    Union_Kernel<kThreads><<<nblks, kThreads, 0, stream>>>(
        dense_seqlens.data_ptr<int>(),
        sparse_seqlens.data_ptr<int>(),
        dense_block_tables.data_ptr<int>(),
        block_tables_0.data_ptr<int>(),
        seqlens_0.data_ptr<int>(),
        block_tables_1.data_ptr<int>(),
        seqlens_1.data_ptr<int>(),
        sparse_block_tables.data_ptr<int>(),
        static_cast<int>(max_blocks_per_seq),
        static_cast<int>(block_size));

    const auto result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess,
                "Union kernel failed: ", ::cudaGetErrorString(result));
}
