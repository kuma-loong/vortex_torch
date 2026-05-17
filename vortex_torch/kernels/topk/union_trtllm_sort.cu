// Union of two (block_table, seqlens) pairs — **CUB sort + unique** variant.
//
// Same ABI / semantics as ``union_trtllm.cu``. Per-row algorithm:
//
//   1. Each thread loads ``ITEMS_PER_THREAD`` ids from the concatenated
//      ``block_tables_0`` + ``block_tables_1`` rows (out-of-range slots
//      get ``INT_MAX``; ``last_block_id`` is also masked to ``INT_MAX``
//      and re-appended at the tail).
//   2. ``cub::BlockRadixSort`` sorts the per-thread arrays into ascending
//      order across the block.
//   3. Each item compares to its predecessor in the sorted layout
//      (loaded from a shared scratch buffer); items strictly greater than
//      the predecessor — and ≠ ``INT_MAX`` — are kept.
//   4. ``cub::BlockScan::ExclusiveSum`` over the per-item "keep" flags
//      gives the output position. Threads write their kept ids to
//      ``sparse_block_tables[bx, pos]``.
//   5. ``last_block_id`` is appended at slot ``u`` and ``sparse_seqlens[bx]``
//      is set to ``u * block_size + last_block_len``.
//
// Output is sorted ascending — deterministic, in contrast to the hash
// variant. trtllm attention is order-invariant over keys (modulo the
// "last block holds the partial token count" invariant we enforce by
// re-placing ``last_block_id`` at the tail).
//
// Template knobs:
//   * NUM_THREADS, ITEMS_PER_THREAD chosen at the C entry per k bucket
//     so ``NUM_THREADS * ITEMS_PER_THREAD >= 2 * max_blocks_per_seq``
//     (worst-case input size after concat).

#include <ATen/core/TensorBase.h>
#include <ATen/core/TensorBody.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Exception.h>
#include <cub/cub.cuh>
#include <cuda.h>
#include <limits>


template <int NUM_THREADS, int ITEMS_PER_THREAD>
__global__ void Union_Kernel_Sort(
const int* __restrict__ dense_seqlens,
int*       __restrict__ sparse_seqlens,
const int* __restrict__ dense_block_tables,
const int* __restrict__ block_tables_0,
const int* __restrict__ seqlens_0,
const int* __restrict__ block_tables_1,
const int* __restrict__ seqlens_1,
int*       __restrict__ sparse_block_tables,
const int  row_stride,
const int  block_size)
{
    constexpr int CAPACITY = NUM_THREADS * ITEMS_PER_THREAD;
    constexpr int SENTINEL = 0x7FFFFFFF;

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
    const int total   = blocks0 + blocks1;

    // ---- Load (blocked layout): each thread holds ITEMS_PER_THREAD contiguous slots. ----
    int keys[ITEMS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        const int idx = tx * ITEMS_PER_THREAD + i;
        int v;
        if (idx < blocks0) {
            v = in0[idx];
        } else if (idx < total) {
            v = in1[idx - blocks0];
        } else {
            v = SENTINEL;
        }
        if (v == last_block_id) v = SENTINEL;
        keys[i] = v;
    }

    // ---- Sort the per-thread arrays into ascending order. ----
    using Sort = cub::BlockRadixSort<int, NUM_THREADS, ITEMS_PER_THREAD>;
    __shared__ typename Sort::TempStorage sort_storage;
    Sort(sort_storage).Sort(keys);

    // ---- Materialize the sorted block into shared memory for adjacent-diff. ----
    __shared__ int s_sorted[CAPACITY];
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        s_sorted[tx * ITEMS_PER_THREAD + i] = keys[i];
    }
    __syncthreads();

    // ---- Mark unique non-sentinel runs (keep first occurrence). ----
    int keep[ITEMS_PER_THREAD];
    int local_count = 0;
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        const int idx = tx * ITEMS_PER_THREAD + i;
        const int v   = s_sorted[idx];
        int k = 0;
        if (v != SENTINEL) {
            if (idx == 0) {
                k = 1;
            } else {
                const int prev = s_sorted[idx - 1];
                k = (v != prev) ? 1 : 0;
            }
        }
        keep[i] = k;
        local_count += k;
    }

    // ---- Exclusive prefix sum over per-thread totals → output base for this thread. ----
    using Scan = cub::BlockScan<int, NUM_THREADS>;
    __shared__ typename Scan::TempStorage scan_storage;
    int base = 0;
    int total_unique = 0;
    Scan(scan_storage).ExclusiveSum(local_count, base, total_unique);

    // ---- Scatter kept ids to the output. ----
    int run = base;
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        if (keep[i]) {
            out[run++] = s_sorted[tx * ITEMS_PER_THREAD + i];
        }
    }
    __syncthreads();

    if (tx == 0) {
        out[total_unique] = last_block_id;
        sparse_seqlens[bx] = total_unique * block_size + last_block_len;
    }
}


// Pick (NUM_THREADS, ITEMS_PER_THREAD) such that capacity ≥ 2 *
// max_blocks_per_seq (worst-case concat). We keep NUM_THREADS = 128
// for tile efficiency and bump ITEMS_PER_THREAD as needed.
#define UNION_SORT_LAUNCH(THREADS, ITEMS)                                                     \
    Union_Kernel_Sort<THREADS, ITEMS><<<nblks, THREADS, 0, stream>>>(                         \
        dense_seqlens.data_ptr<int>(),                                                        \
        sparse_seqlens.data_ptr<int>(),                                                       \
        dense_block_tables.data_ptr<int>(),                                                   \
        block_tables_0.data_ptr<int>(),                                                       \
        seqlens_0.data_ptr<int>(),                                                            \
        block_tables_1.data_ptr<int>(),                                                       \
        seqlens_1.data_ptr<int>(),                                                            \
        sparse_block_tables.data_ptr<int>(),                                                  \
        static_cast<int>(max_blocks_per_seq),                                                 \
        static_cast<int>(block_size))


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

    // Worst-case concat = 2 * max_blocks_per_seq; need NUM_THREADS *
    // ITEMS_PER_THREAD ≥ that. Pick the smallest tile that fits.
    const int64_t capacity_required = 2 * max_blocks_per_seq;
    dim3 nblks(eff_batch_size);

    // Bigger tiles hit the 48KB static-shared-memory limit (CUB BlockRadixSort's
    // TempStorage + the s_sorted scratch buffer); cap at 128 * 32 = 4096 ids.
    if      (capacity_required <=  256) { UNION_SORT_LAUNCH(128,  2); }
    else if (capacity_required <=  512) { UNION_SORT_LAUNCH(128,  4); }
    else if (capacity_required <= 1024) { UNION_SORT_LAUNCH(128,  8); }
    else if (capacity_required <= 2048) { UNION_SORT_LAUNCH(128, 16); }
    else if (capacity_required <= 4096) { UNION_SORT_LAUNCH(128, 32); }
    else {
        TORCH_CHECK(
            false,
            "Union(sort): 2 * max_blocks_per_seq > 4096 not supported "
            "(rebuild with a custom launch table + dynamic shared mem)."
        );
    }

    const auto result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess,
                "Union(sort) kernel failed: ", ::cudaGetErrorString(result));
}

#undef UNION_SORT_LAUNCH
