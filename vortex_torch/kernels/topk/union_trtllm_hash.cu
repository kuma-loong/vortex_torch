// Union of two (block_table, seqlens) pairs — **parallel hash** variant.
//
// Same ABI / semantics as ``union_trtllm.cu`` (the single-threaded
// O(n^2) baseline); the per-row algorithm switches to an open-address
// shared-memory hash table to dedup in parallel:
//
//   1. All ``NUM_THREADS`` threads cooperatively scan the concatenated
//      ``block_tables_0`` + ``block_tables_1`` rows (strided).
//   2. For each id (excluding ``last_block_id``), compute the hash bucket
//      and insert with ``atomicCAS(&s_hash[h], -1, id)``. On a fresh
//      insert, ``atomicAdd(&s_count, 1)`` reserves an output slot and
//      writes the id into ``s_union[pos]`` (deterministic-by-thread but
//      non-deterministic across runs; order in the output is **not**
//      preserved beyond the tail invariant — trtllm attention is
//      order-invariant over keys, so this is safe).
//   3. Threads copy ``s_union[0..u)`` to global ``sparse_block_tables``
//      then append ``last_block_id`` at slot ``u``.
//
// Hash sizing rules (must be power-of-two, kept compile-time):
//   * UNION_HASH_SIZE = 1024 → covers ``bos + 2*k + eos <= ~500`` with
//     load factor ≤ 0.5 (≈ 4KB shared memory).
//   * UNION_MAX_OUT  = 1024 → matches the worst-case dedup-output size.
// Both can be raised via ``-DUNION_HASH_SIZE=... -DUNION_MAX_OUT=...``.

#include <ATen/core/TensorBase.h>
#include <ATen/core/TensorBody.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Exception.h>
#include <cuda.h>

// Hash bucket count (power-of-two) — keep load factor ≤ 0.5 for typical
// ``bos + 2*k + eos`` workloads. 2048 buckets = 8KB shared memory and
// comfortably handles k up to ~512.
#ifndef UNION_HASH_SIZE
#define UNION_HASH_SIZE 2048
#endif
// Worst-case output capacity — must be ≥ max_blocks_per_seq for any
// workload (the dense_block_table can saturate a row when both inputs
// fall back to copying it whole). 4096 int32 slots = 16KB shared mem,
// covers contexts up to ``block_size * 4096`` tokens (≈ 64K with
// block_size=16). Bump and recompile for longer contexts.
#ifndef UNION_MAX_OUT
#define UNION_MAX_OUT 4096
#endif

static_assert(
    (UNION_HASH_SIZE & (UNION_HASH_SIZE - 1)) == 0,
    "UNION_HASH_SIZE must be a power of two"
);

namespace {

__device__ __forceinline__ unsigned int hash_bucket(int id) {
    // Knuth's multiplicative hash; mask to UNION_HASH_SIZE buckets.
    return (static_cast<unsigned int>(id) * 2654435761u) & (UNION_HASH_SIZE - 1);
}

}  // namespace


template <int NUM_THREADS>
__global__ void Union_Kernel_Hash(
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

    __shared__ int s_hash [UNION_HASH_SIZE];
    __shared__ int s_union[UNION_MAX_OUT];
    __shared__ int s_count;

    // Init: hash table = -1 (empty), output count = 0.
    #pragma unroll 4
    for (int i = tx; i < UNION_HASH_SIZE; i += NUM_THREADS) {
        s_hash[i] = -1;
    }
    if (tx == 0) s_count = 0;
    __syncthreads();

    // Cooperative dedup: each thread takes a strided slice of the
    // concatenated input row. ``id == last_block_id`` is filtered out and
    // re-appended at the tail after the union completes.
    for (int i = tx; i < total; i += NUM_THREADS) {
        const int id = (i < blocks0) ? in0[i] : in1[i - blocks0];
        if (id == last_block_id) continue;
        unsigned int h = hash_bucket(id);
        // Linear probing. Hash size is at least 2× worst-case load, so the
        // probe sequence terminates quickly in practice; we cap iterations
        // at UNION_HASH_SIZE for safety.
        #pragma unroll 1
        for (int probe = 0; probe < UNION_HASH_SIZE; ++probe) {
            const int prev = atomicCAS(&s_hash[h], -1, id);
            if (prev == -1) {
                const int pos = atomicAdd(&s_count, 1);
                if (pos < UNION_MAX_OUT) {
                    s_union[pos] = id;
                }
                break;
            }
            if (prev == id) break;     // already present
            h = (h + 1u) & (UNION_HASH_SIZE - 1);
        }
    }
    __syncthreads();

    const int raw_count = s_count;
    const int u = (raw_count < UNION_MAX_OUT) ? raw_count : UNION_MAX_OUT;
    for (int i = tx; i < u; i += NUM_THREADS) {
        out[i] = s_union[i];
    }
    if (tx == 0) {
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
        max_blocks_per_seq <= UNION_MAX_OUT,
        "Union(hash): max_blocks_per_seq > UNION_MAX_OUT (",
        UNION_MAX_OUT, ") not supported — rebuild with -DUNION_MAX_OUT=...");

    dim3 nblks(eff_batch_size);
    Union_Kernel_Hash<kThreads><<<nblks, kThreads, 0, stream>>>(
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
                "Union(hash) kernel failed: ", ::cudaGetErrorString(result));
}
