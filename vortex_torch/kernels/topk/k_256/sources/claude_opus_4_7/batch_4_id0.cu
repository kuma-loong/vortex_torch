// Bin-cache + warp-shuffle cumsum radix top-k. Builds on batch_3_id0
// (warp-shuffle cumsum at T=512/16KB — Pareto winner so far) by caching
// the round-0 uint8 bin per element in shared memory: the histogram pass
// fills it as a side effect (already computing the bin for atomicAdd),
// and the filter pass reads it from smem instead of re-loading scores and
// re-running convert_to_uint8. Saves ~length × (1 DRAM read + 1 bf16→fp32→
// uint8 conversion) per CTA — only round-0's filter pass benefits; later
// rounds operate on a small candidate buffer and re-read input anyway.
//
// Static smem footprint (T=512/16KB): ~4KB hist + 1KB s_indices + 4KB s_bins
// = ~9KB → with 16KB dynamic = 25KB/CTA. L40 supports 3 CTAs/SM by both
// threads (1536/512=3) and smem (100KB/25KB≈4), so occupancy unchanged.
//
// Compile-time knobs (substituted by kernels/topk/dispatcher.py):
//   __THREADS_PER_BLOCK__   — CTA width (multiple of 32, >= 256)
//   __VORTEX_MAX_TOPK__     — upper bound on per-block topk_val (static smem)
//   __SMEM_BYTES__          — dynamic smem budget for two-bank candidate buffer
//   __VORTEX_MAX_SEQ_LEN__  — upper bound on per-block sequence length

#include <ATen/core/TensorBase.h>
#include <ATen/core/TensorBody.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>

namespace {

constexpr int kThreadsPerBlock = __THREADS_PER_BLOCK__;
constexpr size_t kSmem = __SMEM_BYTES__;
constexpr int VORTEX_MAX_TOPK = __VORTEX_MAX_TOPK__;
constexpr int VORTEX_MAX_SEQ_LEN = __VORTEX_MAX_SEQ_LEN__;

__device__ __forceinline__ auto convert_to_uint8(float x) -> uint8_t {
    __half h = __float2half_rn(x);
    uint16_t bits = __half_as_ushort(h);
    uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
    return static_cast<uint8_t>(key >> 8);
}

__device__ __forceinline__ auto convert_to_uint32(float x) -> uint32_t {
    uint32_t bits = __float_as_uint(x);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

template <auto* f, size_t max_dynamic_smem>
void setup_kernel_smem_once() {
    [[maybe_unused]]
    static const auto result = [] {
        return ::cudaFuncSetAttribute(f, ::cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_smem);
    }();
    TORCH_CHECK(result == cudaSuccess, "set_up_kernel_once failed:", ::cudaGetErrorString(result));
}

template <typename T>
__device__ __forceinline__ float vortex_to_float(T x);
template <> __device__ __forceinline__ float vortex_to_float<float>(float x) { return x; }
template <> __device__ __forceinline__ float vortex_to_float<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

// Inclusive suffix-sum on 256 ints — warp-cooperative version.
__device__ __forceinline__ void cumsum_suffix_256(int* s_hist) {
    const int tx = threadIdx.x;
    const int lane = tx & 31;
    const int warp = tx >> 5;

    __shared__ int s_warp_above[8];

    int v = 0;
    if (warp < 8) {
        v = s_hist[warp * 32 + lane];
        #pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            int other = __shfl_down_sync(0xffffffffu, v, d);
            if (lane + d < 32) v += other;
        }
        if (lane == 0) {
            s_warp_above[warp] = v;
        }
    }
    __syncthreads();

    if (warp == 0 && lane < 8) {
        int t = s_warp_above[lane];
        #pragma unroll
        for (int d = 1; d < 8; d <<= 1) {
            int other = __shfl_down_sync(0xffu, t, d, 8);
            if (lane + d < 8) t += other;
        }
        int above = __shfl_down_sync(0xffu, t, 1, 8);
        if (lane == 7) above = 0;
        s_warp_above[lane] = above;
    }
    __syncthreads();

    if (warp < 8) {
        int offset = s_warp_above[warp];
        s_hist[warp * 32 + lane] = v + offset;
    }
    __syncthreads();
}

template <typename ScoreT>
__device__ void fast_topk_bincache(
    const ScoreT* __restrict__ input,
    uint8_t*      __restrict__ s_bins,   // [length] in shared memory
    int*          __restrict__ index,
    int           row_start,
    int           length,
    int           target_k)
{
    int topk = target_k;
    constexpr auto BLOCK_SIZE = kThreadsPerBlock;
    constexpr auto RADIX = 256;
    constexpr auto SMEM_INPUT_SIZE = kSmem / (2 * sizeof(int));

    alignas(128) __shared__ int vh_histogram[RADIX + 128];
    alignas(128) __shared__ int vh_counter;
    alignas(128) __shared__ int vh_threshold_bin_id;
    alignas(128) __shared__ int vh_num_input[2];

    extern __shared__ int vh_input_idx[][SMEM_INPUT_SIZE];

    const int tx = threadIdx.x;

    if (tx < RADIX + 1) vh_histogram[tx] = 0;
    __syncthreads();

    // Round 0 — histogram + bin-cache fill (single pass).
    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
        const auto bin = convert_to_uint8(vortex_to_float(input[idx + row_start]));
        s_bins[idx] = bin;
        ::atomicAdd(&vh_histogram[bin], 1);
    }
    __syncthreads();

    cumsum_suffix_256(vh_histogram);
    if (tx < RADIX && vh_histogram[tx] > topk && vh_histogram[tx + 1] <= topk) {
        vh_threshold_bin_id = tx;
        vh_num_input[0] = 0;
        vh_counter = 0;
    }
    __syncthreads();

    const auto threshold_bin = vh_threshold_bin_id;
    topk -= vh_histogram[threshold_bin + 1];

    if (topk == 0) {
        // Filter from cached bins — no input re-read, no re-convert.
        for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
            const int bin = static_cast<int>(s_bins[idx]);
            if (bin > threshold_bin) {
                const auto pos = ::atomicAdd(&vh_counter, 1);
                index[pos] = idx;
            }
        }
        __syncthreads();
        return;
    } else {
        __syncthreads();
        if (tx < RADIX + 1) vh_histogram[tx] = 0;
        __syncthreads();

        // Filter from cached bins; only the threshold-bin candidates re-read
        // the input to compute the byte-1 sub-bin.
        for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
            const int bin = static_cast<int>(s_bins[idx]);
            if (bin > threshold_bin) {
                const auto pos = ::atomicAdd(&vh_counter, 1);
                index[pos] = idx;
            } else if (bin == threshold_bin) {
                const auto pos = ::atomicAdd(&vh_num_input[0], 1);
                if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
                    vh_input_idx[0][pos] = idx;
                    const auto raw_input = vortex_to_float(input[idx + row_start]);
                    const auto b32 = convert_to_uint32(raw_input);
                    const auto sub_bin = (b32 >> 24) & 0xFF;
                    ::atomicAdd(&vh_histogram[sub_bin], 1);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll 4
    for (int round = 0; round < 4; ++round) {
        __shared__ int vh_last_remain;
        const auto r_idx = round % 2;

        const auto _raw_num_input = vh_num_input[r_idx];
        const auto num_input = (_raw_num_input < int(SMEM_INPUT_SIZE))
                                   ? _raw_num_input
                                   : int(SMEM_INPUT_SIZE);

        cumsum_suffix_256(vh_histogram);
        if (tx < RADIX && vh_histogram[tx] > topk && vh_histogram[tx + 1] <= topk) {
            vh_threshold_bin_id = tx;
            vh_num_input[r_idx ^ 1] = 0;
            vh_last_remain = topk - vh_histogram[tx + 1];
        }
        __syncthreads();

        const auto threshold_bin = vh_threshold_bin_id;
        topk -= vh_histogram[threshold_bin + 1];

        if (topk == 0) {
            for (int i = tx; i < num_input; i += BLOCK_SIZE) {
                const auto idx = vh_input_idx[r_idx][i];
                const auto offset = 24 - round * 8;
                const auto bin = (convert_to_uint32(
                    vortex_to_float(input[idx + row_start])) >> offset) & 0xFF;
                if (bin > threshold_bin) {
                    const auto pos = ::atomicAdd(&vh_counter, 1);
                    index[pos] = idx;
                }
            }
            __syncthreads();
            break;
        } else {
            __syncthreads();
            if (tx < RADIX + 1) vh_histogram[tx] = 0;
            __syncthreads();
            for (int i = tx; i < num_input; i += BLOCK_SIZE) {
                const auto idx = vh_input_idx[r_idx][i];
                const auto raw_input = vortex_to_float(input[idx + row_start]);
                const auto offset = 24 - round * 8;
                const auto bin = (convert_to_uint32(raw_input) >> offset) & 0xFF;
                if (bin > threshold_bin) {
                    const auto pos = ::atomicAdd(&vh_counter, 1);
                    index[pos] = idx;
                } else if (bin == threshold_bin) {
                    if (round == 3) {
                        const auto pos = ::atomicAdd(&vh_last_remain, -1);
                        if (pos > 0) {
                            index[target_k - pos] = idx;
                        }
                    } else {
                        const auto pos = ::atomicAdd(&vh_num_input[r_idx ^ 1], 1);
                        if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
                            vh_input_idx[r_idx ^ 1][pos] = idx;
                            const auto b32 = convert_to_uint32(raw_input);
                            const auto sub_bin = (b32 >> (offset - 8)) & 0xFF;
                            ::atomicAdd(&vh_histogram[sub_bin], 1);
                        }
                    }
                }
            }
            __syncthreads();
        }
    }
}

template <typename ScoreT>
__global__ __launch_bounds__(kThreadsPerBlock)
void TopKOutput_Kernel(
    const ScoreT* __restrict__ score,
    const int*    __restrict__ dense_kv_indptr,
    const int*    __restrict__ sparse_kv_indptr,
    const int*    __restrict__ dense_kv_indices,
    int*          __restrict__ sparse_kv_indices,
    const int     page_reserved_bos,
    const int     page_reserved_eos)
{
    const int bx = blockIdx.x;

    const int start = dense_kv_indptr[bx] + page_reserved_bos;
    const int end   = dense_kv_indptr[bx + 1] - page_reserved_eos;
    const int topk_val = sparse_kv_indptr[bx + 1] - sparse_kv_indptr[bx] - page_reserved_bos - page_reserved_eos;
    const int nblk  = end - start;
    if (nblk <= topk_val) return;

    const ScoreT* __restrict__ score_blk = score + start;
    const int*    __restrict__ idx_blk   = dense_kv_indices + start;
    int*          __restrict__ out_blk   = sparse_kv_indices
                                         + sparse_kv_indptr[bx]
                                         + page_reserved_bos;

    __shared__ int s_indices[VORTEX_MAX_TOPK];
    __shared__ uint8_t s_bins[VORTEX_MAX_SEQ_LEN];
    fast_topk_bincache<ScoreT>(score_blk, s_bins, s_indices, 0, nblk, topk_val);
    __syncthreads();

    const int tx = threadIdx.x;
    for (int i = tx; i < topk_val; i += kThreadsPerBlock) {
        out_blk[i] = idx_blk[s_indices[i]];
    }
}

}  // namespace

void topk(
    const at::Tensor& x,
    const at::Tensor& dense_kv_indptr,
    const at::Tensor& sparse_kv_indptr,
    const at::Tensor& dense_kv_indices,
    at::Tensor&       sparse_kv_indices,
    const int64_t     eff_batch_size,
    const int64_t     reserved_bos,
    const int64_t     reserved_eos,
    const int64_t     max_num_pages)
{
    (void)max_num_pages;

    dim3 nblks(eff_batch_size);
    dim3 nthreads(kThreadsPerBlock);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (x.scalar_type() == at::ScalarType::BFloat16) {
        setup_kernel_smem_once<TopKOutput_Kernel<__nv_bfloat16>, kSmem>();
        TopKOutput_Kernel<__nv_bfloat16><<<nblks, nthreads, kSmem, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            dense_kv_indptr.data_ptr<int>(),
            sparse_kv_indptr.data_ptr<int>(),
            dense_kv_indices.data_ptr<int>(),
            sparse_kv_indices.data_ptr<int>(),
            reserved_bos,
            reserved_eos);
    } else if (x.scalar_type() == at::ScalarType::Float) {
        setup_kernel_smem_once<TopKOutput_Kernel<float>, kSmem>();
        TopKOutput_Kernel<float><<<nblks, nthreads, kSmem, stream>>>(
            x.data_ptr<float>(),
            dense_kv_indptr.data_ptr<int>(),
            sparse_kv_indptr.data_ptr<int>(),
            dense_kv_indices.data_ptr<int>(),
            sparse_kv_indices.data_ptr<int>(),
            reserved_bos,
            reserved_eos);
    } else {
        TORCH_CHECK(false, "topk: unsupported dtype ", x.scalar_type());
    }

    const auto result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess,
                "topk kernel failed: ", ::cudaGetErrorString(result));
}
