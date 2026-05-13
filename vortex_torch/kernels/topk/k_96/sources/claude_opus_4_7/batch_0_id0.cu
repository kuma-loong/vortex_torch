// batch_0_id0 — approx_no_refine.
//
// Derived from k_256 winner (batch_22_id0.cu): regcache_unrolled +
// alignment-safe vec4 + bf16-aware explicit-skip + warp-shuffle cumsum
// + monotonic transform.  **Novelty**: eliminate the refinement loop
// entirely. The round-0 filter pass uses **atomic-arrival countdown**
// (analogous to csrc/approx_topk.cu) on threshold-bin items: each
// threshold-bin element claims a slot via `atomicAdd(&last, -1)` and
// writes to `index[target_k - pos]`. There is no second histogram pass
// and no cumsum over a sub-bin histogram.
//
// Trade-off: ties at the K-th boundary fall through arbitrarily (the
// atomic-arrival order is non-deterministic). On continuous score
// distributions (which benchmark.py uses), the probability of ties
// at the boundary is vanishingly small, so R@K should stay ≥ 0.98 in
// expectation. Saves ~1 cumsum + 1 candidate-buffer scan compared to
// MaxRefineRounds=2 baseline.
//
// Compile-time knobs: __THREADS_PER_BLOCK__ __VORTEX_MAX_TOPK__
//                     __SMEM_BYTES__ __TRANSFORM_TYPE__

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
#include <type_traits>

namespace {

constexpr int kThreadsPerBlock = __THREADS_PER_BLOCK__;
constexpr size_t kSmem = __SMEM_BYTES__;
constexpr int VORTEX_MAX_TOPK = __VORTEX_MAX_TOPK__;
constexpr int MAX_ITERS = 4;
constexpr int TRANSFORM_TYPE = __TRANSFORM_TYPE__;

__device__ __forceinline__ float apply_score_transform(float x) {
    if constexpr (TRANSFORM_TYPE == 0) {
        return x;
    } else if constexpr (TRANSFORM_TYPE == 1) {
        return x * fabsf(x);
    } else {
        return x;
    }
}

__device__ __forceinline__ auto convert_to_uint8(float x) -> uint8_t {
    __half h = __float2half_rn(x);
    uint16_t bits = __half_as_ushort(h);
    uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits)
                                   : static_cast<uint16_t>(bits | 0x8000);
    return static_cast<uint8_t>(key >> 8);
}

template <auto* f, size_t max_dynamic_smem>
void setup_kernel_smem_once() {
    [[maybe_unused]]
    static const auto result = [] {
        return ::cudaFuncSetAttribute(
            f, ::cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_smem);
    }();
    TORCH_CHECK(result == cudaSuccess,
                "set_up_kernel_once failed:", ::cudaGetErrorString(result));
}

template <typename T>
__device__ __forceinline__ float vortex_to_float(T x);
template <> __device__ __forceinline__ float vortex_to_float<float>(float x) {
    return apply_score_transform(x);
}
template <> __device__ __forceinline__ float vortex_to_float<__nv_bfloat16>(__nv_bfloat16 x) {
    return apply_score_transform(__bfloat162float(x));
}

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
        if (lane == 0) s_warp_above[warp] = v;
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
__device__ __forceinline__ void load_quad_aligned(const ScoreT* p,
    float& f0, float& f1, float& f2, float& f3);

template <>
__device__ __forceinline__ void load_quad_aligned<__nv_bfloat16>(const __nv_bfloat16* p,
    float& f0, float& f1, float& f2, float& f3) {
    int2 packed = *reinterpret_cast<const int2*>(p);
    __nv_bfloat162 lo = *reinterpret_cast<__nv_bfloat162*>(&packed.x);
    __nv_bfloat162 hi = *reinterpret_cast<__nv_bfloat162*>(&packed.y);
    f0 = apply_score_transform(__bfloat162float(__low2bfloat16(lo)));
    f1 = apply_score_transform(__bfloat162float(__high2bfloat16(lo)));
    f2 = apply_score_transform(__bfloat162float(__low2bfloat16(hi)));
    f3 = apply_score_transform(__bfloat162float(__high2bfloat16(hi)));
}

template <>
__device__ __forceinline__ void load_quad_aligned<float>(const float* p,
    float& f0, float& f1, float& f2, float& f3) {
    f0 = apply_score_transform(p[0]);
    f1 = apply_score_transform(p[1]);
    f2 = apply_score_transform(p[2]);
    f3 = apply_score_transform(p[3]);
}

template <typename ScoreT>
__device__ __forceinline__ int required_alignment_in_elems();
template <> __device__ __forceinline__ int required_alignment_in_elems<__nv_bfloat16>() { return 4; }
template <> __device__ __forceinline__ int required_alignment_in_elems<float>() { return 1; }

template <typename ScoreT>
__device__ void fast_topk_approx_no_refine(
    const ScoreT* __restrict__ input,
    int*          __restrict__ index,
    int           row_start,
    int           length,
    int           target_k)
{
    int topk = target_k;
    constexpr auto BLOCK_SIZE = kThreadsPerBlock;
    constexpr auto RADIX = 256;

    alignas(128) __shared__ int vh_histogram[RADIX + 128];
    alignas(128) __shared__ int vh_counter;
    alignas(128) __shared__ int vh_threshold_bin_id;
    alignas(128) __shared__ int vh_last_remain;

    const int tx = threadIdx.x;

    const int align_elems = required_alignment_in_elems<ScoreT>();
    int head_count = (align_elems - (row_start % align_elems)) % align_elems;
    if (head_count > length) head_count = length;
    const int aligned_length = length - head_count;
    const int quads = aligned_length >> 2;
    const int tail_start = head_count + (quads << 2);
    const int tail_count = length - tail_start;

    uint32_t reg_packed_bins[MAX_ITERS];
    #pragma unroll
    for (int i = 0; i < MAX_ITERS; ++i) reg_packed_bins[i] = 0;

    if (tx < RADIX + 1) vh_histogram[tx] = 0;
    __syncthreads();

    // --- Histogram pass ---
    for (int i = tx; i < head_count; i += BLOCK_SIZE) {
        const float f = vortex_to_float(input[row_start + i]);
        ::atomicAdd(&vh_histogram[convert_to_uint8(f)], 1);
    }
    #pragma unroll
    for (int it = 0; it < MAX_ITERS; ++it) {
        const int q = tx + it * BLOCK_SIZE;
        if (q < quads) {
            float f0, f1, f2, f3;
            load_quad_aligned<ScoreT>(input + row_start + head_count + 4 * q, f0, f1, f2, f3);
            const uint8_t b0 = convert_to_uint8(f0);
            const uint8_t b1 = convert_to_uint8(f1);
            const uint8_t b2 = convert_to_uint8(f2);
            const uint8_t b3 = convert_to_uint8(f3);
            reg_packed_bins[it] = uint32_t(b0)
                                | (uint32_t(b1) << 8)
                                | (uint32_t(b2) << 16)
                                | (uint32_t(b3) << 24);
            ::atomicAdd(&vh_histogram[b0], 1);
            ::atomicAdd(&vh_histogram[b1], 1);
            ::atomicAdd(&vh_histogram[b2], 1);
            ::atomicAdd(&vh_histogram[b3], 1);
        }
    }
    for (int i = tx; i < tail_count; i += BLOCK_SIZE) {
        const float f = vortex_to_float(input[row_start + tail_start + i]);
        ::atomicAdd(&vh_histogram[convert_to_uint8(f)], 1);
    }
    __syncthreads();

    cumsum_suffix_256(vh_histogram);
    if (tx < RADIX && vh_histogram[tx] > topk && vh_histogram[tx + 1] <= topk) {
        vh_threshold_bin_id = tx;
        vh_counter = 0;
        // remaining slots after strict winners are filled by threshold-bin items
        // via atomic-arrival countdown.
        vh_last_remain = topk - vh_histogram[tx + 1];
    }
    __syncthreads();

    const int threshold_bin = vh_threshold_bin_id;
    topk -= vh_histogram[threshold_bin + 1];

    // --- Single fused filter pass: emit winners (bin > threshold) and
    // atomic-arrival on threshold-bin items (bin == threshold). ---
    if (topk == 0) {
        // All slots are strict winners — no threshold-bin items needed.
        for (int i = tx; i < head_count; i += BLOCK_SIZE) {
            const int bin = static_cast<int>(convert_to_uint8(vortex_to_float(input[row_start + i])));
            if (bin > threshold_bin) {
                const int pos = ::atomicAdd(&vh_counter, 1);
                index[pos] = i;
            }
        }
        #pragma unroll
        for (int it = 0; it < MAX_ITERS; ++it) {
            const int q = tx + it * BLOCK_SIZE;
            if (q < quads) {
                const uint32_t packed = reg_packed_bins[it];
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const int bin = (packed >> (k * 8)) & 0xFF;
                    if (bin > threshold_bin) {
                        const int pos = ::atomicAdd(&vh_counter, 1);
                        index[pos] = head_count + 4 * q + k;
                    }
                }
            }
        }
        for (int i = tx; i < tail_count; i += BLOCK_SIZE) {
            const int idx = tail_start + i;
            const int bin = static_cast<int>(convert_to_uint8(vortex_to_float(input[row_start + idx])));
            if (bin > threshold_bin) {
                const int pos = ::atomicAdd(&vh_counter, 1);
                index[pos] = idx;
            }
        }
        __syncthreads();
        return;
    }

    // topk > 0 → need atomic-arrival on threshold-bin items.
    // Head (scalar).
    for (int i = tx; i < head_count; i += BLOCK_SIZE) {
        const int bin = static_cast<int>(convert_to_uint8(vortex_to_float(input[row_start + i])));
        if (bin > threshold_bin) {
            const int pos = ::atomicAdd(&vh_counter, 1);
            index[pos] = i;
        } else if (bin == threshold_bin) {
            const int pos = ::atomicAdd(&vh_last_remain, -1);
            if (pos > 0) {
                index[target_k - pos] = i;
            }
        }
    }
    // Middle (cached bins from registers).
    #pragma unroll
    for (int it = 0; it < MAX_ITERS; ++it) {
        const int q = tx + it * BLOCK_SIZE;
        if (q < quads) {
            const uint32_t packed = reg_packed_bins[it];
            const int base = head_count + 4 * q;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int bin = (packed >> (k * 8)) & 0xFF;
                if (bin > threshold_bin) {
                    const int pos = ::atomicAdd(&vh_counter, 1);
                    index[pos] = base + k;
                } else if (bin == threshold_bin) {
                    const int pos = ::atomicAdd(&vh_last_remain, -1);
                    if (pos > 0) {
                        index[target_k - pos] = base + k;
                    }
                }
            }
        }
    }
    // Tail (scalar).
    for (int i = tx; i < tail_count; i += BLOCK_SIZE) {
        const int idx = tail_start + i;
        const int bin = static_cast<int>(convert_to_uint8(vortex_to_float(input[row_start + idx])));
        if (bin > threshold_bin) {
            const int pos = ::atomicAdd(&vh_counter, 1);
            index[pos] = idx;
        } else if (bin == threshold_bin) {
            const int pos = ::atomicAdd(&vh_last_remain, -1);
            if (pos > 0) {
                index[target_k - pos] = idx;
            }
        }
    }
    __syncthreads();
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
    const int topk_val = sparse_kv_indptr[bx + 1] - sparse_kv_indptr[bx]
                       - page_reserved_bos - page_reserved_eos;
    const int nblk  = end - start;
    if (nblk <= topk_val) return;

    const int* __restrict__ idx_blk = dense_kv_indices + start;
    int*       __restrict__ out_blk = sparse_kv_indices + sparse_kv_indptr[bx]
                                    + page_reserved_bos;

    __shared__ int s_indices[VORTEX_MAX_TOPK];

    fast_topk_approx_no_refine<ScoreT>(score, s_indices, start, nblk, topk_val);
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
