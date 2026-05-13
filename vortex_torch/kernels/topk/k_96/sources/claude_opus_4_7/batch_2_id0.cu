// batch_2_id0 — adaptive_bf16_direct.
//
// **Compositional novelty**: combines the two batch_0/batch_1 wins:
//   1. **bf16_direct** (from batch_0_id1) — drop the fp16-detour
//      initial bin; extract directly from bf16 top byte.  Saves
//      ~3-5 cycles/element on every histogram and filter pass.
//   2. **adaptive_approx_skip** (from batch_1_id0) — when
//      `last_remain ≤ TOLERATE_THRESH`, skip the entire refinement
//      loop and use atomic-arrival countdown.  Saves the second
//      cumsum + candidate-buffer scan on tight distributions.
//
// The bf16-direct path uses **MaxRefineRounds=1** for bf16 (refinement
// at offset=16, i.e. bf16 low byte) — same as batch_0_id1. For float
// it keeps MaxRefineRounds=2 (offsets 16, 8 of fp32).
//
// Compile-time knobs: __THREADS_PER_BLOCK__ __VORTEX_MAX_TOPK__
//                     __SMEM_BYTES__ __TRANSFORM_TYPE__ __TOLERATE_THRESH__

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
constexpr int TOLERATE_THRESH = __TOLERATE_THRESH__;

__device__ __forceinline__ float apply_score_transform(float x) {
    if constexpr (TRANSFORM_TYPE == 0) {
        return x;
    } else if constexpr (TRANSFORM_TYPE == 1) {
        return x * fabsf(x);
    } else {
        return x;
    }
}

// Direct bf16 → 8-bit radix bin (top byte after sign flip).
__device__ __forceinline__ uint8_t bf16_to_uint8(float f) {
    __nv_bfloat16 b = __float2bfloat16_rn(f);
    uint16_t bits = *reinterpret_cast<uint16_t*>(&b);
    uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits)
                                   : static_cast<uint16_t>(bits | 0x8000);
    return static_cast<uint8_t>(key >> 8);
}

__device__ __forceinline__ uint16_t bf16_to_uint16_key(float f) {
    __nv_bfloat16 b = __float2bfloat16_rn(f);
    uint16_t bits = *reinterpret_cast<uint16_t*>(&b);
    return (bits & 0x8000) ? static_cast<uint16_t>(~bits)
                           : static_cast<uint16_t>(bits | 0x8000);
}

__device__ __forceinline__ auto convert_to_uint8(float x) -> uint8_t {
    __half h = __float2half_rn(x);
    uint16_t bits = __half_as_ushort(h);
    uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits)
                                   : static_cast<uint16_t>(bits | 0x8000);
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

// dtype-specific initial bin: bf16-direct for bf16, fp16-detour for float.
template <typename ScoreT>
__device__ __forceinline__ uint8_t initial_bin(float f);
template <> __device__ __forceinline__ uint8_t initial_bin<__nv_bfloat16>(float f) {
    return bf16_to_uint8(f);
}
template <> __device__ __forceinline__ uint8_t initial_bin<float>(float f) {
    return convert_to_uint8(f);
}

template <typename ScoreT>
__device__ __forceinline__ int refinement_bin0(float f) {
    if constexpr (std::is_same_v<ScoreT, __nv_bfloat16>) {
        return bf16_to_uint16_key(f) & 0xFF;
    } else {
        return (convert_to_uint32(f) >> 16) & 0xFF;
    }
}

template <typename ScoreT>
__device__ __forceinline__ int refinement_bin_round(float f, int round) {
    if constexpr (std::is_same_v<ScoreT, __nv_bfloat16>) {
        // bf16 only has 1 meaningful refinement byte (low byte of key).
        return bf16_to_uint16_key(f) & 0xFF;
    } else {
        return (convert_to_uint32(f) >> (16 - round * 8)) & 0xFF;
    }
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

template <typename ScoreT, int MaxRefineRounds>
__device__ void fast_topk_combined(
    const ScoreT* __restrict__ input,
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
    alignas(128) __shared__ int vh_last_remain;

    extern __shared__ int vh_input_idx[][SMEM_INPUT_SIZE];

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

    // --- Histogram pass: bf16-direct top-byte bin extraction ---
    for (int i = tx; i < head_count; i += BLOCK_SIZE) {
        const float f = vortex_to_float(input[row_start + i]);
        ::atomicAdd(&vh_histogram[initial_bin<ScoreT>(f)], 1);
    }
    #pragma unroll
    for (int it = 0; it < MAX_ITERS; ++it) {
        const int q = tx + it * BLOCK_SIZE;
        if (q < quads) {
            float f0, f1, f2, f3;
            load_quad_aligned<ScoreT>(input + row_start + head_count + 4 * q, f0, f1, f2, f3);
            const uint8_t b0 = initial_bin<ScoreT>(f0);
            const uint8_t b1 = initial_bin<ScoreT>(f1);
            const uint8_t b2 = initial_bin<ScoreT>(f2);
            const uint8_t b3 = initial_bin<ScoreT>(f3);
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
        ::atomicAdd(&vh_histogram[initial_bin<ScoreT>(f)], 1);
    }
    __syncthreads();

    cumsum_suffix_256(vh_histogram);
    if (tx < RADIX && vh_histogram[tx] > topk && vh_histogram[tx + 1] <= topk) {
        vh_threshold_bin_id = tx;
        vh_num_input[0] = 0;
        vh_counter = 0;
        vh_last_remain = topk - vh_histogram[tx + 1];
    }
    __syncthreads();

    const int initial_threshold_bin = vh_threshold_bin_id;
    const int last_remain = vh_last_remain;
    topk -= vh_histogram[initial_threshold_bin + 1];

    if (topk == 0) {
        for (int i = tx; i < head_count; i += BLOCK_SIZE) {
            if (int(initial_bin<ScoreT>(vortex_to_float(input[row_start + i]))) > initial_threshold_bin) {
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
                    if (bin > initial_threshold_bin) {
                        const int pos = ::atomicAdd(&vh_counter, 1);
                        index[pos] = head_count + 4 * q + k;
                    }
                }
            }
        }
        for (int i = tx; i < tail_count; i += BLOCK_SIZE) {
            const int idx = tail_start + i;
            if (int(initial_bin<ScoreT>(vortex_to_float(input[row_start + idx]))) > initial_threshold_bin) {
                const int pos = ::atomicAdd(&vh_counter, 1);
                index[pos] = idx;
            }
        }
        __syncthreads();
        return;
    }

    // ----- ADAPTIVE FAST PATH -----
    if (last_remain <= TOLERATE_THRESH) {
        for (int i = tx; i < head_count; i += BLOCK_SIZE) {
            const int bin = int(initial_bin<ScoreT>(vortex_to_float(input[row_start + i])));
            if (bin > initial_threshold_bin) {
                const int pos = ::atomicAdd(&vh_counter, 1);
                index[pos] = i;
            } else if (bin == initial_threshold_bin) {
                const int pos = ::atomicAdd(&vh_last_remain, -1);
                if (pos > 0) {
                    index[target_k - pos] = i;
                }
            }
        }
        #pragma unroll
        for (int it = 0; it < MAX_ITERS; ++it) {
            const int q = tx + it * BLOCK_SIZE;
            if (q < quads) {
                const uint32_t packed = reg_packed_bins[it];
                const int base = head_count + 4 * q;
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const int bin = (packed >> (k * 8)) & 0xFF;
                    if (bin > initial_threshold_bin) {
                        const int pos = ::atomicAdd(&vh_counter, 1);
                        index[pos] = base + k;
                    } else if (bin == initial_threshold_bin) {
                        const int pos = ::atomicAdd(&vh_last_remain, -1);
                        if (pos > 0) {
                            index[target_k - pos] = base + k;
                        }
                    }
                }
            }
        }
        for (int i = tx; i < tail_count; i += BLOCK_SIZE) {
            const int idx = tail_start + i;
            const int bin = int(initial_bin<ScoreT>(vortex_to_float(input[row_start + idx])));
            if (bin > initial_threshold_bin) {
                const int pos = ::atomicAdd(&vh_counter, 1);
                index[pos] = idx;
            } else if (bin == initial_threshold_bin) {
                const int pos = ::atomicAdd(&vh_last_remain, -1);
                if (pos > 0) {
                    index[target_k - pos] = idx;
                }
            }
        }
        __syncthreads();
        return;
    }

    // ----- STANDARD REFINEMENT (bf16: 1 round, float: 2 rounds) -----
    __syncthreads();
    if (tx < RADIX + 1) vh_histogram[tx] = 0;
    __syncthreads();

    auto handle_scalar = [&](float f, int idx) {
        const int bin = int(initial_bin<ScoreT>(f));
        if (bin > initial_threshold_bin) {
            const int pos = ::atomicAdd(&vh_counter, 1);
            index[pos] = idx;
        } else if (bin == initial_threshold_bin) {
            const int pos = ::atomicAdd(&vh_num_input[0], 1);
            if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
                vh_input_idx[0][pos] = idx;
                const int sub_bin = refinement_bin0<ScoreT>(f);
                ::atomicAdd(&vh_histogram[sub_bin], 1);
            }
        }
    };
    for (int i = tx; i < head_count; i += BLOCK_SIZE) {
        handle_scalar(vortex_to_float(input[row_start + i]), i);
    }
    #pragma unroll
    for (int it = 0; it < MAX_ITERS; ++it) {
        const int q = tx + it * BLOCK_SIZE;
        if (q < quads) {
            const uint32_t packed = reg_packed_bins[it];
            const int base = head_count + 4 * q;

            int bins[4];
            bool any_threshold = false;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                bins[k] = (packed >> (k * 8)) & 0xFF;
                if (bins[k] > initial_threshold_bin) {
                    const int pos = ::atomicAdd(&vh_counter, 1);
                    index[pos] = base + k;
                } else if (bins[k] == initial_threshold_bin) {
                    any_threshold = true;
                }
            }
            if (any_threshold) {
                float f0, f1, f2, f3;
                load_quad_aligned<ScoreT>(input + row_start + base, f0, f1, f2, f3);
                float fs[4] = {f0, f1, f2, f3};
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    if (bins[k] == initial_threshold_bin) {
                        const int pos = ::atomicAdd(&vh_num_input[0], 1);
                        if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
                            vh_input_idx[0][pos] = base + k;
                            const int sub_bin = refinement_bin0<ScoreT>(fs[k]);
                            ::atomicAdd(&vh_histogram[sub_bin], 1);
                        }
                    }
                }
            }
        }
    }
    for (int i = tx; i < tail_count; i += BLOCK_SIZE) {
        const int idx = tail_start + i;
        handle_scalar(vortex_to_float(input[row_start + idx]), idx);
    }
    __syncthreads();

#pragma unroll
    for (int round = 0; round < MaxRefineRounds; ++round) {
        const int r_idx = round % 2;
        const int _raw = vh_num_input[r_idx];
        const int num_input = (_raw < int(SMEM_INPUT_SIZE)) ? _raw : int(SMEM_INPUT_SIZE);

        cumsum_suffix_256(vh_histogram);
        if (tx < RADIX && vh_histogram[tx] > topk && vh_histogram[tx + 1] <= topk) {
            vh_threshold_bin_id = tx;
            vh_num_input[r_idx ^ 1] = 0;
        }
        __syncthreads();

        const int threshold_bin = vh_threshold_bin_id;
        topk -= vh_histogram[threshold_bin + 1];

        const bool is_last_useful_round = (round == MaxRefineRounds - 1);

        if (topk == 0) {
            for (int i = tx; i < num_input; i += BLOCK_SIZE) {
                const int idx = vh_input_idx[r_idx][i];
                const float f = vortex_to_float(input[idx + row_start]);
                const int bin = refinement_bin_round<ScoreT>(f, round);
                if (bin > threshold_bin) {
                    const int pos = ::atomicAdd(&vh_counter, 1);
                    index[pos] = idx;
                }
            }
            __syncthreads();
            return;
        }

        if (is_last_useful_round) {
            __shared__ int vh_last_remain2;
            if (tx == 0) vh_last_remain2 = topk;
            __syncthreads();
            for (int i = tx; i < num_input; i += BLOCK_SIZE) {
                const int idx = vh_input_idx[r_idx][i];
                const float f = vortex_to_float(input[idx + row_start]);
                const int bin = refinement_bin_round<ScoreT>(f, round);
                if (bin > threshold_bin) {
                    const int pos = ::atomicAdd(&vh_counter, 1);
                    index[pos] = idx;
                } else if (bin == threshold_bin) {
                    const int pos = ::atomicAdd(&vh_last_remain2, -1);
                    if (pos > 0) {
                        index[target_k - pos] = idx;
                    }
                }
            }
            __syncthreads();
            return;
        }

        __syncthreads();
        if (tx < RADIX + 1) vh_histogram[tx] = 0;
        __syncthreads();
        for (int i = tx; i < num_input; i += BLOCK_SIZE) {
            const int idx = vh_input_idx[r_idx][i];
            const float f = vortex_to_float(input[idx + row_start]);
            const int bin = refinement_bin_round<ScoreT>(f, round);
            if (bin > threshold_bin) {
                const int pos = ::atomicAdd(&vh_counter, 1);
                index[pos] = idx;
            } else if (bin == threshold_bin) {
                const int pos = ::atomicAdd(&vh_num_input[r_idx ^ 1], 1);
                if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
                    vh_input_idx[r_idx ^ 1][pos] = idx;
                    // bf16 has no further refinement; just dump into bin 0
                    if constexpr (std::is_same_v<ScoreT, __nv_bfloat16>) {
                        ::atomicAdd(&vh_histogram[0], 1);
                    } else {
                        const int sub_bin = refinement_bin_round<ScoreT>(f, round + 1);
                        ::atomicAdd(&vh_histogram[sub_bin], 1);
                    }
                }
            }
        }
        __syncthreads();
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
    const int topk_val = sparse_kv_indptr[bx + 1] - sparse_kv_indptr[bx]
                       - page_reserved_bos - page_reserved_eos;
    const int nblk  = end - start;
    if (nblk <= topk_val) return;

    const int* __restrict__ idx_blk = dense_kv_indices + start;
    int*       __restrict__ out_blk = sparse_kv_indices + sparse_kv_indptr[bx]
                                    + page_reserved_bos;

    __shared__ int s_indices[VORTEX_MAX_TOPK];

    if constexpr (std::is_same_v<ScoreT, __nv_bfloat16>) {
        fast_topk_combined<ScoreT, /*MaxRefineRounds=*/1>(score, s_indices, start, nblk, topk_val);
    } else {
        fast_topk_combined<ScoreT, /*MaxRefineRounds=*/2>(score, s_indices, start, nblk, topk_val);
    }
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
