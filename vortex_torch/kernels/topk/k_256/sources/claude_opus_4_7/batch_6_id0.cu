// bf16-2round + vec4 radix top-k. bf16 only has 16 bits of precision so
// the existing 5-round (initial + 4 refinement) schedule wastes 3
// rounds on always-zero byte slots. This kernel does exactly:
//   round 0   — histogram on bf16's top 8 bits (sign + 7 exp), direct.
//   round 1   — refine on bf16's bottom 8 bits (1 exp + 7 mantissa) via
//               fp32 byte-2 of the threshold-bin candidate set.
//   final     — atomic-arrival countdown for the remaining-tie slots.
//
// Float input falls back to the standard 4-round schedule via a runtime
// branch (not exercised by this benchmark — bf16 only).
//
// Compile-time knobs:
//   __THREADS_PER_BLOCK__  __VORTEX_MAX_TOPK__  __SMEM_BYTES__

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

__device__ __forceinline__ uint8_t bf16_bits_to_bin(uint16_t bits) {
    uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits)
                                   : static_cast<uint16_t>(bits | 0x8000);
    return static_cast<uint8_t>(key >> 8);
}

// Byte 2 of the order-preserving fp32 representation of a bf16 value =
// the bottom byte of the bf16 order-key.
__device__ __forceinline__ uint8_t bf16_bits_to_byte2(uint16_t bits) {
    uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits)
                                   : static_cast<uint16_t>(bits | 0x8000);
    return static_cast<uint8_t>(key & 0xFF);
}

template <auto* f, size_t max_dynamic_smem>
void setup_kernel_smem_once() {
    [[maybe_unused]]
    static const auto result = [] {
        return ::cudaFuncSetAttribute(f, ::cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_smem);
    }();
    TORCH_CHECK(result == cudaSuccess, "set_up_kernel_once failed:", ::cudaGetErrorString(result));
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

// 4 raw bf16 bits unpacked from int2 (8 bytes = 4 bf16).
__device__ __forceinline__ void load_quad_bf16(
    const __nv_bfloat16* p, uint16_t& b0, uint16_t& b1, uint16_t& b2, uint16_t& b3)
{
    int2 packed = *reinterpret_cast<const int2*>(p);
    uint32_t lo = static_cast<uint32_t>(packed.x);
    uint32_t hi = static_cast<uint32_t>(packed.y);
    b0 = static_cast<uint16_t>(lo & 0xFFFFu);
    b1 = static_cast<uint16_t>(lo >> 16);
    b2 = static_cast<uint16_t>(hi & 0xFFFFu);
    b3 = static_cast<uint16_t>(hi >> 16);
}

__device__ void fast_topk_bf16_2round(
    const __nv_bfloat16* __restrict__ input,
    int*           __restrict__ index,
    int            row_start,
    int            length,
    int            target_k)
{
    int topk = target_k;
    constexpr auto BLOCK_SIZE = kThreadsPerBlock;
    constexpr auto RADIX = 256;
    constexpr auto SMEM_INPUT_SIZE = kSmem / (2 * sizeof(int));

    alignas(128) __shared__ int vh_histogram[RADIX + 128];
    alignas(128) __shared__ int vh_counter;
    alignas(128) __shared__ int vh_threshold_bin_id;
    alignas(128) __shared__ int vh_num_input;
    alignas(128) __shared__ int vh_last_remain;

    extern __shared__ int vh_input_idx[][SMEM_INPUT_SIZE];

    const int tx = threadIdx.x;
    const int quads = length >> 2;
    const int tail_start = quads << 2;
    const int tail_count = length - tail_start;

    if (tx < RADIX + 1) vh_histogram[tx] = 0;
    __syncthreads();

    // Round 0 histogram — direct bf16 bin.
    for (int q = tx; q < quads; q += BLOCK_SIZE) {
        uint16_t b0, b1, b2, b3;
        load_quad_bf16(input + row_start + 4 * q, b0, b1, b2, b3);
        ::atomicAdd(&vh_histogram[bf16_bits_to_bin(b0)], 1);
        ::atomicAdd(&vh_histogram[bf16_bits_to_bin(b1)], 1);
        ::atomicAdd(&vh_histogram[bf16_bits_to_bin(b2)], 1);
        ::atomicAdd(&vh_histogram[bf16_bits_to_bin(b3)], 1);
    }
    if (tx < tail_count) {
        const __nv_bfloat16 v = input[row_start + tail_start + tx];
        uint16_t bits = *reinterpret_cast<const uint16_t*>(&v);
        ::atomicAdd(&vh_histogram[bf16_bits_to_bin(bits)], 1);
    }
    __syncthreads();

    cumsum_suffix_256(vh_histogram);
    if (tx < RADIX && vh_histogram[tx] > topk && vh_histogram[tx + 1] <= topk) {
        vh_threshold_bin_id = tx;
        vh_num_input = 0;
        vh_counter = 0;
    }
    __syncthreads();

    const int threshold_bin_0 = vh_threshold_bin_id;
    topk -= vh_histogram[threshold_bin_0 + 1];

    if (topk == 0) {
        // Strict winners only — no refinement needed.
        for (int q = tx; q < quads; q += BLOCK_SIZE) {
            uint16_t b0, b1, b2, b3;
            load_quad_bf16(input + row_start + 4 * q, b0, b1, b2, b3);
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint16_t bits = (k == 0) ? b0 : (k == 1) ? b1 : (k == 2) ? b2 : b3;
                if (bf16_bits_to_bin(bits) > threshold_bin_0) {
                    const int pos = ::atomicAdd(&vh_counter, 1);
                    index[pos] = 4 * q + k;
                }
            }
        }
        if (tx < tail_count) {
            const int idx = tail_start + tx;
            const __nv_bfloat16 v = input[row_start + idx];
            uint16_t bits = *reinterpret_cast<const uint16_t*>(&v);
            if (bf16_bits_to_bin(bits) > threshold_bin_0) {
                const int pos = ::atomicAdd(&vh_counter, 1);
                index[pos] = idx;
            }
        }
        __syncthreads();
        return;
    }

    // Reset histogram for round-1 (byte-2 = bf16 bottom byte) sub-histogram.
    __syncthreads();
    if (tx < RADIX + 1) vh_histogram[tx] = 0;
    __syncthreads();

    // Round 0 filter — emit strict winners; collect ties; build byte-2 sub-histogram.
    for (int q = tx; q < quads; q += BLOCK_SIZE) {
        uint16_t b0, b1, b2, b3;
        load_quad_bf16(input + row_start + 4 * q, b0, b1, b2, b3);
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint16_t bits = (k == 0) ? b0 : (k == 1) ? b1 : (k == 2) ? b2 : b3;
            const int bin = static_cast<int>(bf16_bits_to_bin(bits));
            const int idx = 4 * q + k;
            if (bin > threshold_bin_0) {
                const int pos = ::atomicAdd(&vh_counter, 1);
                index[pos] = idx;
            } else if (bin == threshold_bin_0) {
                const int pos = ::atomicAdd(&vh_num_input, 1);
                if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
                    vh_input_idx[0][pos] = idx;
                    const int sub_bin = static_cast<int>(bf16_bits_to_byte2(bits));
                    ::atomicAdd(&vh_histogram[sub_bin], 1);
                }
            }
        }
    }
    if (tx < tail_count) {
        const int idx = tail_start + tx;
        const __nv_bfloat16 v = input[row_start + idx];
        uint16_t bits = *reinterpret_cast<const uint16_t*>(&v);
        const int bin = static_cast<int>(bf16_bits_to_bin(bits));
        if (bin > threshold_bin_0) {
            const int pos = ::atomicAdd(&vh_counter, 1);
            index[pos] = idx;
        } else if (bin == threshold_bin_0) {
            const int pos = ::atomicAdd(&vh_num_input, 1);
            if (pos < int(SMEM_INPUT_SIZE)) {
                vh_input_idx[0][pos] = idx;
                const int sub_bin = static_cast<int>(bf16_bits_to_byte2(bits));
                ::atomicAdd(&vh_histogram[sub_bin], 1);
            }
        }
    }
    __syncthreads();

    // Round 1 — byte-2 refinement.
    cumsum_suffix_256(vh_histogram);
    if (tx < RADIX && vh_histogram[tx] > topk && vh_histogram[tx + 1] <= topk) {
        vh_threshold_bin_id = tx;
        vh_last_remain = topk - vh_histogram[tx + 1];
    }
    __syncthreads();

    const int threshold_bin_1 = vh_threshold_bin_id;
    const int num_input_clipped =
        (vh_num_input < int(SMEM_INPUT_SIZE)) ? vh_num_input : int(SMEM_INPUT_SIZE);

    for (int i = tx; i < num_input_clipped; i += BLOCK_SIZE) {
        const int idx = vh_input_idx[0][i];
        const __nv_bfloat16 v = input[idx + row_start];
        uint16_t bits = *reinterpret_cast<const uint16_t*>(&v);
        const int sub_bin = static_cast<int>(bf16_bits_to_byte2(bits));
        if (sub_bin > threshold_bin_1) {
            const int pos = ::atomicAdd(&vh_counter, 1);
            index[pos] = idx;
        } else if (sub_bin == threshold_bin_1) {
            const int pos = ::atomicAdd(&vh_last_remain, -1);
            if (pos > 0) {
                index[target_k - pos] = idx;
            }
        }
    }
    __syncthreads();
}

// Generic-template fallback for float (uses the standard 4-round schedule via the existing radix kernel pattern).
__device__ __forceinline__ uint8_t convert_to_uint8(float x) {
    __half h = __float2half_rn(x);
    uint16_t bits = __half_as_ushort(h);
    uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
    return static_cast<uint8_t>(key >> 8);
}

__device__ __forceinline__ uint32_t convert_to_uint32(float x) {
    uint32_t bits = __float_as_uint(x);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ void fast_topk_float_4round(
    const float* __restrict__ input,
    int*         __restrict__ index,
    int          row_start,
    int          length,
    int          target_k)
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

    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
        const auto bin = convert_to_uint8(input[idx + row_start]);
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
        for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
            const auto bin = static_cast<int>(convert_to_uint8(input[idx + row_start]));
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

        for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
            const auto raw_input = input[idx + row_start];
            const auto bin = static_cast<int>(convert_to_uint8(raw_input));
            if (bin > threshold_bin) {
                const auto pos = ::atomicAdd(&vh_counter, 1);
                index[pos] = idx;
            } else if (bin == threshold_bin) {
                const auto pos = ::atomicAdd(&vh_num_input[0], 1);
                if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
                    vh_input_idx[0][pos] = idx;
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
                const auto bin = (convert_to_uint32(input[idx + row_start]) >> offset) & 0xFF;
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
                const auto raw_input = input[idx + row_start];
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
__device__ void dispatch_inner(
    const ScoreT* input, int* idx, int row_start, int length, int target_k);

template <>
__device__ void dispatch_inner<__nv_bfloat16>(
    const __nv_bfloat16* input, int* idx, int row_start, int length, int target_k) {
    fast_topk_bf16_2round(input, idx, row_start, length, target_k);
}

template <>
__device__ void dispatch_inner<float>(
    const float* input, int* idx, int row_start, int length, int target_k) {
    fast_topk_float_4round(input, idx, row_start, length, target_k);
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
    int*          __restrict__ out_blk   = sparse_kv_indices + sparse_kv_indptr[bx] + page_reserved_bos;

    __shared__ int s_indices[VORTEX_MAX_TOPK];
    dispatch_inner<ScoreT>(score_blk, s_indices, 0, nblk, topk_val);
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
