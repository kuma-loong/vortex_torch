// Direct-emit (no s_indices, no post-filter gather) + vec4 + warp-shuffle
// cumsum radix top-k. Each emit point fuses the page-index remap into the
// write: out_blk[pos] = idx_blk[idx]. Saves the 1KB s_indices static smem
// and the post-filter gather loop. Per-emit global reads are unchanged
// (still 1 read of idx_blk[idx] + 1 write of out_blk[pos]) — the win is
// removing the synchronisation barrier between the filter and the gather.
//
// Compile-time knobs: __THREADS_PER_BLOCK__ __VORTEX_MAX_TOPK__ __SMEM_BYTES__

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
__device__ __forceinline__ void load_quad(const ScoreT* p, float& f0, float& f1, float& f2, float& f3);

template <>
__device__ __forceinline__ void load_quad<__nv_bfloat16>(const __nv_bfloat16* p, float& f0, float& f1, float& f2, float& f3) {
    int2 packed = *reinterpret_cast<const int2*>(p);
    __nv_bfloat162 lo = *reinterpret_cast<__nv_bfloat162*>(&packed.x);
    __nv_bfloat162 hi = *reinterpret_cast<__nv_bfloat162*>(&packed.y);
    f0 = __bfloat162float(__low2bfloat16(lo));
    f1 = __bfloat162float(__high2bfloat16(lo));
    f2 = __bfloat162float(__low2bfloat16(hi));
    f3 = __bfloat162float(__high2bfloat16(hi));
}

template <>
__device__ __forceinline__ void load_quad<float>(const float* p, float& f0, float& f1, float& f2, float& f3) {
    f0 = p[0]; f1 = p[1]; f2 = p[2]; f3 = p[3];
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
    const int target_k = sparse_kv_indptr[bx + 1] - sparse_kv_indptr[bx] - page_reserved_bos - page_reserved_eos;
    const int length  = end - start;
    if (length <= target_k) return;

    const ScoreT* __restrict__ score_blk = score + start;
    const int*    __restrict__ idx_blk   = dense_kv_indices + start;
    int*          __restrict__ out_blk   = sparse_kv_indices + sparse_kv_indptr[bx] + page_reserved_bos;

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
    const int quads = length >> 2;
    const int tail_start = quads << 2;
    const int tail_count = length - tail_start;

    if (tx < RADIX + 1) vh_histogram[tx] = 0;
    __syncthreads();

    // Round 0 histogram.
    for (int q = tx; q < quads; q += BLOCK_SIZE) {
        float f0, f1, f2, f3;
        load_quad<ScoreT>(score_blk + 4 * q, f0, f1, f2, f3);
        ::atomicAdd(&vh_histogram[convert_to_uint8(f0)], 1);
        ::atomicAdd(&vh_histogram[convert_to_uint8(f1)], 1);
        ::atomicAdd(&vh_histogram[convert_to_uint8(f2)], 1);
        ::atomicAdd(&vh_histogram[convert_to_uint8(f3)], 1);
    }
    if (tx < tail_count) {
        ::atomicAdd(&vh_histogram[convert_to_uint8(vortex_to_float(score_blk[tail_start + tx]))], 1);
    }
    __syncthreads();

    cumsum_suffix_256(vh_histogram);
    if (tx < RADIX && vh_histogram[tx] > topk && vh_histogram[tx + 1] <= topk) {
        vh_threshold_bin_id = tx;
        vh_num_input[0] = 0;
        vh_counter = 0;
    }
    __syncthreads();

    const int threshold_bin_0 = vh_threshold_bin_id;
    topk -= vh_histogram[threshold_bin_0 + 1];

    if (topk == 0) {
        // Filter pass — direct emit to out_blk via idx_blk gather.
        for (int q = tx; q < quads; q += BLOCK_SIZE) {
            float f0, f1, f2, f3;
            load_quad<ScoreT>(score_blk + 4 * q, f0, f1, f2, f3);
            float fs[4] = {f0, f1, f2, f3};
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int idx = 4 * q + k;
                if (convert_to_uint8(fs[k]) > threshold_bin_0) {
                    const int pos = ::atomicAdd(&vh_counter, 1);
                    out_blk[pos] = idx_blk[idx];
                }
            }
        }
        if (tx < tail_count) {
            const int idx = tail_start + tx;
            if (convert_to_uint8(vortex_to_float(score_blk[idx])) > threshold_bin_0) {
                const int pos = ::atomicAdd(&vh_counter, 1);
                out_blk[pos] = idx_blk[idx];
            }
        }
        return;  // no further __syncthreads needed (each thread is independent)
    }

    __syncthreads();
    if (tx < RADIX + 1) vh_histogram[tx] = 0;
    __syncthreads();

    // Round 0 filter + sub-histogram build on byte 3 of fp32.
    for (int q = tx; q < quads; q += BLOCK_SIZE) {
        float f0, f1, f2, f3;
        load_quad<ScoreT>(score_blk + 4 * q, f0, f1, f2, f3);
        float fs[4] = {f0, f1, f2, f3};
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float f = fs[k];
            const int idx = 4 * q + k;
            const int bin = static_cast<int>(convert_to_uint8(f));
            if (bin > threshold_bin_0) {
                const int pos = ::atomicAdd(&vh_counter, 1);
                out_blk[pos] = idx_blk[idx];
            } else if (bin == threshold_bin_0) {
                const int pos = ::atomicAdd(&vh_num_input[0], 1);
                if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
                    vh_input_idx[0][pos] = idx;
                    const auto b32 = convert_to_uint32(f);
                    const auto sub_bin = (b32 >> 24) & 0xFF;
                    ::atomicAdd(&vh_histogram[sub_bin], 1);
                }
            }
        }
    }
    if (tx < tail_count) {
        const int idx = tail_start + tx;
        const float f = vortex_to_float(score_blk[idx]);
        const int bin = static_cast<int>(convert_to_uint8(f));
        if (bin > threshold_bin_0) {
            const int pos = ::atomicAdd(&vh_counter, 1);
            out_blk[pos] = idx_blk[idx];
        } else if (bin == threshold_bin_0) {
            const int pos = ::atomicAdd(&vh_num_input[0], 1);
            if (pos < int(SMEM_INPUT_SIZE)) {
                vh_input_idx[0][pos] = idx;
                const auto b32 = convert_to_uint32(f);
                const auto sub_bin = (b32 >> 24) & 0xFF;
                ::atomicAdd(&vh_histogram[sub_bin], 1);
            }
        }
    }
    __syncthreads();

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
                    vortex_to_float(score_blk[idx])) >> offset) & 0xFF;
                if (bin > threshold_bin) {
                    const int pos = ::atomicAdd(&vh_counter, 1);
                    out_blk[pos] = idx_blk[idx];
                }
            }
            break;
        } else {
            __syncthreads();
            if (tx < RADIX + 1) vh_histogram[tx] = 0;
            __syncthreads();
            for (int i = tx; i < num_input; i += BLOCK_SIZE) {
                const auto idx = vh_input_idx[r_idx][i];
                const auto raw_input = vortex_to_float(score_blk[idx]);
                const auto offset = 24 - round * 8;
                const auto bin = (convert_to_uint32(raw_input) >> offset) & 0xFF;
                if (bin > threshold_bin) {
                    const int pos = ::atomicAdd(&vh_counter, 1);
                    out_blk[pos] = idx_blk[idx];
                } else if (bin == threshold_bin) {
                    if (round == 3) {
                        const int pos = ::atomicAdd(&vh_last_remain, -1);
                        if (pos > 0) {
                            out_blk[target_k - pos] = idx_blk[idx];
                        }
                    } else {
                        const int pos = ::atomicAdd(&vh_num_input[r_idx ^ 1], 1);
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
