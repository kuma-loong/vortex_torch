// bf16-aware explicit-skip rounds 2-3 + vec4 + warp-shuffle cumsum.
// Builds on batch_5_id0 (vec4 + warp) by recognising that for bf16 input,
// fp32 bytes 1 and 0 are always zero. The existing 4-round loop spends
// rounds 2 and 3 doing no-op cumsum + zero-histogram refinement, then
// finally hits the round=3 atomic-arrival branch. This kernel:
//   round 0 (fp16-detour initial) → emit strict + collect ties;
//   round 1 (byte 3 = bf16 top byte refinement) → emit strict + collect ties;
//   round 2 (byte 2 = bf16 bottom byte refinement) → emit strict + atomic-arrival.
// Skips the always-no-op rounds 3-4. Saves ~2 cumsum + 2 hist-reset +
// 2 candidate-buffer scans per kernel.
//
// Float input retains the standard 4-round schedule.
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
constexpr int VORTEX_MAX_TOPK = __VORTEX_MAX_TOPK__;

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

// Shared core for both bf16 (3-useful-rounds) and float (4-rounds) — the
// "max_useful_rounds" template parameter caps how many refinement rounds
// run before the final atomic-arrival.
template <typename ScoreT, int MaxRefineRounds>
__device__ void fast_topk_skip_rounds(
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

    extern __shared__ int vh_input_idx[][SMEM_INPUT_SIZE];

    const int tx = threadIdx.x;
    const int quads = length >> 2;
    const int tail_start = quads << 2;
    const int tail_count = length - tail_start;

    if (tx < RADIX + 1) vh_histogram[tx] = 0;
    __syncthreads();

    for (int q = tx; q < quads; q += BLOCK_SIZE) {
        float f0, f1, f2, f3;
        load_quad<ScoreT>(input + row_start + 4 * q, f0, f1, f2, f3);
        ::atomicAdd(&vh_histogram[convert_to_uint8(f0)], 1);
        ::atomicAdd(&vh_histogram[convert_to_uint8(f1)], 1);
        ::atomicAdd(&vh_histogram[convert_to_uint8(f2)], 1);
        ::atomicAdd(&vh_histogram[convert_to_uint8(f3)], 1);
    }
    if (tx < tail_count) {
        ::atomicAdd(&vh_histogram[convert_to_uint8(vortex_to_float(input[row_start + tail_start + tx]))], 1);
    }
    __syncthreads();

    cumsum_suffix_256(vh_histogram);
    if (tx < RADIX && vh_histogram[tx] > topk && vh_histogram[tx + 1] <= topk) {
        vh_threshold_bin_id = tx;
        vh_num_input[0] = 0;
        vh_counter = 0;
    }
    __syncthreads();

    const int initial_threshold_bin = vh_threshold_bin_id;
    topk -= vh_histogram[initial_threshold_bin + 1];

    if (topk == 0) {
        for (int q = tx; q < quads; q += BLOCK_SIZE) {
            float f0, f1, f2, f3;
            load_quad<ScoreT>(input + row_start + 4 * q, f0, f1, f2, f3);
            float fs[4] = {f0, f1, f2, f3};
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                if (convert_to_uint8(fs[k]) > initial_threshold_bin) {
                    const int pos = ::atomicAdd(&vh_counter, 1);
                    index[pos] = 4 * q + k;
                }
            }
        }
        if (tx < tail_count) {
            const int idx = tail_start + tx;
            if (convert_to_uint8(vortex_to_float(input[row_start + idx])) > initial_threshold_bin) {
                const int pos = ::atomicAdd(&vh_counter, 1);
                index[pos] = idx;
            }
        }
        __syncthreads();
        return;
    }

    __syncthreads();
    if (tx < RADIX + 1) vh_histogram[tx] = 0;
    __syncthreads();

    for (int q = tx; q < quads; q += BLOCK_SIZE) {
        float f0, f1, f2, f3;
        load_quad<ScoreT>(input + row_start + 4 * q, f0, f1, f2, f3);
        float fs[4] = {f0, f1, f2, f3};
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int bin = static_cast<int>(convert_to_uint8(fs[k]));
            const int idx = 4 * q + k;
            if (bin > initial_threshold_bin) {
                const int pos = ::atomicAdd(&vh_counter, 1);
                index[pos] = idx;
            } else if (bin == initial_threshold_bin) {
                const int pos = ::atomicAdd(&vh_num_input[0], 1);
                if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
                    vh_input_idx[0][pos] = idx;
                    const auto b32 = convert_to_uint32(fs[k]);
                    const auto sub_bin = (b32 >> 24) & 0xFF;
                    ::atomicAdd(&vh_histogram[sub_bin], 1);
                }
            }
        }
    }
    if (tx < tail_count) {
        const int idx = tail_start + tx;
        const float f = vortex_to_float(input[row_start + idx]);
        const int bin = static_cast<int>(convert_to_uint8(f));
        if (bin > initial_threshold_bin) {
            const int pos = ::atomicAdd(&vh_counter, 1);
            index[pos] = idx;
        } else if (bin == initial_threshold_bin) {
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

    // Useful refinement rounds. After the last useful round, fall through to
    // an atomic-arrival pass to fill any remaining slots.
    int last_round_idx = 0;  // last (round idx) we executed; used to know the offset for atomic-arrival.
#pragma unroll
    for (int round = 0; round < MaxRefineRounds; ++round) {
        const int r_idx = round % 2;
        const int _raw_num_input = vh_num_input[r_idx];
        const int num_input = (_raw_num_input < int(SMEM_INPUT_SIZE))
                                  ? _raw_num_input
                                  : int(SMEM_INPUT_SIZE);

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
                const auto offset = 24 - round * 8;
                const auto bin = (convert_to_uint32(
                    vortex_to_float(input[idx + row_start])) >> offset) & 0xFF;
                if (bin > threshold_bin) {
                    const int pos = ::atomicAdd(&vh_counter, 1);
                    index[pos] = idx;
                }
            }
            __syncthreads();
            return;
        }

        if (is_last_useful_round) {
            // Final round: emit strict + atomic-arrival countdown.
            __shared__ int vh_last_remain;
            if (tx == 0) vh_last_remain = topk;
            __syncthreads();
            for (int i = tx; i < num_input; i += BLOCK_SIZE) {
                const int idx = vh_input_idx[r_idx][i];
                const auto offset = 24 - round * 8;
                const auto bin = (convert_to_uint32(
                    vortex_to_float(input[idx + row_start])) >> offset) & 0xFF;
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
            return;
        }

        // Continue: collect ties + build next-byte sub-histogram.
        __syncthreads();
        if (tx < RADIX + 1) vh_histogram[tx] = 0;
        __syncthreads();
        for (int i = tx; i < num_input; i += BLOCK_SIZE) {
            const int idx = vh_input_idx[r_idx][i];
            const auto raw_input = vortex_to_float(input[idx + row_start]);
            const auto offset = 24 - round * 8;
            const auto bin = (convert_to_uint32(raw_input) >> offset) & 0xFF;
            if (bin > threshold_bin) {
                const int pos = ::atomicAdd(&vh_counter, 1);
                index[pos] = idx;
            } else if (bin == threshold_bin) {
                const int pos = ::atomicAdd(&vh_num_input[r_idx ^ 1], 1);
                if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
                    vh_input_idx[r_idx ^ 1][pos] = idx;
                    const auto b32 = convert_to_uint32(raw_input);
                    const auto sub_bin = (b32 >> (offset - 8)) & 0xFF;
                    ::atomicAdd(&vh_histogram[sub_bin], 1);
                }
            }
        }
        __syncthreads();
        last_round_idx = round;
    }
    (void)last_round_idx;
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

    // For bf16, fp32 bytes 1 and 0 are zero — only 2 useful refinement rounds.
    // For float, 4 useful refinement rounds.
    if constexpr (std::is_same_v<ScoreT, __nv_bfloat16>) {
        fast_topk_skip_rounds<ScoreT, /*MaxRefineRounds=*/2>(
            score_blk, s_indices, 0, nblk, topk_val);
    } else {
        fast_topk_skip_rounds<ScoreT, /*MaxRefineRounds=*/4>(
            score_blk, s_indices, 0, nblk, topk_val);
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
