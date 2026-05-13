// Alignment-safe vec4 + bf16-aware explicit-skip + warp-shuffle cumsum.
// Same algorithm as batch_7_id1 (current champion) but with **no hard
// constraint on seq_len**: processes a per-segment scalar HEAD until the
// input pointer reaches 8-byte alignment (for int2 vec4 loads), then a
// vec4 MIDDLE, then a scalar TAIL. Works for any `row_start` and any
// `length ≥ 0`.
//
// For benchmark seq_lens (all multiples of 8) the head/tail are empty
// and behaviour matches batch_7_id1. For arbitrary seq_len the kernel
// still produces correct results, with a small overhead proportional to
// (1 + length % 4) elements processed scalarly.
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
#include <type_traits>

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

// Vec4 load from an 8-byte-aligned pointer. Specialised for bf16 (int2 path)
// and float (4 scalar loads — already 4-byte aligned).
template <typename ScoreT>
__device__ __forceinline__ void load_quad_aligned(const ScoreT* p,
    float& f0, float& f1, float& f2, float& f3);

template <>
__device__ __forceinline__ void load_quad_aligned<__nv_bfloat16>(const __nv_bfloat16* p,
    float& f0, float& f1, float& f2, float& f3) {
    int2 packed = *reinterpret_cast<const int2*>(p);
    __nv_bfloat162 lo = *reinterpret_cast<__nv_bfloat162*>(&packed.x);
    __nv_bfloat162 hi = *reinterpret_cast<__nv_bfloat162*>(&packed.y);
    f0 = __bfloat162float(__low2bfloat16(lo));
    f1 = __bfloat162float(__high2bfloat16(lo));
    f2 = __bfloat162float(__low2bfloat16(hi));
    f3 = __bfloat162float(__high2bfloat16(hi));
}

template <>
__device__ __forceinline__ void load_quad_aligned<float>(const float* p,
    float& f0, float& f1, float& f2, float& f3) {
    f0 = p[0]; f1 = p[1]; f2 = p[2]; f3 = p[3];
}

// Required bf16-element alignment for one int2 (8-byte) load: 4 elements.
// For float, 1 element (already 4-byte aligned via element index).
template <typename ScoreT>
__device__ __forceinline__ int required_alignment_in_elems();

template <>
__device__ __forceinline__ int required_alignment_in_elems<__nv_bfloat16>() { return 4; }

template <>
__device__ __forceinline__ int required_alignment_in_elems<float>() { return 1; }

// Helper to scalarly process a single element for round-0 histogram.
template <typename ScoreT>
__device__ __forceinline__ void hist_scalar(
    const ScoreT* base, int idx, int* hist)
{
    const float f = vortex_to_float(base[idx]);
    ::atomicAdd(&hist[convert_to_uint8(f)], 1);
}

template <typename ScoreT, int MaxRefineRounds>
__device__ void fast_topk_aligned(
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

    // === Compute alignment partition over the [0, length) index range. ===
    const int align_elems = required_alignment_in_elems<ScoreT>();
    // head_count = elements to process scalarly until (row_start + head_count) % align_elems == 0
    int head_count = (align_elems - (row_start % align_elems)) % align_elems;
    if (head_count > length) head_count = length;
    const int aligned_length = length - head_count;          // elements after the head
    const int quads = aligned_length >> 2;                   // vec4 quads
    const int tail_start = head_count + (quads << 2);        // absolute index of tail start
    const int tail_count = length - tail_start;              // elements after the vec4 middle

    if (tx < RADIX + 1) vh_histogram[tx] = 0;
    __syncthreads();

    // --- Head (scalar) ---
    for (int i = tx; i < head_count; i += BLOCK_SIZE) {
        hist_scalar<ScoreT>(input + row_start, i, vh_histogram);
    }
    // --- Middle (vec4) ---
    for (int q = tx; q < quads; q += BLOCK_SIZE) {
        float f0, f1, f2, f3;
        load_quad_aligned<ScoreT>(input + row_start + head_count + 4 * q, f0, f1, f2, f3);
        ::atomicAdd(&vh_histogram[convert_to_uint8(f0)], 1);
        ::atomicAdd(&vh_histogram[convert_to_uint8(f1)], 1);
        ::atomicAdd(&vh_histogram[convert_to_uint8(f2)], 1);
        ::atomicAdd(&vh_histogram[convert_to_uint8(f3)], 1);
    }
    // --- Tail (scalar) ---
    for (int i = tx; i < tail_count; i += BLOCK_SIZE) {
        hist_scalar<ScoreT>(input + row_start, tail_start + i, vh_histogram);
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
        // --- Head ---
        for (int i = tx; i < head_count; i += BLOCK_SIZE) {
            if (convert_to_uint8(vortex_to_float(input[row_start + i])) > initial_threshold_bin) {
                const int pos = ::atomicAdd(&vh_counter, 1);
                index[pos] = i;
            }
        }
        // --- Middle vec4 ---
        for (int q = tx; q < quads; q += BLOCK_SIZE) {
            float f0, f1, f2, f3;
            load_quad_aligned<ScoreT>(input + row_start + head_count + 4 * q, f0, f1, f2, f3);
            float fs[4] = {f0, f1, f2, f3};
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                if (convert_to_uint8(fs[k]) > initial_threshold_bin) {
                    const int pos = ::atomicAdd(&vh_counter, 1);
                    index[pos] = head_count + 4 * q + k;
                }
            }
        }
        // --- Tail ---
        for (int i = tx; i < tail_count; i += BLOCK_SIZE) {
            const int idx = tail_start + i;
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

    // Filter pass — emit strict + collect ties + byte-3 sub-histogram.
    auto handle_one = [&](float f, int idx) {
        const int bin = static_cast<int>(convert_to_uint8(f));
        if (bin > initial_threshold_bin) {
            const int pos = ::atomicAdd(&vh_counter, 1);
            index[pos] = idx;
        } else if (bin == initial_threshold_bin) {
            const int pos = ::atomicAdd(&vh_num_input[0], 1);
            if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
                vh_input_idx[0][pos] = idx;
                const auto b32 = convert_to_uint32(f);
                const auto sub_bin = (b32 >> 24) & 0xFF;
                ::atomicAdd(&vh_histogram[sub_bin], 1);
            }
        }
    };

    // --- Head ---
    for (int i = tx; i < head_count; i += BLOCK_SIZE) {
        handle_one(vortex_to_float(input[row_start + i]), i);
    }
    // --- Middle vec4 ---
    for (int q = tx; q < quads; q += BLOCK_SIZE) {
        float f0, f1, f2, f3;
        load_quad_aligned<ScoreT>(input + row_start + head_count + 4 * q, f0, f1, f2, f3);
        const int base = head_count + 4 * q;
        handle_one(f0, base);
        handle_one(f1, base + 1);
        handle_one(f2, base + 2);
        handle_one(f3, base + 3);
    }
    // --- Tail ---
    for (int i = tx; i < tail_count; i += BLOCK_SIZE) {
        const int idx = tail_start + i;
        handle_one(vortex_to_float(input[row_start + idx]), idx);
    }
    __syncthreads();

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
    int*          __restrict__ out_blk   = sparse_kv_indices + sparse_kv_indptr[bx] + page_reserved_bos;

    __shared__ int s_indices[VORTEX_MAX_TOPK];

    if constexpr (std::is_same_v<ScoreT, __nv_bfloat16>) {
        // `row_start` here is 0 — we pass score_blk already offset, so alignment is relative to score_blk itself.
        // The aligned helper inspects (row_start % align_elems) and computes head_count from that.
        // Convert "absolute" alignment via pointer-to-base alignment: assume the underlying `score` tensor
        // is at least 16-byte aligned (true for CUDA allocations), so alignment relative to score_blk is
        // determined by `start % align_elems` = `start % 4` for bf16.
        const int eff_row_start = start;  // pass the absolute offset into the aligned helper
        // re-invoke with score (not score_blk) and absolute eff_row_start so alignment matches the base allocation.
        fast_topk_aligned<ScoreT, /*MaxRefineRounds=*/2>(score, s_indices, eff_row_start, nblk, topk_val);
    } else {
        fast_topk_aligned<ScoreT, /*MaxRefineRounds=*/4>(score, s_indices, start, nblk, topk_val);
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
