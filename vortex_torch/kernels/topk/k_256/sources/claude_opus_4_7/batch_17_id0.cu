// Split-K top-K for small batch sizes.
//
// Strategy: when `eff_batch_size ≤ SPLIT_THRESHOLD` the regular "one CTA per
// segment" launch leaves most SMs idle (L40 has ~324 max-residency CTAs).
// This kernel launches `bs × num_splits` CTAs, where each CTA computes a
// local top-K on its sub-segment, writes the K indices to a global
// workspace, and atomically increments a per-segment done counter. The
// CTA that observes `prev_count == num_splits - 1` (the last arrival) does
// the final merge: gather `num_splits × K` candidate scores, run a single-
// segment radix top-K on them, write the result to `sparse_kv_indices`.
//
// Two paths in the host function:
//   - bs > SPLIT_THRESHOLD: launch the existing aligned vec4 + explicit-skip
//     kernel (one CTA per segment).
//   - bs ≤ SPLIT_THRESHOLD: allocate workspace, launch split kernel.
//
// num_splits is chosen at runtime: `min(MAX_SPLITS, max(1, 256 / bs))`.
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
#include <math_constants.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace {

constexpr int kThreadsPerBlock = __THREADS_PER_BLOCK__;
constexpr size_t kSmem = __SMEM_BYTES__;
constexpr int VORTEX_MAX_TOPK = __VORTEX_MAX_TOPK__;
constexpr int NUM_SPLITS_CAP = __NUM_SPLITS_CAP__;  // compile-time max num_splits
constexpr int SPLIT_THRESHOLD = 64;  // bs ≤ this triggers split-K

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

template <typename ScoreT>
__device__ __forceinline__ int required_alignment_in_elems();
template <> __device__ __forceinline__ int required_alignment_in_elems<__nv_bfloat16>() { return 4; }
template <> __device__ __forceinline__ int required_alignment_in_elems<float>() { return 1; }

// =====================================================================
// fast_topk_aligned — alignment-safe vec4 + bf16-aware explicit-skip.
// (Same as batch_9_id0.cu; reused here for both the regular path AND
// the local top-K within a split.)
// =====================================================================
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

    const int align_elems = required_alignment_in_elems<ScoreT>();
    int head_count = (align_elems - (row_start % align_elems)) % align_elems;
    if (head_count > length) head_count = length;
    const int aligned_length = length - head_count;
    const int quads = aligned_length >> 2;
    const int tail_start = head_count + (quads << 2);
    const int tail_count = length - tail_start;

    if (tx < RADIX + 1) vh_histogram[tx] = 0;
    __syncthreads();

    for (int i = tx; i < head_count; i += BLOCK_SIZE) {
        const float f = vortex_to_float(input[row_start + i]);
        ::atomicAdd(&vh_histogram[convert_to_uint8(f)], 1);
    }
    for (int q = tx; q < quads; q += BLOCK_SIZE) {
        float f0, f1, f2, f3;
        load_quad_aligned<ScoreT>(input + row_start + head_count + 4 * q, f0, f1, f2, f3);
        ::atomicAdd(&vh_histogram[convert_to_uint8(f0)], 1);
        ::atomicAdd(&vh_histogram[convert_to_uint8(f1)], 1);
        ::atomicAdd(&vh_histogram[convert_to_uint8(f2)], 1);
        ::atomicAdd(&vh_histogram[convert_to_uint8(f3)], 1);
    }
    for (int i = tx; i < tail_count; i += BLOCK_SIZE) {
        const float f = vortex_to_float(input[row_start + tail_start + i]);
        ::atomicAdd(&vh_histogram[convert_to_uint8(f)], 1);
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
        for (int i = tx; i < head_count; i += BLOCK_SIZE) {
            if (convert_to_uint8(vortex_to_float(input[row_start + i])) > initial_threshold_bin) {
                const int pos = ::atomicAdd(&vh_counter, 1);
                index[pos] = i;
            }
        }
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
    for (int i = tx; i < head_count; i += BLOCK_SIZE) {
        handle_one(vortex_to_float(input[row_start + i]), i);
    }
    for (int q = tx; q < quads; q += BLOCK_SIZE) {
        float f0, f1, f2, f3;
        load_quad_aligned<ScoreT>(input + row_start + head_count + 4 * q, f0, f1, f2, f3);
        const int base = head_count + 4 * q;
        handle_one(f0, base);
        handle_one(f1, base + 1);
        handle_one(f2, base + 2);
        handle_one(f3, base + 3);
    }
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

// =====================================================================
// fast_topk_subset — top-K over an indirect candidate set (used for the
// merge phase of split-K). Candidates are described by `cand_indices`
// (segment-relative, may contain -1 sentinels for invalid).
// =====================================================================
template <typename ScoreT, int MaxRefineRounds>
__device__ void fast_topk_subset(
    const ScoreT* __restrict__ score_seg,  // base of this segment in the score array
    const int*    __restrict__ cand_indices,  // [num_cands] — segment-relative idx or -1
    int           num_cands,
    int*          __restrict__ index,       // out: top-K segment-relative indices
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

    // Round 0 histogram — scalar; candidates may be sparse / non-aligned.
    for (int i = tx; i < num_cands; i += BLOCK_SIZE) {
        const int seg_rel = cand_indices[i];
        if (seg_rel < 0) continue;
        const float f = vortex_to_float(score_seg[seg_rel]);
        ::atomicAdd(&vh_histogram[convert_to_uint8(f)], 1);
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
        for (int i = tx; i < num_cands; i += BLOCK_SIZE) {
            const int seg_rel = cand_indices[i];
            if (seg_rel < 0) continue;
            if (convert_to_uint8(vortex_to_float(score_seg[seg_rel])) > initial_threshold_bin) {
                const int pos = ::atomicAdd(&vh_counter, 1);
                index[pos] = seg_rel;
            }
        }
        __syncthreads();
        return;
    }

    __syncthreads();
    if (tx < RADIX + 1) vh_histogram[tx] = 0;
    __syncthreads();

    for (int i = tx; i < num_cands; i += BLOCK_SIZE) {
        const int seg_rel = cand_indices[i];
        if (seg_rel < 0) continue;
        const float f = vortex_to_float(score_seg[seg_rel]);
        const int bin = static_cast<int>(convert_to_uint8(f));
        if (bin > initial_threshold_bin) {
            const int pos = ::atomicAdd(&vh_counter, 1);
            index[pos] = seg_rel;
        } else if (bin == initial_threshold_bin) {
            const int pos = ::atomicAdd(&vh_num_input[0], 1);
            if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
                vh_input_idx[0][pos] = seg_rel;
                const auto b32 = convert_to_uint32(f);
                const auto sub_bin = (b32 >> 24) & 0xFF;
                ::atomicAdd(&vh_histogram[sub_bin], 1);
            }
        }
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
                const int seg_rel = vh_input_idx[r_idx][i];
                const auto offset = 24 - round * 8;
                const auto bin = (convert_to_uint32(
                    vortex_to_float(score_seg[seg_rel])) >> offset) & 0xFF;
                if (bin > threshold_bin) {
                    const int pos = ::atomicAdd(&vh_counter, 1);
                    index[pos] = seg_rel;
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
                const int seg_rel = vh_input_idx[r_idx][i];
                const auto offset = 24 - round * 8;
                const auto bin = (convert_to_uint32(
                    vortex_to_float(score_seg[seg_rel])) >> offset) & 0xFF;
                if (bin > threshold_bin) {
                    const int pos = ::atomicAdd(&vh_counter, 1);
                    index[pos] = seg_rel;
                } else if (bin == threshold_bin) {
                    const int pos = ::atomicAdd(&vh_last_remain, -1);
                    if (pos > 0) {
                        index[target_k - pos] = seg_rel;
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
            const int seg_rel = vh_input_idx[r_idx][i];
            const auto raw_input = vortex_to_float(score_seg[seg_rel]);
            const auto offset = 24 - round * 8;
            const auto bin = (convert_to_uint32(raw_input) >> offset) & 0xFF;
            if (bin > threshold_bin) {
                const int pos = ::atomicAdd(&vh_counter, 1);
                index[pos] = seg_rel;
            } else if (bin == threshold_bin) {
                const int pos = ::atomicAdd(&vh_num_input[r_idx ^ 1], 1);
                if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
                    vh_input_idx[r_idx ^ 1][pos] = seg_rel;
                    const auto b32 = convert_to_uint32(raw_input);
                    const auto sub_bin = (b32 >> (offset - 8)) & 0xFF;
                    ::atomicAdd(&vh_histogram[sub_bin], 1);
                }
            }
        }
        __syncthreads();
    }
}

// =====================================================================
// Split-K kernel: each CTA handles one (segment, split) pair.
// =====================================================================
template <typename ScoreT, int MaxRefineRounds>
__global__ __launch_bounds__(kThreadsPerBlock)
void SplitTopK_Kernel(
    const ScoreT* __restrict__ score,
    const int*    __restrict__ dense_kv_indptr,
    const int*    __restrict__ sparse_kv_indptr,
    const int*    __restrict__ dense_kv_indices,
    int*          __restrict__ sparse_kv_indices,
    int*          __restrict__ workspace,    // [bs * num_splits * MAX_TOPK]
    int*          __restrict__ done_counter, // [bs]
    const int     page_reserved_bos,
    const int     page_reserved_eos,
    const int     num_splits)
{
    const int bx = blockIdx.x;
    const int seg = bx / num_splits;
    const int split = bx % num_splits;

    const int seg_start = dense_kv_indptr[seg] + page_reserved_bos;
    const int seg_end   = dense_kv_indptr[seg + 1] - page_reserved_eos;
    const int seg_len   = seg_end - seg_start;
    const int target_k  = sparse_kv_indptr[seg + 1] - sparse_kv_indptr[seg] - page_reserved_bos - page_reserved_eos;

    if (seg_len <= target_k) {
        // Edge case: just identity for this segment. Only the first split CTA handles it.
        if (split == 0) {
            int* out_blk = sparse_kv_indices + sparse_kv_indptr[seg] + page_reserved_bos;
            const int* idx_blk = dense_kv_indices + seg_start;
            const int tx = threadIdx.x;
            for (int i = tx; i < seg_len; i += kThreadsPerBlock) {
                out_blk[i] = idx_blk[i];
            }
        }
        return;
    }

    // Split boundaries.
    const int split_size_base = seg_len / num_splits;
    const int split_extra     = seg_len - split_size_base * num_splits;
    const int split_start_in_seg = split * split_size_base + min(split, split_extra);
    const int actual_split_size  = split_size_base + (split < split_extra ? 1 : 0);

    __shared__ int s_local[VORTEX_MAX_TOPK];

    // Phase 1 — local top-K of this split's elements.
    int local_size = 0;
    if (actual_split_size <= target_k) {
        // Smaller than K — keep all indices.
        const int tx = threadIdx.x;
        for (int i = tx; i < actual_split_size; i += kThreadsPerBlock) {
            s_local[i] = i;
        }
        local_size = actual_split_size;
        __syncthreads();
    } else {
        // Compute local top-K. row_start is "absolute" so alignment works on the base allocation.
        fast_topk_aligned<ScoreT, MaxRefineRounds>(
            score, s_local, seg_start + split_start_in_seg, actual_split_size, target_k);
        local_size = target_k;
    }

    // Write `s_local + split_start_in_seg` (now segment-relative) to global workspace.
    int* my_ws = workspace + (seg * num_splits + split) * VORTEX_MAX_TOPK;
    const int tx = threadIdx.x;
    for (int i = tx; i < local_size; i += kThreadsPerBlock) {
        my_ws[i] = s_local[i] + split_start_in_seg;
    }
    // Pad invalid slots with -1.
    for (int i = local_size + tx; i < VORTEX_MAX_TOPK; i += kThreadsPerBlock) {
        my_ws[i] = -1;
    }

    __threadfence();
    __syncthreads();

    // Atomic done counter.
    __shared__ int s_is_last;
    if (tx == 0) {
        const int prev = ::atomicAdd(&done_counter[seg], 1);
        s_is_last = (prev == num_splits - 1) ? 1 : 0;
    }
    __syncthreads();

    if (!s_is_last) return;

    // Phase 2 — simple bitonic-sort merge in smem (user feedback: simpler than
    // running another radix top-K). With NUM_SPLITS_CAP=2, we have at most
    // 2 * VORTEX_MAX_TOPK = 512 candidates → fits in smem.
    constexpr int MERGE_SIZE = 2 * VORTEX_MAX_TOPK;  // 512 for K=256
    constexpr int MERGE_SORT_T = MERGE_SIZE;          // = 512 = kThreadsPerBlock
    static_assert(MERGE_SORT_T <= 1024, "merge bitonic sort needs T threads ≥ MERGE_SIZE");

    __shared__ float s_keys[MERGE_SIZE];
    __shared__ int   s_vals[MERGE_SIZE];

    const int actual_total = num_splits * VORTEX_MAX_TOPK;
    const int* all_cands = workspace + seg * num_splits * VORTEX_MAX_TOPK;

    // Load candidates' (score, idx) pairs; pad with (-INF, -1) for sentinels & beyond actual_total.
    for (int i = tx; i < MERGE_SIZE; i += kThreadsPerBlock) {
        if (i < actual_total) {
            const int seg_rel = all_cands[i];
            if (seg_rel < 0) {
                s_keys[i] = -CUDART_INF_F;
                s_vals[i] = -1;
            } else {
                s_keys[i] = vortex_to_float(score[seg_start + seg_rel]);
                s_vals[i] = seg_rel;
            }
        } else {
            s_keys[i] = -CUDART_INF_F;
            s_vals[i] = -1;
        }
    }
    __syncthreads();

    // Bitonic sort (ascending). Each thread handles 1 element at index `tx`
    // (threads ≥ MERGE_SIZE are idle for the sort).
    // After sort: top-K = highest at indices [MERGE_SIZE-K, MERGE_SIZE-1].
    for (int k = 2; k <= MERGE_SIZE; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            const int ixj = tx ^ j;
            if (tx < MERGE_SIZE && ixj > tx && ixj < MERGE_SIZE) {
                const bool dir_up = ((tx & k) == 0);  // ascending in this block
                const float a = s_keys[tx];
                const float b = s_keys[ixj];
                const bool swap = dir_up ? (a > b) : (a < b);
                if (swap) {
                    s_keys[tx] = b; s_keys[ixj] = a;
                    const int va = s_vals[tx];
                    s_vals[tx] = s_vals[ixj]; s_vals[ixj] = va;
                }
            }
            __syncthreads();
        }
    }

    // Output top target_k via dense_kv_indices remap (indices reversed: highest-first).
    int* out_blk = sparse_kv_indices + sparse_kv_indptr[seg] + page_reserved_bos;
    const int* idx_blk = dense_kv_indices + seg_start;
    for (int i = tx; i < target_k; i += kThreadsPerBlock) {
        const int seg_rel = s_vals[MERGE_SIZE - 1 - i];
        out_blk[i] = idx_blk[seg_rel];
    }

    // Reset done_counter[seg] to 0 for the next call.
    if (tx == 0) {
        done_counter[seg] = 0;
    }
}

// =====================================================================
// Regular top-K kernel (one CTA per segment) — used for large batch sizes.
// =====================================================================
template <typename ScoreT, int MaxRefineRounds>
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

    const int* __restrict__ idx_blk = dense_kv_indices + start;
    int*       __restrict__ out_blk = sparse_kv_indices + sparse_kv_indptr[bx] + page_reserved_bos;

    __shared__ int s_indices[VORTEX_MAX_TOPK];
    fast_topk_aligned<ScoreT, MaxRefineRounds>(score, s_indices, start, nblk, topk_val);
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

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const int bs = static_cast<int>(eff_batch_size);

    if (bs > SPLIT_THRESHOLD) {
        // Regular path — one CTA per segment.
        dim3 nblks(bs);
        dim3 nthreads(kThreadsPerBlock);
        if (x.scalar_type() == at::ScalarType::BFloat16) {
            setup_kernel_smem_once<TopKOutput_Kernel<__nv_bfloat16, 2>, kSmem>();
            TopKOutput_Kernel<__nv_bfloat16, 2><<<nblks, nthreads, kSmem, stream>>>(
                reinterpret_cast<__nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
                dense_kv_indptr.data_ptr<int>(),
                sparse_kv_indptr.data_ptr<int>(),
                dense_kv_indices.data_ptr<int>(),
                sparse_kv_indices.data_ptr<int>(),
                reserved_bos, reserved_eos);
        } else if (x.scalar_type() == at::ScalarType::Float) {
            setup_kernel_smem_once<TopKOutput_Kernel<float, 4>, kSmem>();
            TopKOutput_Kernel<float, 4><<<nblks, nthreads, kSmem, stream>>>(
                x.data_ptr<float>(),
                dense_kv_indptr.data_ptr<int>(),
                sparse_kv_indptr.data_ptr<int>(),
                dense_kv_indices.data_ptr<int>(),
                sparse_kv_indices.data_ptr<int>(),
                reserved_bos, reserved_eos);
        } else {
            TORCH_CHECK(false, "topk: unsupported dtype ", x.scalar_type());
        }
    } else {
        // Split-K path — `num_splits` CTAs per segment.
        // Smart heuristic: each split needs sl/num_splits ≥ 2*K elements
        // for the local top-K to be meaningful. Cap by NUM_SPLITS_CAP.
        const int sl_max = static_cast<int>(max_num_pages);
        int num_splits = sl_max / (2 * VORTEX_MAX_TOPK);  // floor(sl / (2K))
        if (num_splits > NUM_SPLITS_CAP) num_splits = NUM_SPLITS_CAP;
        if (num_splits < 1) num_splits = 1;

        if (num_splits == 1) {
            // Fall back to regular kernel (no split overhead).
            dim3 nblks(bs);
            dim3 nthreads(kThreadsPerBlock);
            if (x.scalar_type() == at::ScalarType::BFloat16) {
                setup_kernel_smem_once<TopKOutput_Kernel<__nv_bfloat16, 2>, kSmem>();
                TopKOutput_Kernel<__nv_bfloat16, 2><<<nblks, nthreads, kSmem, stream>>>(
                    reinterpret_cast<__nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
                    dense_kv_indptr.data_ptr<int>(),
                    sparse_kv_indptr.data_ptr<int>(),
                    dense_kv_indices.data_ptr<int>(),
                    sparse_kv_indices.data_ptr<int>(),
                    reserved_bos, reserved_eos);
            } else if (x.scalar_type() == at::ScalarType::Float) {
                setup_kernel_smem_once<TopKOutput_Kernel<float, 4>, kSmem>();
                TopKOutput_Kernel<float, 4><<<nblks, nthreads, kSmem, stream>>>(
                    x.data_ptr<float>(),
                    dense_kv_indptr.data_ptr<int>(),
                    sparse_kv_indptr.data_ptr<int>(),
                    dense_kv_indices.data_ptr<int>(),
                    sparse_kv_indices.data_ptr<int>(),
                    reserved_bos, reserved_eos);
            }
            const auto result = cudaGetLastError();
            TORCH_CHECK(result == cudaSuccess, "topk kernel failed: ", ::cudaGetErrorString(result));
            return;
        }

        // Module-level cached workspace + done counter, reset by last-arrival CTA.
        // First call: at::zeros initializes the counter; subsequent calls inherit
        // the reset-by-last-arrival pattern (counter is back to 0 after each call).
        static at::Tensor cached_workspace;
        static at::Tensor cached_done_counter;
        const int64_t needed_ws = static_cast<int64_t>(bs) * num_splits * VORTEX_MAX_TOPK;
        const int64_t needed_dc = bs;

        auto opts_int = dense_kv_indptr.options().dtype(at::kInt);
        if (!cached_workspace.defined() || cached_workspace.numel() < needed_ws) {
            cached_workspace = at::empty({needed_ws}, opts_int);
        }
        if (!cached_done_counter.defined() || cached_done_counter.numel() < needed_dc) {
            cached_done_counter = at::zeros({needed_dc}, opts_int);
        } else {
            // Counter was reset to 0 by last-arrival CTA; no need to re-init.
            // (Trust the kernel-side invariant.)
        }

        int* workspace_ptr   = cached_workspace.data_ptr<int>();
        int* done_counter_ptr = cached_done_counter.data_ptr<int>();

        dim3 nblks(bs * num_splits);
        dim3 nthreads(kThreadsPerBlock);

        if (x.scalar_type() == at::ScalarType::BFloat16) {
            setup_kernel_smem_once<SplitTopK_Kernel<__nv_bfloat16, 2>, kSmem>();
            SplitTopK_Kernel<__nv_bfloat16, 2><<<nblks, nthreads, kSmem, stream>>>(
                reinterpret_cast<__nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
                dense_kv_indptr.data_ptr<int>(),
                sparse_kv_indptr.data_ptr<int>(),
                dense_kv_indices.data_ptr<int>(),
                sparse_kv_indices.data_ptr<int>(),
                workspace_ptr,
                done_counter_ptr,
                reserved_bos, reserved_eos, num_splits);
        } else if (x.scalar_type() == at::ScalarType::Float) {
            setup_kernel_smem_once<SplitTopK_Kernel<float, 4>, kSmem>();
            SplitTopK_Kernel<float, 4><<<nblks, nthreads, kSmem, stream>>>(
                x.data_ptr<float>(),
                dense_kv_indptr.data_ptr<int>(),
                sparse_kv_indptr.data_ptr<int>(),
                dense_kv_indices.data_ptr<int>(),
                sparse_kv_indices.data_ptr<int>(),
                workspace_ptr,
                done_counter_ptr,
                reserved_bos, reserved_eos, num_splits);
        } else {
            TORCH_CHECK(false, "topk: unsupported dtype ", x.scalar_type());
        }
    }

    const auto result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess,
                "topk kernel failed: ", ::cudaGetErrorString(result));
}
