/**
 * approx_topk.cu — single-pass 8-bit radix approximate top-K, with an
 * optional second pass when the threshold bin is too wide.
 *
 *   Pass 1 — histogram on byte 0 of fp32 keys, find threshold bin tbin0,
 *            compute last_remain0 = target_k - hist0[tbin0+1].
 *
 *   If last_remain0 <= tolerate_ratio * target_k:
 *      Single-pass emit: items with bin0 > tbin0 are strict winners,
 *      items with bin0 == tbin0 are filled in atomic-arrival order.
 *
 *   Otherwise (threshold bin too wide):
 *      Pass 2 — emit byte-0 strict winners and build a sub-histogram on
 *               byte 1 of the byte-0 == tbin0 items.
 *      Find tbin1 within that sub-histogram, then a final emit pass:
 *      items with bin0 == tbin0 && bin1 > tbin1 are strict winners,
 *      items with bin0 == tbin0 && bin1 == tbin1 fill the remaining
 *      slots in atomic-arrival order.  No further refinement.
 *
 *   tolerate_ratio = 1.0 -> always single-pass (cheapest, loosest).
 *   tolerate_ratio = 0.0 -> always two-pass (tighter approximation).
 */

#include "register.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {

constexpr int kThreadsPerBlock = 1024;
constexpr int RADIX = 256;
constexpr int VORTEX_MAX_TOPK = 2048;

template <typename T>
__device__ __forceinline__ float to_float(T x);

template <>
__device__ __forceinline__ float to_float<float>(float x) { return x; }

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

// fp32 -> total-order uint32 (sign-flip).  Higher key == higher score.
__device__ __forceinline__ uint32_t score_to_key32(float x) {
    uint32_t bits = __float_as_uint(x);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}


template <typename ScoreT>
__device__ void approx_topk_inner(
    const ScoreT* __restrict__ input,
    int*          __restrict__ index,
    const int     length,
    const int     target_k,
    const int     tolerate_thresh)
{
    constexpr int BLOCK_SIZE = kThreadsPerBlock;

    alignas(128) __shared__ int hist_buf[2][RADIX + 128];
    alignas(128) __shared__ int s_threshold_bin;
    alignas(128) __shared__ int s_counter;        // strict-winner write head
    alignas(128) __shared__ int s_last_remain;    // atomic-arrival countdown

    auto& hist = hist_buf[0];
    const int tx = threadIdx.x;

    // Reverse inclusive cumulative sum; final result lands in hist_buf[0].
    auto run_cumsum = [&] {
        #pragma unroll 8
        for (int i = 0; i < 8; ++i) {
            if (C10_LIKELY(tx < RADIX)) {
                const int j = 1 << i;
                const int k = i & 1;
                int v = hist_buf[k][tx];
                if (tx < RADIX - j) v += hist_buf[k][tx + j];
                hist_buf[k ^ 1][tx] = v;
            }
            __syncthreads();
        }
    };

    // ---------------- Pass 1: histogram on byte 0 ----------------
    if (tx < RADIX + 1) hist[tx] = 0;
    __syncthreads();

    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
        const auto bin =
            (score_to_key32(to_float<ScoreT>(input[idx])) >> 24) & 0xFFu;
        ::atomicAdd(&hist[bin], 1);
    }
    __syncthreads();

    run_cumsum();

    if (tx < RADIX && hist[tx] > target_k && hist[tx + 1] <= target_k) {
        s_threshold_bin = tx;
        s_counter       = 0;
        s_last_remain   = target_k - hist[tx + 1];
    }
    __syncthreads();

    const int tbin0        = s_threshold_bin;
    const int last_remain0 = s_last_remain;

    // ---------------- Single-pass emit ----------------
    if (last_remain0 <= tolerate_thresh) {
        for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
            const auto bin =
                (score_to_key32(to_float<ScoreT>(input[idx])) >> 24) & 0xFFu;
            if (bin > tbin0) {
                const int pos = ::atomicAdd(&s_counter, 1);
                index[pos] = idx;
            } else if (bin == tbin0) {
                const int pos = ::atomicAdd(&s_last_remain, -1);
                if (pos > 0) {
                    index[target_k - pos] = idx;
                }
            }
        }
        __syncthreads();
        return;
    }

    // ---------------- Pass 2: emit byte-0 strict + byte-1 sub-histogram ----------------
    if (tx < RADIX + 1) hist[tx] = 0;
    __syncthreads();

    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
        const auto key32 = score_to_key32(to_float<ScoreT>(input[idx]));
        const auto bin0  = (key32 >> 24) & 0xFFu;
        if (bin0 > tbin0) {
            const int pos = ::atomicAdd(&s_counter, 1);
            index[pos] = idx;
        } else if (bin0 == tbin0) {
            const auto bin1 = (key32 >> 16) & 0xFFu;
            ::atomicAdd(&hist[bin1], 1);
        }
    }
    __syncthreads();

    run_cumsum();

    // Find byte-1 threshold bin against the new top-k = last_remain0.
    if (tx < RADIX && hist[tx] > last_remain0 && hist[tx + 1] <= last_remain0) {
        s_threshold_bin = tx;
        s_last_remain   = last_remain0 - hist[tx + 1];
    }
    __syncthreads();

    const int tbin1 = s_threshold_bin;

    // ---------------- Pass 3: byte-1 strict + atomic-arrival ----------------
    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
        const auto key32 = score_to_key32(to_float<ScoreT>(input[idx]));
        const auto bin0  = (key32 >> 24) & 0xFFu;
        if (bin0 != tbin0) continue;
        const auto bin1 = (key32 >> 16) & 0xFFu;
        if (bin1 > tbin1) {
            const int pos = ::atomicAdd(&s_counter, 1);
            index[pos] = idx;
        } else if (bin1 == tbin1) {
            const int pos = ::atomicAdd(&s_last_remain, -1);
            if (pos > 0) {
                index[target_k - pos] = idx;
            }
        }
    }
    __syncthreads();
}


template <typename ScoreT>
__global__ __launch_bounds__(kThreadsPerBlock)
void ApproxTopK_Kernel(
    const ScoreT* __restrict__ score,
    const int*    __restrict__ dense_kv_indptr,
    const int*    __restrict__ sparse_kv_indptr,
    const int*    __restrict__ dense_kv_indices,
    int*          __restrict__ sparse_kv_indices,
    const int     page_reserved_bos,
    const int     page_reserved_eos,
    const float   tolerate_ratio)
{
    const int bx = blockIdx.x;

    const int start    = dense_kv_indptr[bx] + page_reserved_bos;
    const int end      = dense_kv_indptr[bx + 1] - page_reserved_eos;
    const int target_k = sparse_kv_indptr[bx + 1] - sparse_kv_indptr[bx]
                         - page_reserved_bos - page_reserved_eos;
    const int nblk     = end - start;
    if (nblk <= target_k) return;

    int tolerate_thresh =
        static_cast<int>(tolerate_ratio * static_cast<float>(target_k));
    if (tolerate_thresh < 0)        tolerate_thresh = 0;
    if (tolerate_thresh > target_k) tolerate_thresh = target_k;

    const ScoreT* __restrict__ score_blk = score + start;
    const int*    __restrict__ idx_blk   = dense_kv_indices + start;
    int*          __restrict__ out_blk   = sparse_kv_indices
                                         + sparse_kv_indptr[bx]
                                         + page_reserved_bos;

    __shared__ int s_indices[VORTEX_MAX_TOPK];

    approx_topk_inner<ScoreT>(score_blk, s_indices, nblk, target_k, tolerate_thresh);
    __syncthreads();

    const int tx = threadIdx.x;
    for (int i = tx; i < target_k; i += kThreadsPerBlock) {
        out_blk[i] = idx_blk[s_indices[i]];
    }
}

}  // namespace


void approx_topk_output(
    const at::Tensor& x,
    const at::Tensor& dense_kv_indptr,
    const at::Tensor& sparse_kv_indptr,
    const at::Tensor& dense_kv_indices,
    at::Tensor&       sparse_kv_indices,
    const int64_t     eff_batch_size,
    const int64_t     reserved_bos,
    const int64_t     reserved_eos,
    const int64_t     max_num_pages,
    const double      tolerate_ratio)
{
    (void)max_num_pages;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    dim3 nblks(eff_batch_size);
    dim3 nthreads(kThreadsPerBlock);

    const float tol = static_cast<float>(tolerate_ratio);

    if (x.scalar_type() == at::ScalarType::BFloat16) {
        ApproxTopK_Kernel<__nv_bfloat16><<<nblks, nthreads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            dense_kv_indptr.data_ptr<int>(),
            sparse_kv_indptr.data_ptr<int>(),
            dense_kv_indices.data_ptr<int>(),
            sparse_kv_indices.data_ptr<int>(),
            static_cast<int>(reserved_bos),
            static_cast<int>(reserved_eos),
            tol);
    } else if (x.scalar_type() == at::ScalarType::Float) {
        ApproxTopK_Kernel<float><<<nblks, nthreads, 0, stream>>>(
            x.data_ptr<float>(),
            dense_kv_indptr.data_ptr<int>(),
            sparse_kv_indptr.data_ptr<int>(),
            dense_kv_indices.data_ptr<int>(),
            sparse_kv_indices.data_ptr<int>(),
            static_cast<int>(reserved_bos),
            static_cast<int>(reserved_eos),
            tol);
    } else {
        TORCH_CHECK(false,
                    "approx_topk_output: unsupported dtype ", x.scalar_type());
    }

    const auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "approx_topk_output kernel failed: ", cudaGetErrorString(err));
}
