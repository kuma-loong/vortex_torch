// Sort top-k for the trtllm attention backend (indptr-free ABI).
//
// Same CUB BlockRadixSort logic as ``sort_topk.cu``, but everything is
// block-table-shaped:
//   * score / dense_block_tables / sparse_block_tables are 2D buffers
//     keyed at ``[eff_bs, max_blocks_per_seq]``.
//   * Per-row block counts are derived from ``dense_seqlens[bx]`` (token
//     count, written by the trtllm planner) and ``block_size``:
//         block_count = (tokens + block_size - 1) / block_size
//   * Per-row topk budget is derived from ``sparse_seqlens[bx]`` the same
//     way.
// There is intentionally **no** ``dense_kv_indptr`` / ``sparse_kv_indptr``
// argument — this kernel never reads a prefix-sum buffer.
//
// Compile-time toggle:
//   -DVORTEX_ENABLE_FP8  — compile in Float8_e4m3fn / Float8_e5m2 support.

#include <ATen/core/TensorBase.h>
#include <ATen/core/TensorBody.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Exception.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include <math_constants.h>
#ifdef VORTEX_ENABLE_FP8
#include <cuda_fp8.h>
#endif


template <typename T>
__device__ __forceinline__ float score_to_float(T v);

template <>
__device__ __forceinline__ float score_to_float<float>(float v) {
    return v;
}

template <>
__device__ __forceinline__ float score_to_float<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

#ifdef VORTEX_ENABLE_FP8
__device__ __forceinline__ float fp8_to_fp32(__nv_fp8_e4m3 v) {
    return __half2float(static_cast<__half>(
        __nv_cvt_fp8_to_halfraw(v.__x, __NV_E4M3)));
}

__device__ __forceinline__ float fp8_to_fp32(__nv_fp8_e5m2 v) {
    return __half2float(static_cast<__half>(
        __nv_cvt_fp8_to_halfraw(v.__x, __NV_E5M2)));
}

template <>
__device__ __forceinline__ float score_to_float<__nv_fp8_e4m3>(__nv_fp8_e4m3 v) {
    return fp8_to_fp32(v);
}

template <>
__device__ __forceinline__ float score_to_float<__nv_fp8_e5m2>(__nv_fp8_e5m2 v) {
    return fp8_to_fp32(v);
}
#endif


template <typename T>
__device__ __forceinline__ T score_neg_inf();

template <>
__device__ __forceinline__ float score_neg_inf<float>() {
    return -CUDART_INF_F;
}

template <>
__device__ __forceinline__ __nv_bfloat16 score_neg_inf<__nv_bfloat16>() {
    return __float2bfloat16(-CUDART_INF_F);
}

#ifdef VORTEX_ENABLE_FP8
template <>
__device__ __forceinline__ __nv_fp8_e4m3 score_neg_inf<__nv_fp8_e4m3>() {
    __nv_fp8_e4m3 v;
    v.__x = 0xFE;
    return v;
}

template <>
__device__ __forceinline__ __nv_fp8_e5m2 score_neg_inf<__nv_fp8_e5m2>() {
    __nv_fp8_e5m2 v;
    v.__x = 0xFC;
    return v;
}
#endif


template <typename ScoreT, int NUM_THREADS, int ITEM_PER_THREAD>
__global__ void TopKOutput_Kernel(
const ScoreT* __restrict__ score,
const int*    __restrict__ dense_seqlens,      // [eff_bs] tokens
const int*    __restrict__ sparse_seqlens,     // [eff_bs] tokens
const int*    __restrict__ dense_block_tables,
int*          __restrict__ sparse_block_tables,
const int     row_stride,                       // = max_blocks_per_seq
const int     block_reserved_bos,
const int     block_reserved_eos,
const int     block_size)
{
    const int bx = blockIdx.x;

    // Derive per-row block counts from seqlens (tokens) + block_size. The
    // ``Sgl_Decode_Plan_Workload_V2_Kernel<IS_TRTLLM=true>`` planner bakes
    // the same ``row * row_stride + col`` linear layout into
    // ``winfo_kv_offsets``, so the per-row score base matches here.
    const int dense_block_len  = (dense_seqlens[bx]  + block_size - 1) / block_size;
    const int sparse_block_len = (sparse_seqlens[bx] + block_size - 1) / block_size;
    const int row_offset = bx * row_stride;
    const int score_start = row_offset + block_reserved_bos;
    const int topk_val = sparse_block_len - block_reserved_bos - block_reserved_eos;
    const int nblk = dense_block_len - block_reserved_bos - block_reserved_eos;
    if (nblk <= topk_val) return;
    const ScoreT* __restrict__ score_blk = score + score_start;
    const int*    __restrict__ idx_blk   = dense_block_tables  + row_offset + block_reserved_bos;
    int*          __restrict__ out_blk   = sparse_block_tables + row_offset + block_reserved_bos;

    const ScoreT ninf_score = score_neg_inf<ScoreT>();

    ScoreT key_raw[ITEM_PER_THREAD];
    float  key[ITEM_PER_THREAD];
    int    val[ITEM_PER_THREAD];

    using BLF  = cub::BlockLoad<ScoreT, NUM_THREADS, ITEM_PER_THREAD, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BLI  = cub::BlockLoad<int,    NUM_THREADS, ITEM_PER_THREAD, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BSI  = cub::BlockStore<int,   NUM_THREADS, ITEM_PER_THREAD, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using Sort = cub::BlockRadixSort<float, NUM_THREADS, ITEM_PER_THREAD, int>;

    __shared__ union {
        typename BLF::TempStorage  lf;
        typename BLI::TempStorage  li;
        typename BSI::TempStorage  si;
        typename Sort::TempStorage sort;
    } temp;

    BLF(temp.lf).Load(score_blk, key_raw, nblk, ninf_score);

    #pragma unroll
    for (int i = 0; i < ITEM_PER_THREAD; ++i){
        key[i] = score_to_float<ScoreT>(key_raw[i]);
    }
    __syncthreads();

    BLI(temp.li).Load(idx_blk, val, nblk, 0);
    __syncthreads();

    Sort(temp.sort).SortDescending(key, val);
    __syncthreads();

    const int valid_out = min(topk_val, nblk);
    BSI(temp.si).Store(out_blk, val, valid_out);
}


template <typename ScoreT>
static void dispatch_topk(
const ScoreT*       x_ptr,
const at::Tensor&   dense_seqlens,
const at::Tensor&   sparse_seqlens,
const at::Tensor&   dense_block_tables,
at::Tensor&         sparse_block_tables,
const int64_t       eff_batch_size,
const int64_t       reserved_bos,
const int64_t       reserved_eos,
const int64_t       row_stride,
const int64_t       block_size,
cudaStream_t        stream)
{
    dim3 nblks(eff_batch_size);

    #define LAUNCH_TOPK(THREADS, ITEMS)                                             \
        TopKOutput_Kernel<ScoreT, THREADS, ITEMS><<<nblks, THREADS, 0, stream>>>(   \
            x_ptr,                                                                  \
            dense_seqlens.data_ptr<int>(),                                          \
            sparse_seqlens.data_ptr<int>(),                                         \
            dense_block_tables.data_ptr<int>(),                                     \
            sparse_block_tables.data_ptr<int>(),                                    \
            static_cast<int>(row_stride),                                           \
            reserved_bos,                                                           \
            reserved_eos,                                                           \
            static_cast<int>(block_size))

    if (row_stride <= 128)          { LAUNCH_TOPK(128, 1);  }
    else if (row_stride <= 256)     { LAUNCH_TOPK(128, 2);  }
    else if (row_stride <= 512)     { LAUNCH_TOPK(128, 4);  }
    else if (row_stride <= 1024)    { LAUNCH_TOPK(128, 8);  }
    else if (row_stride <= 1536)    { LAUNCH_TOPK(128, 12); }
    else if (row_stride <= 2048)    { LAUNCH_TOPK(128, 16); }
    else if (row_stride <= 2560)    { LAUNCH_TOPK(256, 10); }
    else if (row_stride <= 3072)    { LAUNCH_TOPK(256, 12); }
    else if (row_stride <= 3584)    { LAUNCH_TOPK(256, 14); }
    else if (row_stride <= 4096)    { LAUNCH_TOPK(256, 16); }
    else {
        TORCH_CHECK(false, "topk: row_stride (max_blocks_per_seq) > 4096 not supported");
    }

    #undef LAUNCH_TOPK
}


void topk(
const at::Tensor& x,
const at::Tensor& dense_seqlens,
const at::Tensor& sparse_seqlens,
const at::Tensor& dense_block_tables,
at::Tensor&       sparse_block_tables,
const int64_t     eff_batch_size,
const int64_t     reserved_bos,
const int64_t     reserved_eos,
const int64_t     max_blocks_per_seq,
const int64_t     block_size
){
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const auto dtype = x.scalar_type();

    if (dtype == at::ScalarType::Float) {
        dispatch_topk<float>(
            x.data_ptr<float>(),
            dense_seqlens, sparse_seqlens, dense_block_tables, sparse_block_tables,
            eff_batch_size, reserved_bos, reserved_eos, max_blocks_per_seq, block_size, stream);
    } else if (dtype == at::ScalarType::BFloat16) {
        dispatch_topk<__nv_bfloat16>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
            dense_seqlens, sparse_seqlens, dense_block_tables, sparse_block_tables,
            eff_batch_size, reserved_bos, reserved_eos, max_blocks_per_seq, block_size, stream);
    }
#ifdef VORTEX_ENABLE_FP8
    else if (dtype == at::ScalarType::Float8_e4m3fn) {
        dispatch_topk<__nv_fp8_e4m3>(
            reinterpret_cast<const __nv_fp8_e4m3*>(x.data_ptr()),
            dense_seqlens, sparse_seqlens, dense_block_tables, sparse_block_tables,
            eff_batch_size, reserved_bos, reserved_eos, max_blocks_per_seq, block_size, stream);
    } else if (dtype == at::ScalarType::Float8_e5m2) {
        dispatch_topk<__nv_fp8_e5m2>(
            reinterpret_cast<const __nv_fp8_e5m2*>(x.data_ptr()),
            dense_seqlens, sparse_seqlens, dense_block_tables, sparse_block_tables,
            eff_batch_size, reserved_bos, reserved_eos, max_blocks_per_seq, block_size, stream);
    }
#endif
    else {
        TORCH_CHECK(false, "topk: unsupported dtype ", dtype);
    }

    const auto result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess,
                "topk kernel failed: ", ::cudaGetErrorString(result));
}
