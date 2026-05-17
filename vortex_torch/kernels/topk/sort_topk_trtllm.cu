// Sort top-k for the trtllm attention backend.
//
// Same CUB BlockRadixSort logic as ``sort_topk.cu`` but the dense / sparse
// index buffers are 2D ``block_tables`` of shape ``[eff_bs,
// max_blocks_per_seq]``. The C entry-point keeps the legacy parameter
// names (``dense_kv_indices`` / ``sparse_kv_indices`` / ``max_num_pages``)
// for ABI parity with sort_topk.cu — semantically they are
// ``dense_block_tables`` / ``sparse_block_tables`` / row stride.
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
const int*    __restrict__ dense_kv_indptr,
const int*    __restrict__ sparse_kv_indptr,
const int*    __restrict__ dense_block_tables,
int*          __restrict__ sparse_block_tables,
const int     row_stride,                       // = max_blocks_per_seq
const int     page_reserved_bos,
const int     page_reserved_eos)
{
    const int bx = blockIdx.x;

    // Number of dense blocks for this (req, kv_head) row.
    const int dense_block_len = dense_kv_indptr[bx + 1] - dense_kv_indptr[bx];
    const int score_start = dense_kv_indptr[bx] + page_reserved_bos;
    const int topk_val =
        (sparse_kv_indptr[bx + 1] - sparse_kv_indptr[bx])
        - page_reserved_bos - page_reserved_eos;
    const int nblk = dense_block_len - page_reserved_bos - page_reserved_eos;
    if (nblk <= topk_val) return;

    const int row_offset = bx * row_stride;
    const ScoreT* __restrict__ score_blk = score + score_start;
    const int*    __restrict__ idx_blk   = dense_block_tables  + row_offset + page_reserved_bos;
    int*          __restrict__ out_blk   = sparse_block_tables + row_offset + page_reserved_bos;

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
const at::Tensor&   dense_kv_indptr,
const at::Tensor&   sparse_kv_indptr,
const at::Tensor&   dense_block_tables,
at::Tensor&         sparse_block_tables,
const int64_t       eff_batch_size,
const int64_t       reserved_bos,
const int64_t       reserved_eos,
const int64_t       row_stride,
cudaStream_t        stream)
{
    dim3 nblks(eff_batch_size);

    #define LAUNCH_TOPK(THREADS, ITEMS)                                             \
        TopKOutput_Kernel<ScoreT, THREADS, ITEMS><<<nblks, THREADS, 0, stream>>>(   \
            x_ptr,                                                                  \
            dense_kv_indptr.data_ptr<int>(),                                        \
            sparse_kv_indptr.data_ptr<int>(),                                       \
            dense_block_tables.data_ptr<int>(),                                     \
            sparse_block_tables.data_ptr<int>(),                                    \
            static_cast<int>(row_stride),                                           \
            reserved_bos,                                                           \
            reserved_eos)

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
const at::Tensor& dense_kv_indptr,
const at::Tensor& sparse_kv_indptr,
const at::Tensor& dense_kv_indices,    // semantically dense_block_tables
at::Tensor&       sparse_kv_indices,   // semantically sparse_block_tables
const int64_t     eff_batch_size,
const int64_t     reserved_bos,
const int64_t     reserved_eos,
const int64_t     max_num_pages        // semantically row_stride = max_blocks_per_seq
){
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const auto dtype = x.scalar_type();

    if (dtype == at::ScalarType::Float) {
        dispatch_topk<float>(
            x.data_ptr<float>(),
            dense_kv_indptr, sparse_kv_indptr, dense_kv_indices, sparse_kv_indices,
            eff_batch_size, reserved_bos, reserved_eos, max_num_pages, stream);
    } else if (dtype == at::ScalarType::BFloat16) {
        dispatch_topk<__nv_bfloat16>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
            dense_kv_indptr, sparse_kv_indptr, dense_kv_indices, sparse_kv_indices,
            eff_batch_size, reserved_bos, reserved_eos, max_num_pages, stream);
    }
#ifdef VORTEX_ENABLE_FP8
    else if (dtype == at::ScalarType::Float8_e4m3fn) {
        dispatch_topk<__nv_fp8_e4m3>(
            reinterpret_cast<const __nv_fp8_e4m3*>(x.data_ptr()),
            dense_kv_indptr, sparse_kv_indptr, dense_kv_indices, sparse_kv_indices,
            eff_batch_size, reserved_bos, reserved_eos, max_num_pages, stream);
    } else if (dtype == at::ScalarType::Float8_e5m2) {
        dispatch_topk<__nv_fp8_e5m2>(
            reinterpret_cast<const __nv_fp8_e5m2*>(x.data_ptr()),
            dense_kv_indptr, sparse_kv_indptr, dense_kv_indices, sparse_kv_indices,
            eff_batch_size, reserved_bos, reserved_eos, max_num_pages, stream);
    }
#endif
    else {
        TORCH_CHECK(false, "topk: unsupported dtype ", dtype);
    }

    const auto result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess,
                "topk kernel failed: ", ::cudaGetErrorString(result));
}
