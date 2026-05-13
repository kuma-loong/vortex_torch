"""JIT-compiled CUDA backend for the sglang prefill planner and the two
chunkwise NH/HN transposes.

Mirrors the layout of :mod:`vortex_torch.indexer.planner_sglang` but
bundles three entry points — ``sglang_plan_prefill``,
``Chunkwise_NH2HN_Transpose`` and ``Chunkwise_HN2NH_Transpose`` — into
a single ``load_inline`` module so the (de)compile cost is paid once.

The CUDA source is byte-equivalent to the prefill / transpose region of
``csrc/utils_sglang.cu``; the only differences are that the includes
are inlined here (since ``register.h`` lives in the precompiled
extension) and the file omits unrelated kernels (decode planner, FA3
variants, etc.).
"""
import torch
from torch.utils.cpp_extension import load_inline


CPP_SOURCE = r"""
#include <torch/extension.h>
#include <tuple>

void sglang_plan_prefill(
    const at::Tensor&  cached_seq_lens,
    at::Tensor&        dense_kv_indptr,
    at::Tensor&        dense_kv_indices,
    const at::Tensor&  input_seq_lens,
    at::Tensor&        qo_indptr_ragged,
    at::Tensor&        qo_indptr_paged,
    at::Tensor&        kv_last_page_len,
    const at::Tensor&  req_to_token,
    const at::Tensor&  req_indices,
    at::Tensor&        batch_table,
    const int64_t      page_size,
    const int64_t      num_kv_heads
);

at::Tensor Chunkwise_NH2HN_Transpose(
    const at::Tensor&   x,
    const at::Tensor&   indptr,
    const at::Tensor&   batch_table,
    const int64_t       num_qo_heads,
    const int64_t       num_kv_heads,
    const int64_t       head_dim
);

std::tuple<at::Tensor, at::Tensor> Chunkwise_HN2NH_Transpose(
    const at::Tensor&   x,
    const at::Tensor&   y,
    const at::Tensor&   indptr,
    const at::Tensor&   batch_table,
    const int64_t       num_qo_heads,
    const int64_t       num_kv_heads,
    const int64_t       head_dim
);
"""


CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cub/cub.cuh>
#include <stdint.h>
#include <tuple>


__global__ void Sgl_Prefill_Plan_Indptr_Kernel(
    const int*  __restrict__ cached_seq_lens,
    const int*  __restrict__ input_seq_lens,
    int*        __restrict__ kv_indptr,
    int*        __restrict__ qo_indptr_ragged,
    int*        __restrict__ qo_indptr_paged,
    const int   batch_size,
    const int   num_kv_heads,
    const int   page_size
) {
    const int tx = threadIdx.x;
    using BlockScan = cub::BlockScan<int, 1024>;
    __shared__ typename BlockScan::TempStorage temp_storage;
    int input_seq_cumsum = (tx < batch_size) ? input_seq_lens[tx] : 0;
    int num_cached_pages = (tx < batch_size) ? ((cached_seq_lens[tx] + page_size - 1) / page_size) : 0;
    int cached_seq_cumsum = (tx < batch_size) ? num_cached_pages : 0;

    BlockScan(temp_storage).InclusiveSum(input_seq_cumsum, input_seq_cumsum);
    __syncthreads();
    BlockScan(temp_storage).InclusiveSum(cached_seq_cumsum, cached_seq_cumsum);
    __syncthreads();

    if (tx < batch_size) {
        qo_indptr_ragged[tx + 1] = input_seq_cumsum;
        #pragma unroll
        for (int i = 0; i < num_kv_heads; ++i) {
            qo_indptr_paged[num_kv_heads * (tx + 1) - i] =
                input_seq_cumsum * num_kv_heads - i * input_seq_lens[tx];
            kv_indptr[num_kv_heads * (tx + 1) - i] =
                cached_seq_cumsum * num_kv_heads - i * num_cached_pages;
        }
    }

    if (tx == 0) {
        qo_indptr_ragged[0] = 0;
        qo_indptr_paged[0] = 0;
        kv_indptr[0] = 0;
    }
}


__global__ void Sgl_Prefill_Plan_IndicesBatchTable_Kernel(
    const int*   __restrict__ req_to_token,
    const long*  __restrict__ req_indices,
    const int*   __restrict__ cache_seq_lens,
    const int*   __restrict__ input_seq_lens,
    const int*   __restrict__ kv_indptr,
    const int*   __restrict__ qo_indptr_ragged,
    const int    page_size,
    const int    num_kv_heads,
    const int    req_to_token_stride,
    int*         __restrict__ kv_last_page_len,
    int*         __restrict__ kv_indices,
    uint16_t*    __restrict__ batch_table
) {
    const int block_size = blockDim.x;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int* token_indices = req_to_token + req_indices[bx] * req_to_token_stride;
    const int num_cached_pages = (cache_seq_lens[bx] + page_size - 1) / page_size;
    int* output = kv_indices + kv_indptr[bx * num_kv_heads + by];
    const int last_len = cache_seq_lens[bx] % page_size;
    kv_last_page_len[bx * num_kv_heads + by] = (last_len == 0) ? page_size : last_len;

    int pos = tx;
    while (pos < num_cached_pages) {
        int data = token_indices[pos * page_size];
        output[pos] = (data / page_size) * (num_kv_heads) + by;
        pos += block_size;
    }

    const int qo_len = input_seq_lens[bx];
    uint16_t* batch_table_output = batch_table + qo_indptr_ragged[bx];
    pos = tx;
    if (by == 0) {
        while (pos < qo_len) {
            batch_table_output[pos] = static_cast<uint16_t>(bx);
            pos += block_size;
        }
    }
}


void sglang_plan_prefill(
    const at::Tensor&  cached_seq_lens,
    at::Tensor&        dense_kv_indptr,
    at::Tensor&        dense_kv_indices,
    const at::Tensor&  input_seq_lens,
    at::Tensor&        qo_indptr_ragged,
    at::Tensor&        qo_indptr_paged,
    at::Tensor&        kv_last_page_len,
    const at::Tensor&  req_to_token,
    const at::Tensor&  req_indices,
    at::Tensor&        batch_table,
    const int64_t      page_size,
    const int64_t      num_kv_heads
) {
    const int batch_size = cached_seq_lens.size(0);
    const int req_to_token_stride = req_to_token.size(1);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    Sgl_Prefill_Plan_Indptr_Kernel<<<1, 1024, 0, stream>>>(
        cached_seq_lens.data_ptr<int>(),
        input_seq_lens.data_ptr<int>(),
        dense_kv_indptr.data_ptr<int>(),
        qo_indptr_ragged.data_ptr<int>(),
        qo_indptr_paged.data_ptr<int>(),
        batch_size,
        static_cast<int>(num_kv_heads),
        static_cast<int>(page_size)
    );

    dim3 nblks(batch_size, static_cast<unsigned int>(num_kv_heads));
    dim3 nthrs(128);
    Sgl_Prefill_Plan_IndicesBatchTable_Kernel<<<nblks, nthrs, 0, stream>>>(
        req_to_token.data_ptr<int>(),
        req_indices.data_ptr<long>(),
        cached_seq_lens.data_ptr<int>(),
        input_seq_lens.data_ptr<int>(),
        dense_kv_indptr.data_ptr<int>(),
        qo_indptr_ragged.data_ptr<int>(),
        static_cast<int>(page_size),
        static_cast<int>(num_kv_heads),
        req_to_token_stride,
        kv_last_page_len.data_ptr<int>(),
        dense_kv_indices.data_ptr<int>(),
        batch_table.data_ptr<uint16_t>()
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


__global__ void Chunkwise_NH2HN_Transpose_Kernel(
    const __nv_bfloat16*  __restrict__ x,
    const int*            __restrict__ indptr,
    const uint16_t*       __restrict__ batch_table,
    const int             num_kv_heads,
    const int             group_size,
    const int             head_dim,
    __nv_bfloat16*        __restrict__ output
) {
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;
    const int tx = threadIdx.x;
    const int batch_idx = static_cast<int>(batch_table[bx]);

    const int batch_offset = indptr[batch_idx];
    const int batch_q_len = indptr[batch_idx + 1] - indptr[batch_idx];
    const int token_offset = bx - indptr[batch_idx];

    const __nv_bfloat16* src = x + bx * num_kv_heads * group_size * head_dim +
                               by * group_size * head_dim + bz * head_dim;

    __nv_bfloat16* dst = output + batch_offset * num_kv_heads * group_size * head_dim +
                         by * batch_q_len * group_size * head_dim +
                         token_offset * group_size * head_dim +
                         bz * head_dim;

    constexpr int B16_PER_FLOAT2 = 4;
    const int nvec = head_dim / B16_PER_FLOAT2;

    const float2* __restrict__ src2 = reinterpret_cast<const float2*>(src);
    float2* __restrict__ dst2       = reinterpret_cast<float2*>(dst);

    for (int i = tx; i < nvec; i += blockDim.x) {
        dst2[i] = src2[i];
    }
}


at::Tensor Chunkwise_NH2HN_Transpose(
    const at::Tensor&   x,
    const at::Tensor&   indptr,
    const at::Tensor&   batch_table,
    const int64_t       num_qo_heads,
    const int64_t       num_kv_heads,
    const int64_t       head_dim
) {
    const int x_len = x.size(0);
    const int group_size = num_qo_heads / num_kv_heads;
    at::Tensor output = torch::empty(
        {x_len * num_kv_heads, group_size, head_dim}, x.options()
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    dim3 nblks(x_len, static_cast<unsigned int>(num_kv_heads),
               static_cast<unsigned int>(group_size));
    dim3 nthrs(static_cast<unsigned int>(head_dim / 4));
    Chunkwise_NH2HN_Transpose_Kernel<<<nblks, nthrs, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        indptr.data_ptr<int>(),
        batch_table.data_ptr<uint16_t>(),
        static_cast<int>(num_kv_heads),
        group_size,
        static_cast<int>(head_dim),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>())
    );

    return output;
}


__global__ void Chunkwise_HN2NH_Transpose_Kernel(
    const __nv_bfloat16* __restrict__ x,
    const float*         __restrict__ y,
    const int*           __restrict__ indptr,
    const uint16_t*      __restrict__ batch_table,
    const int            num_kv_heads,
    const int            group_size,
    const int            head_dim,
    __nv_bfloat16*       __restrict__ x_output,
    float*               __restrict__ y_output
) {
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;
    const int tx = threadIdx.x;

    const int batch_idx = static_cast<int>(batch_table[bx]);
    const int batch_offset = indptr[batch_idx];
    const int batch_q_len = indptr[batch_idx + 1] - indptr[batch_idx];
    const int token_offset = bx - indptr[batch_idx];

    const __nv_bfloat16* src_x = x + batch_offset * num_kv_heads * group_size * head_dim +
                                 by * batch_q_len * group_size * head_dim +
                                 token_offset * group_size * head_dim +
                                 bz * head_dim;

    __nv_bfloat16* dst_x = x_output + bx * num_kv_heads * group_size * head_dim +
                           by * group_size * head_dim + bz * head_dim;

    constexpr int B16_PER_FLOAT2 = 4;
    const int nvec = head_dim / B16_PER_FLOAT2;

    const float2* __restrict__ src2_x = reinterpret_cast<const float2*>(src_x);
    float2* __restrict__ dst2_x       = reinterpret_cast<float2*>(dst_x);

    for (int i = tx; i < nvec; i += blockDim.x) {
        dst2_x[i] = src2_x[i];
    }

    const float* src_y = y + batch_offset * num_kv_heads * group_size +
                         by * batch_q_len * group_size + token_offset * group_size + bz;
    float* dst_y = y_output + bx * num_kv_heads * group_size +
                   by * group_size + bz;

    if (tx == 0) {
        *dst_y = *src_y;
    }
}


std::tuple<at::Tensor, at::Tensor> Chunkwise_HN2NH_Transpose(
    const at::Tensor&   x,
    const at::Tensor&   y,
    const at::Tensor&   indptr,
    const at::Tensor&   batch_table,
    const int64_t       num_qo_heads,
    const int64_t       num_kv_heads,
    const int64_t       head_dim
) {
    const int x_len = x.size(0) / num_kv_heads;
    const int group_size = x.size(1);

    at::Tensor x_output = torch::empty(
        {x_len, num_qo_heads, head_dim}, x.options()
    );
    at::Tensor y_output = torch::empty(
        {x_len, num_qo_heads}, y.options()
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    dim3 nblks(x_len, static_cast<unsigned int>(num_kv_heads),
               static_cast<unsigned int>(group_size));
    dim3 nthrs(static_cast<unsigned int>(head_dim / 4));
    Chunkwise_HN2NH_Transpose_Kernel<<<nblks, nthrs, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        y.data_ptr<float>(),
        indptr.data_ptr<int>(),
        batch_table.data_ptr<uint16_t>(),
        static_cast<int>(num_kv_heads),
        group_size,
        static_cast<int>(head_dim),
        reinterpret_cast<__nv_bfloat16*>(x_output.data_ptr<at::BFloat16>()),
        y_output.data_ptr<float>()
    );

    return std::make_tuple(x_output, y_output);
}
"""


_MODULE = None


def get_sglang_prefill_module(verbose: bool = False):
    """Compile (once) and return the load-inline module exposing
    ``sglang_plan_prefill``, ``Chunkwise_NH2HN_Transpose`` and
    ``Chunkwise_HN2NH_Transpose``."""
    global _MODULE
    if _MODULE is not None:
        return _MODULE

    _MODULE = load_inline(
        name="sglang_prefill_ext",
        cpp_sources=CPP_SOURCE,
        cuda_sources=CUDA_SOURCE,
        functions=[
            "sglang_plan_prefill",
            "Chunkwise_NH2HN_Transpose",
            "Chunkwise_HN2NH_Transpose",
        ],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
        ],
        with_cuda=True,
        verbose=verbose,
    )
    return _MODULE
