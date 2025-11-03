#include "register.h"
#include <cub/cub.cuh>


template <int NUM_THREADS, int ITEM_PER_THREAD>
__global__ void TopKOutput_F32_Kernel(
const float* __restrict__ score,
const int*   __restrict__ dense_kv_indptr,
const int*   __restrict__ sparse_kv_indptr,
const int*   __restrict__ dense_kv_indices,
int*         __restrict__ sparse_kv_indices,
const int    topk_val,
const int    page_reserved_bos,
const int    page_reserved_eos)
{
     const int bx = blockIdx.x;
    const int tx = threadIdx.x;

    const int start = dense_kv_indptr[bx] + page_reserved_bos;
    const int end   = dense_kv_indptr[bx + 1] - page_reserved_eos;
    const int nblk  = end - start;
    if (nblk <= topk_val) return;

    const float* __restrict__ score_blk = score + start;
    const int*   __restrict__ idx_blk   = dense_kv_indices + start;
    int*         __restrict__ out_blk   = sparse_kv_indices + sparse_kv_indptr[bx] + page_reserved_bos;

    float key[ITEM_PER_THREAD];
    int   val[ITEM_PER_THREAD];

    using BLF  = cub::BlockLoad<float, NUM_THREADS, ITEM_PER_THREAD, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BLI  = cub::BlockLoad<int,   NUM_THREADS, ITEM_PER_THREAD, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BSI  = cub::BlockStore<int,  NUM_THREADS, ITEM_PER_THREAD, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using Sort = cub::BlockRadixSort<float, NUM_THREADS, ITEM_PER_THREAD, int>;

    __shared__ union {
        typename BLF::TempStorage  lf;
        typename BLI::TempStorage  li;
        typename BSI::TempStorage  si;
        typename Sort::TempStorage sort;
    } temp;

    BLF(temp.lf).Load(score_blk, key, nblk, -INFINITY);
    __syncthreads();
    BLI(temp.li).Load(idx_blk,   val, nblk, 0);
    __syncthreads();

    Sort(temp.sort).SortDescending(key, val);
    __syncthreads();

    const int valid_out = min(topk_val, nblk);
    BSI(temp.si).Store(out_blk, /*per-thread regs*/ val, valid_out);
}


void topk_output(
const at::Tensor& x,
const at::Tensor& dense_kv_indptr,
const at::Tensor& sparse_kv_indptr,
const at::Tensor& dense_kv_indices,
at::Tensor&       sparse_kv_indices,
const int64_t     eff_batch_size,
const int64_t     topk_val,
const int64_t     reserved_bos,
const int64_t     reserved_eos,
const int64_t     max_num_pages
){


    dim3 nblks(eff_batch_size);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    if (max_num_pages <= 128){
        TopKOutput_F32_Kernel<128, 1><<<nblks, 128, 0, stream>>>(
            x.data_ptr<float>(),
            dense_kv_indptr.data_ptr<int>(),
            sparse_kv_indptr.data_ptr<int>(),
            dense_kv_indices.data_ptr<int>(),
            sparse_kv_indices.data_ptr<int>(),
            topk_val,
            reserved_bos,
            reserved_eos
        );
    } else if (max_num_pages <= 256){
        TopKOutput_F32_Kernel<128, 2><<<nblks, 128, 0, stream>>>(
            x.data_ptr<float>(),
            dense_kv_indptr.data_ptr<int>(),
            sparse_kv_indptr.data_ptr<int>(),
            dense_kv_indices.data_ptr<int>(),
            sparse_kv_indices.data_ptr<int>(),
            topk_val,
            reserved_bos,
            reserved_eos
        );
    } else if (max_num_pages <= 512){
        TopKOutput_F32_Kernel<128, 4><<<nblks, 128, 0, stream>>>(
            x.data_ptr<float>(),
            dense_kv_indptr.data_ptr<int>(),
            sparse_kv_indptr.data_ptr<int>(),
            dense_kv_indices.data_ptr<int>(),
            sparse_kv_indices.data_ptr<int>(),
            topk_val,
            reserved_bos,
            reserved_eos
        );
    } else if (max_num_pages <= 1024){
        TopKOutput_F32_Kernel<256, 4><<<nblks, 256, 0, stream>>>(
            x.data_ptr<float>(),
            dense_kv_indptr.data_ptr<int>(),
            sparse_kv_indptr.data_ptr<int>(),
            dense_kv_indices.data_ptr<int>(),
            sparse_kv_indices.data_ptr<int>(),
            topk_val,
            reserved_bos,
            reserved_eos
        );
    } else if (max_num_pages <= 2048){
        TopKOutput_F32_Kernel<256, 8><<<nblks, 256, 0, stream>>>(
            x.data_ptr<float>(),
            dense_kv_indptr.data_ptr<int>(),
            sparse_kv_indptr.data_ptr<int>(),
            dense_kv_indices.data_ptr<int>(),
            sparse_kv_indices.data_ptr<int>(),
            topk_val,
            reserved_bos,
            reserved_eos
        );
    } else if (max_num_pages <= 4096){
        TopKOutput_F32_Kernel<512, 8><<<nblks, 512, 0, stream>>>(
            x.data_ptr<float>(),
            dense_kv_indptr.data_ptr<int>(),
            sparse_kv_indptr.data_ptr<int>(),
            dense_kv_indices.data_ptr<int>(),
            sparse_kv_indices.data_ptr<int>(),
            topk_val,
            reserved_bos,
            reserved_eos
        );
    } else {
        TORCH_CHECK(false);
    }

}
