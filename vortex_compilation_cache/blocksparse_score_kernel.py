"""
CUDA kernel replacement for blocksparseattention_9fbd0903_subgraph_0_impl.

Computes dot-product scores between averaged query vectors and centroid vectors
using a persistent-thread CUDA kernel.

Kernel semantics (matching the Triton original):
  For each workload i in [0, num_workloads):
    q_avg = mean(query[batch_idx*2 : batch_idx*2+2, :128], dim=0)   # (128,)
    for each page of 4 centroids at indices[ragged_idx + page*4]:
      output[ragged_idx + page*4 + k] = dot(q_avg, centroids[page_index + k])
"""

import torch
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// cp.async helpers (SM_80+ / Ampere+).  16-byte granularity.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void cp_async_16B(uint32_t smem_int_ptr,
                                             const void* gmem_ptr) {
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_int_ptr), "l"(gmem_ptr));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template<int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

// ---------------------------------------------------------------------------
// Persistent-thread kernel with TWO-LEVEL double buffering via cp.async:
//
//   • Outer (per-workload):  Q_i   → s_q_raw      [2×256 bf16 = 1024 B]
//   • Inner (per-page):      C_p   → s_centroids  [2×512 bf16 = 2048 B]
//
// All cp.async operations share a single FIFO-ordered group pipeline.  To
// avoid the outer Q_{i+1} group forcing an early drain inside the page loop,
// we issue Q_{i+1} from the LAST page of iter i → it becomes the youngest
// group and survives alongside the next iter's first C_0 prefetch.
//
// 128 threads / 4 warps per block.  Each warp handles one of the 4 centroid
// rows in a page during compute.
// ---------------------------------------------------------------------------
__global__ void blocksparse_score_kernel(
    const int*            __restrict__ indices,            // dense_kv_indices
    const int*            __restrict__ winfo_x_indices,    // winfo_q_indices
    const int*            __restrict__ winfo_y_offsets,    // winfo_kv_offsets
    const int*            __restrict__ winfo_y_lens,       // winfo_kv_lens
    const int*            __restrict__ winfo_num_workloads,
    const __nv_bfloat16*  __restrict__ query,              // [N, 2, 128] viewed as [N*2, 128]
    const int                          query_dim0,         // N
    const __nv_bfloat16*  __restrict__ centroids,          // [M, 128]
    const int                          centroids_dim0,     // M
    __nv_bfloat16*        __restrict__ output,             // flat score buffer
    const int                          output_dim0
) {
    const int tid     = threadIdx.x;          // 0..127
    const int warp_id = tid >> 5;             // 0..3
    const int lane_id = tid & 31;             // 0..31

    const int n_workloads = *winfo_num_workloads;
    const int pid         = blockIdx.x;
    const int num_blocks  = gridDim.x;

    // Balanced work distribution (mirrors the Triton partitioning)
    const int per   = n_workloads / num_blocks;
    const int r     = n_workloads % num_blocks;
    const int start = pid * per + min(pid, r);
    const int end   = start + per + (pid < r ? 1 : 0);

    if (start >= end) return;

    // -- Shared memory -------------------------------------------------------
    //   s_q_raw     : double-buffered raw query rows  (1024 B)
    //   s_centroids : double-buffered 4-centroid page (2048 B, [buf][row][col])
    //   s_query     : averaged query, fp32 (512 B)
    __shared__ __align__(16) __nv_bfloat16 s_q_raw     [2][256];
    __shared__ __align__(16) __nv_bfloat16 s_centroids [2][4][128];
    __shared__ float s_query[128];

    // ── Q prefetch: 512 B per workload.  32 threads × 16 B = 512 B ──────────
    auto prefetch_query = [&](int buf, int batch_idx) {
        if (tid < 32) {
            const int elem_off = tid * 8;                                       // bf16
            const __nv_bfloat16* src = query + batch_idx * 256 + elem_off;
            uint32_t dst = __cvta_generic_to_shared(&s_q_raw[buf][elem_off]);
            cp_async_16B(dst, src);
        }
    };

    // ── C prefetch: 4 rows × 128 bf16 = 1024 B.  64 threads × 16 B = 1024 B ─
    auto prefetch_centroid = [&](int buf, int page_index) {
        if (tid < 64) {
            const int elem_off = tid * 8;                                       // bf16, linear in 512-elem slab
            const __nv_bfloat16* src = centroids + page_index * 128 + elem_off;
            uint32_t dst = __cvta_generic_to_shared(
                &s_centroids[buf][0][0] + elem_off);
            cp_async_16B(dst, src);
        }
    };

    // ─── Prologue: kick off the first workload's query load ────────────────
    {
        const int batch_idx0 = winfo_x_indices[start];
        prefetch_query(0, batch_idx0);
        cp_async_commit();                        // group: Q_start
    }

    int cur_q = 0;
    for (int i = start; i < end; i++) {
        const bool has_next_iter = (i + 1 < end);
        const int  nxt_q         = cur_q ^ 1;

        // ── Wait for Q_i (the only outstanding group on entry) ────────────
        cp_async_wait_group<0>();
        __syncthreads();

        // Iter metadata (cheap scalar loads)
        const int _len       = winfo_y_lens[i];
        const int _page      = (_len + 3) >> 2;
        const int ragged_idx = winfo_y_offsets[i];

        // Build averaged query from the buffered raw rows
        float q0 = __bfloat162float(s_q_raw[cur_q][tid]);         // row 0
        float q1 = __bfloat162float(s_q_raw[cur_q][128 + tid]);   // row 1
        s_query[tid] = (q0 + q1) * 0.5f;
        __syncthreads();

        // ── Inner prologue: kick off page 0's centroid load ───────────────
        if (_page > 0) {
            const int pg0 = indices[ragged_idx];
            prefetch_centroid(0, pg0);
            cp_async_commit();                    // group: C_0
        }

        int cur_c = 0;
        for (int p = 0; p < _page; p++) {
            const bool has_next_page = (p + 1 < _page);
            const int  nxt_c         = cur_c ^ 1;

            // Issue the NEXT async load: either C_{p+1} or (on the last page)
            // Q_{i+1} for the next outer iter.  This becomes the youngest
            // group → wait_group<1> leaves it in flight during this compute.
            if (has_next_page) {
                const int pg_next = indices[ragged_idx + (p + 1) * 4];
                prefetch_centroid(nxt_c, pg_next);
                cp_async_commit();
                cp_async_wait_group<1>();         // keep C_{p+1}, wait C_p
            } else if (has_next_iter) {
                const int bidx_next = winfo_x_indices[i + 1];
                prefetch_query(nxt_q, bidx_next);
                cp_async_commit();
                cp_async_wait_group<1>();         // keep Q_{i+1}, wait C_p
            } else {
                cp_async_wait_group<0>();         // last page, last iter
            }
            __syncthreads();

            // ── Compute page p using s_centroids[cur_c] ────────────────────
            float dot = 0.0f;
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                const int col = lane_id * 4 + k;
                float c = __bfloat162float(s_centroids[cur_c][warp_id][col]);
                dot += s_query[col] * c;
            }

            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                dot += __shfl_down_sync(0xffffffff, dot, offset);
            }

            if (lane_id == 0) {
                const int item = p * 4 + warp_id;
                if (item < _len) {
                    output[ragged_idx + item] = __float2bfloat16(dot);
                }
            }

            // Block-wide barrier so no warp is still reading the centroid
            // buffer before the next iteration's cp.async rewrites it.
            __syncthreads();

            cur_c = nxt_c;
        }

        // Edge case: _page == 0 but we still owe Q_{i+1} for next iter
        if (_page == 0 && has_next_iter) {
            const int bidx_next = winfo_x_indices[i + 1];
            prefetch_query(nxt_q, bidx_next);
            cp_async_commit();
        }

        cur_q = nxt_q;
    }
}

// ---------------------------------------------------------------------------
// Thin C++ wrapper callable from Python via pybind11
// ---------------------------------------------------------------------------
void blocksparse_score_cuda(
    torch::Tensor indices,
    torch::Tensor winfo_x_indices,
    torch::Tensor winfo_y_offsets,
    torch::Tensor winfo_y_lens,
    torch::Tensor winfo_num_workloads,
    torch::Tensor query,
    torch::Tensor centroids,
    torch::Tensor output,
    int num_sms
) {
    blocksparse_score_kernel<<<num_sms, 128>>>(
        indices.data_ptr<int>(),
        winfo_x_indices.data_ptr<int>(),
        winfo_y_offsets.data_ptr<int>(),
        winfo_y_lens.data_ptr<int>(),
        winfo_num_workloads.data_ptr<int>(),
        reinterpret_cast<const __nv_bfloat16*>(query.data_ptr<at::BFloat16>()),
        static_cast<int>(query.size(0)),
        reinterpret_cast<const __nv_bfloat16*>(centroids.data_ptr<at::BFloat16>()),
        static_cast<int>(centroids.size(0)),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        static_cast<int>(output.size(0))
    );
}
"""

_CPP_SRC = r"""
void blocksparse_score_cuda(
    torch::Tensor indices,
    torch::Tensor winfo_x_indices,
    torch::Tensor winfo_y_offsets,
    torch::Tensor winfo_y_lens,
    torch::Tensor winfo_num_workloads,
    torch::Tensor query,
    torch::Tensor centroids,
    torch::Tensor output,
    int num_sms
);
"""

# JIT-compile once on first import
_module = None

def _get_module():
    global _module
    if _module is None:
        _module = load_inline(
            name="blocksparse_score_cuda_ext",
            cpp_sources=[_CPP_SRC],
            cuda_sources=[_CUDA_SRC],
            functions=["blocksparse_score_cuda"],
            verbose=False,
            extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
        )
    return _module


# Number of SMs to launch (match the Triton grid)
_NUM_SMS = 568


def blocksparse_score_impl(tensor_0, tensor_1, tensor_3, ctx):
    """
    Drop-in replacement for blocksparseattention_9fbd0903_subgraph_0_impl.

    Args:
        tensor_0: query tensor  [N, 2, 128] bf16
        tensor_1: centroids     [M, 128]    bf16
        tensor_3: output scores [T, 1, 1]   bf16
        ctx:      context with workload metadata tensors
    """
    mod = _get_module()
    mod.blocksparse_score_cuda(
        ctx.dense_kv_indices,
        ctx.winfo_q_indices,
        ctx.winfo_kv_offsets,
        ctx.winfo_kv_lens,
        ctx.winfo_num_workloads,
        tensor_0,
        tensor_1,
        tensor_3.view(-1),          # flatten [T,1,1] → [T]
        _NUM_SMS,
    )
