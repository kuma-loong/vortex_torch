/**
 * @NOTE: This file is adapted from
 * https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_v32/topk_selector.py
 * We:
 * 1. adapt from tilelang to pure cuda
 * 2. optimize the performance a little
 * 3. fix the potential illegal memory access
 */
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
 #include <optional>

 namespace {

 constexpr int kThreadsPerBlock = 1024;

 #ifdef USE_ROCM
 // On ROCm, the per-workgroup LDS budget depends on the target arch, so we inject a
 // per-arch value from `setup_rocm.py` via `-DSGL_TOPK_DYNAMIC_SMEM_BYTES=...`.
 #ifdef SGL_TOPK_DYNAMIC_SMEM_BYTES
 constexpr size_t kSmem = static_cast<size_t>(SGL_TOPK_DYNAMIC_SMEM_BYTES);
 #else
 constexpr size_t kSmem = 48 * 1024;  // bytes
 #endif
 #else
 // Reduced from 128KB to 32KB to improve occupancy.
 // Each radix pass needs at most ~TopK candidates in the threshold bin,
 // so 4K entries per round (2 rounds = 8K entries = 32KB) is sufficient.
 constexpr size_t kSmem = 8 * 1024 * sizeof(uint32_t);  // 32KB (bytes)
 #endif

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
 #ifdef USE_ROCM
     // hipify will turn cudaFuncSetAttribute -> hipFuncSetAttribute. On ROCm,
     // hipFuncSetAttribute expects `const void*` and hipcc does not accept passing
     // a function pointer directly, so cast explicitly.
     return ::cudaFuncSetAttribute(
         reinterpret_cast<const void*>(f), ::cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_smem);
 #else
     // CUDA: keep original behavior (no cast needed).
     return ::cudaFuncSetAttribute(f, ::cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_smem);
 #endif
   }();
   TORCH_CHECK(result == cudaSuccess, "set_up_kernel_once failed:", ::cudaGetErrorString(result));
 }

 // ======================================================================
 // Vortex integration: BOS/EOS-aware segmented TopK with index remapping
 // ======================================================================

 template <typename T>
 __device__ __forceinline__ float vortex_to_float(T x);

 template <>
 __device__ __forceinline__ float vortex_to_float<float>(float x) { return x; }

 template <>
 __device__ __forceinline__ float vortex_to_float<__nv_bfloat16>(__nv_bfloat16 x) {
     return __bfloat162float(x);
 }

 constexpr int VORTEX_MAX_TOPK = 2048;

 // Templated version of fast_topk_cuda_tl:
 //   - ScoreT: float or __nv_bfloat16
 //   - target_k: runtime parameter (replaces compile-time TopK)
 template <typename ScoreT>
 __device__ void fast_topk_vortex(
     const ScoreT* __restrict__ input,
     int*          __restrict__ index,
     int           row_start,
     int           length,
     int           target_k)
 {
     int topk = target_k;
     constexpr auto BLOCK_SIZE = 1024;
     constexpr auto RADIX = 256;
     constexpr auto SMEM_INPUT_SIZE = kSmem / (2 * sizeof(int));

     alignas(128) __shared__ int vh_histogram_buf[2][RADIX + 128];
     alignas(128) __shared__ int vh_counter;
     alignas(128) __shared__ int vh_threshold_bin_id;
     alignas(128) __shared__ int vh_num_input[2];

     auto& vh_histogram = vh_histogram_buf[0];
     extern __shared__ int vh_input_idx[][SMEM_INPUT_SIZE];

     const int tx = threadIdx.x;

     // Stage 1: 8-bit coarse histogram
     if (tx < RADIX + 1) vh_histogram[tx] = 0;
     __syncthreads();

     for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
         const auto bin = convert_to_uint8(vortex_to_float(input[idx + row_start]));
         ::atomicAdd(&vh_histogram[bin], 1);
     }
     __syncthreads();

     const auto run_cumsum = [&] {
 #pragma unroll 8
         for (int i = 0; i < 8; ++i) {
             static_assert(1 << 8 == RADIX);
             if (C10_LIKELY(tx < RADIX)) {
                 const auto j = 1 << i;
                 const auto k = i & 1;
                 auto value = vh_histogram_buf[k][tx];
                 if (tx < RADIX - j) {
                     value += vh_histogram_buf[k][tx + j];
                 }
                 vh_histogram_buf[k ^ 1][tx] = value;
             }
             __syncthreads();
         }
     };

     run_cumsum();
     if (tx < RADIX && vh_histogram[tx] > topk && vh_histogram[tx + 1] <= topk) {
         vh_threshold_bin_id = tx;
         vh_num_input[0] = 0;
         vh_counter = 0;
     }
     __syncthreads();

     const auto threshold_bin = vh_threshold_bin_id;
     topk -= vh_histogram[threshold_bin + 1];

     if (topk == 0) {
         for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
             const auto bin = static_cast<int>(
                 convert_to_uint8(vortex_to_float(input[idx + row_start])));
             if (bin > threshold_bin) {
                 const auto pos = ::atomicAdd(&vh_counter, 1);
                 index[pos] = idx;
             }
         }
         __syncthreads();
         return;
     } else {
         __syncthreads();
         if (tx < RADIX + 1) vh_histogram[tx] = 0;
         __syncthreads();

         for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
             const auto raw_input = vortex_to_float(input[idx + row_start]);
             const auto bin = static_cast<int>(convert_to_uint8(raw_input));
             if (bin > threshold_bin) {
                 const auto pos = ::atomicAdd(&vh_counter, 1);
                 index[pos] = idx;
             } else if (bin == threshold_bin) {
                 const auto pos = ::atomicAdd(&vh_num_input[0], 1);
                 if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
                     vh_input_idx[0][pos] = idx;
                     const auto b32 = convert_to_uint32(raw_input);
                     const auto sub_bin = (b32 >> 24) & 0xFF;
                     ::atomicAdd(&vh_histogram[sub_bin], 1);
                 }
             }
         }
         __syncthreads();
     }

     // Stage 2: refine with 8-bit radix passes
 #pragma unroll 4
     for (int round = 0; round < 4; ++round) {
         __shared__ int vh_last_remain;
         const auto r_idx = round % 2;

         const auto _raw_num_input = vh_num_input[r_idx];
         const auto num_input = (_raw_num_input < int(SMEM_INPUT_SIZE))
                                    ? _raw_num_input
                                    : int(SMEM_INPUT_SIZE);

         run_cumsum();
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
                     vortex_to_float(input[idx + row_start])) >> offset) & 0xFF;
                 if (bin > threshold_bin) {
                     const auto pos = ::atomicAdd(&vh_counter, 1);
                     index[pos] = idx;
                 }
             }
             __syncthreads();
             break;
         } else {
             __syncthreads();
             if (tx < RADIX + 1) vh_histogram[tx] = 0;
             __syncthreads();
             for (int i = tx; i < num_input; i += BLOCK_SIZE) {
                 const auto idx = vh_input_idx[r_idx][i];
                 const auto raw_input = vortex_to_float(input[idx + row_start]);
                 const auto offset = 24 - round * 8;
                 const auto bin = (convert_to_uint32(raw_input) >> offset) & 0xFF;
                 if (bin > threshold_bin) {
                     const auto pos = ::atomicAdd(&vh_counter, 1);
                     index[pos] = idx;
                 } else if (bin == threshold_bin) {
                     if (round == 3) {
                         const auto pos = ::atomicAdd(&vh_last_remain, -1);
                         if (pos > 0) {
                             index[target_k - pos] = idx;
                         }
                     } else {
                         const auto pos = ::atomicAdd(&vh_num_input[r_idx ^ 1], 1);
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

 // Wrapper kernel: one CUDA block per batch*head segment
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
     int*          __restrict__ out_blk   = sparse_kv_indices
                                          + sparse_kv_indptr[bx]
                                          + page_reserved_bos;

     __shared__ int s_indices[VORTEX_MAX_TOPK];
     fast_topk_vortex<ScoreT>(score_blk, s_indices, 0, nblk, topk_val);
     __syncthreads();

     // Remap position indices -> page indices via dense_kv_indices
     const int tx = threadIdx.x;
     for (int i = tx; i < topk_val; i += kThreadsPerBlock) {
         out_blk[i] = idx_blk[s_indices[i]];
     }
 }

 }  // namespace

 // ======================================================================
 // Vortex host entry point — same interface as topk_output in topk.cu
 // ======================================================================
 void topk_output_v2(
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
         TORCH_CHECK(false,
                     "topk_output: unsupported dtype ",
                     x.scalar_type());
     }

     const auto result = cudaGetLastError();
     TORCH_CHECK(result == cudaSuccess,
                 "topk_output kernel failed: ", ::cudaGetErrorString(result));
 }
