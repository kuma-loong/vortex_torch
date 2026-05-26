// Custom CUDA block-table MLA decode kernel (B200 / sm_100).
//
// Workload (matches vortex's decode_blocktable_mla*): per request b, gather its
// selected `seqlen` latent rows via block_table, attend H query heads against the
// fused 576-d latent (ckv=512 doubles as V, kpe=64 rope), write o[b,H,512].
//
// Design: split-KV flash decode.
//   * One CTA per (request, kv-split). The split dimension supplies the
//     parallelism that fills the 148 SMs at small batch (single-pass starves).
//   * blockDim = (32, H): warp lane along head_dim, blockDim.y = one query head.
//     All H heads live in the same CTA and read each K row from SHARED MEMORY,
//     so every latent row is read from HBM exactly once per request (no head
//     redundancy — the thing BLOCK_H fought in Triton is structurally gone).
//   * Per head the score is a 576-wide dot done as a warp shuffle reduction
//     (no tensor-core MMA: at H=16/20 the GEMV is tiny and memory-bound).
//   * Interleaved dim ownership (lane + 32*k) => coalesced, bank-conflict-free
//     shared reads and 16B (int4 = 8 bf16) vectorized HBM gathers.
//   * splits==1 writes o directly (no fp32 mid-buffer round-trip); splits>1
//     writes per-split (acc, m, d) partials reduced by a tiny stage-2 kernel.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>

#define CKV 512
#define KPE 64
#define HD  576
#define LANES 32
#define CK (CKV / LANES)   // = 16 ckv dims per lane
#define KK (KPE / LANES)   // = 2  kpe dims per lane
#define LOG2E 1.4426950408889634f

typedef __nv_bfloat16 bf16;

__device__ __forceinline__ float warpReduceSum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
  return v;
}

// ---- stage 1: per (batch, split) partial attention over a token chunk ----
template <int TILE>
__global__ void mla_stage1(
    const bf16* __restrict__ Q,        // [bs, H, 576]
    const bf16* __restrict__ Latent,   // [num_slots, 576]
    const int*  __restrict__ BlockTable,  // [bs, max_blocks]
    const int*  __restrict__ Seqlens,  // [bs]
    bf16* __restrict__ O,              // [bs, H, 512]  (used iff splits==1)
    float* __restrict__ MidAcc,        // [bs, H, splits, 512]
    float* __restrict__ MidM,          // [bs, H, splits]
    float* __restrict__ MidD,          // [bs, H, splits]
    float sm_scale, int block_size, int H, int splits, int max_blocks) {
  const int b = blockIdx.x;
  const int s = blockIdx.y;
  const int lane = threadIdx.x;       // 0..31
  const int h = threadIdx.y;          // 0..H-1
  const int nthreads = LANES * H;
  const int tid = h * LANES + lane;

  const int seqlen = Seqlens[b];
  const int chunk = (seqlen + splits - 1) / splits;
  const int kv_start = s * chunk;
  const int kv_end = min(kv_start + chunk, seqlen);

  // double-buffered K tiles: overlap the next tile's gather (cp.async) with the
  // current tile's compute (the pipelining Triton gets from num_stages=2, which
  // the v1/v2 load->sync->compute->sync loop lacked => latency-bound at low occ).
  extern __shared__ char smem[];
  bf16* Ksh = reinterpret_cast<bf16*>(smem);           // [2][TILE][HD]
  const int BUF = TILE * HD;

  // load this head's q slice into registers (interleaved ownership)
  const bf16* qrow = Q + ((long)b * H + h) * HD;
  float qc[CK], qp[KK];
#pragma unroll
  for (int k = 0; k < CK; ++k) qc[k] = __bfloat162float(qrow[lane + LANES * k]);
#pragma unroll
  for (int k = 0; k < KK; ++k) qp[k] = __bfloat162float(qrow[CKV + lane + LANES * k]);

  float m = -INFINITY, d = 0.f, acc[CK];
#pragma unroll
  for (int k = 0; k < CK; ++k) acc[k] = 0.f;

  const float scale = sm_scale * LOG2E;
  const int* bt = BlockTable + (long)b * max_blocks;
  const int ntiles = (kv_end > kv_start) ? (kv_end - kv_start + TILE - 1) / TILE : 0;

#define LOAD_TILE(t)                                                          \
  do {                                                                        \
    const int base_ = kv_start + (t) * TILE;                                  \
    const int valid_ = min(TILE, kv_end - base_);                             \
    const int nvec_ = valid_ * (HD / 8);                                      \
    bf16* dst_ = Ksh + ((t) & 1) * BUF;                                       \
    for (int v = tid; v < nvec_; v += nthreads) {                             \
      const int j = v / (HD / 8);                                            \
      const int dd = (v % (HD / 8)) * 8;                                     \
      const int n = base_ + j;                                               \
      const int page = bt[n / block_size];                                   \
      __pipeline_memcpy_async(                                                \
          dst_ + j * HD + dd,                                                 \
          Latent + (long)(page * block_size + (n % block_size)) * HD + dd, 16); \
    }                                                                         \
    __pipeline_commit();                                                      \
  } while (0)

  if (ntiles > 0) LOAD_TILE(0);
  for (int t = 0; t < ntiles; ++t) {
    const int base = kv_start + t * TILE;
    const int valid = min(TILE, kv_end - base);
    if (t + 1 < ntiles) { LOAD_TILE(t + 1); __pipeline_wait_prior(1); }
    else __pipeline_wait_prior(0);
    __syncthreads();
    const bf16* Kt = Ksh + (t & 1) * BUF;

    // --- phase 1: T independent score reductions (pipeline shuffle latency) ---
    float score[TILE];
#pragma unroll
    for (int j = 0; j < TILE; ++j) {
      if (j >= valid) break;
      const bf16* krow = Kt + j * HD;
      float partial = 0.f;
#pragma unroll
      for (int k = 0; k < CK; ++k) partial += qc[k] * __bfloat162float(krow[lane + LANES * k]);
#pragma unroll
      for (int k = 0; k < KK; ++k) partial += qp[k] * __bfloat162float(krow[CKV + lane + LANES * k]);
      score[j] = warpReduceSum(partial) * scale;
    }
    // --- phase 2: one softmax recurrence step for the whole tile ---
    float m_tile = -INFINITY;
#pragma unroll
    for (int j = 0; j < TILE; ++j) if (j < valid) m_tile = fmaxf(m_tile, score[j]);
    float m_new = fmaxf(m, m_tile);
    float alpha = exp2f(m - m_new);
#pragma unroll
    for (int k = 0; k < CK; ++k) acc[k] *= alpha;
    d *= alpha;
#pragma unroll
    for (int j = 0; j < TILE; ++j) {
      if (j >= valid) break;
      const bf16* krow = Kt + j * HD;
      float p = exp2f(score[j] - m_new);
      d += p;
#pragma unroll
      for (int k = 0; k < CK; ++k) acc[k] += p * __bfloat162float(krow[lane + LANES * k]);
    }
    m = m_new;
    __syncthreads();
  }
#undef LOAD_TILE

  if (kv_end <= kv_start) {            // empty split: emit neutral element
    if (splits > 1) {
      MidM[((long)b * H + h) * splits + s] = -INFINITY;
      MidD[((long)b * H + h) * splits + s] = 0.f;
    }
    return;
  }

  if (splits == 1) {
    const float inv = 1.f / d;
    bf16* orow = O + ((long)b * H + h) * CKV;
#pragma unroll
    for (int k = 0; k < CK; ++k) orow[lane + LANES * k] = __float2bfloat16(acc[k] * inv);
  } else {
    float* arow = MidAcc + (((long)b * H + h) * splits + s) * CKV;
#pragma unroll
    for (int k = 0; k < CK; ++k) arow[lane + LANES * k] = acc[k];   // UNnormalized
    if (lane == 0) {
      MidM[((long)b * H + h) * splits + s] = m;
      MidD[((long)b * H + h) * splits + s] = d;
    }
  }
}

// ---- stage 2: merge per-split partials (one warp per (batch, head)) ----
__global__ void mla_stage2(
    const float* __restrict__ MidAcc, const float* __restrict__ MidM,
    const float* __restrict__ MidD, bf16* __restrict__ O,
    int H, int splits) {
  const int b = blockIdx.x;
  const int h = blockIdx.y;
  const int lane = threadIdx.x;        // 0..31
  const long hbase = (long)b * H + h;

  float M = -INFINITY;
#pragma unroll 1
  for (int s = 0; s < splits; ++s) M = fmaxf(M, MidM[hbase * splits + s]);

  float acc[CK], d = 0.f;
#pragma unroll
  for (int k = 0; k < CK; ++k) acc[k] = 0.f;

  for (int s = 0; s < splits; ++s) {
    float ms = MidM[hbase * splits + s];
    if (ms == -INFINITY) continue;
    float w = exp2f(ms - M);
    d += MidD[hbase * splits + s] * w;
    const float* arow = MidAcc + (hbase * splits + s) * CKV;
#pragma unroll
    for (int k = 0; k < CK; ++k) acc[k] += arow[lane + LANES * k] * w;
  }
  const float inv = 1.f / d;
  bf16* orow = O + hbase * CKV;
#pragma unroll
  for (int k = 0; k < CK; ++k) orow[lane + LANES * k] = __float2bfloat16(acc[k] * inv);
}

// host launchers ----------------------------------------------------------
static int g_sm = 0;

void mla_decode(torch::Tensor q, torch::Tensor latent, torch::Tensor block_table,
                torch::Tensor seqlens, torch::Tensor o, double sm_scale,
                int64_t block_size, int64_t splits) {
  const int bs = q.size(0), H = q.size(1);
  const int max_blocks = block_table.size(1);
  constexpr int TILE = 16;
  dim3 grid(bs, splits), block(LANES, H);
  size_t smem = (size_t)2 * TILE * HD * sizeof(bf16);   // double-buffered
  auto stream = at::cuda::getCurrentCUDAStream();

  if (splits == 1) {
    mla_stage1<TILE><<<grid, block, smem, stream>>>(
        (const bf16*)q.data_ptr(), (const bf16*)latent.data_ptr(),
        (const int*)block_table.data_ptr(), (const int*)seqlens.data_ptr(),
        (bf16*)o.data_ptr(), nullptr, nullptr, nullptr,
        (float)sm_scale, (int)block_size, H, 1, max_blocks);
  } else {
    auto opt = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());
    auto mid_acc = torch::empty({bs, H, (long)splits, CKV}, opt);
    auto mid_m = torch::empty({bs, H, (long)splits}, opt);
    auto mid_d = torch::empty({bs, H, (long)splits}, opt);
    mla_stage1<TILE><<<grid, block, smem, stream>>>(
        (const bf16*)q.data_ptr(), (const bf16*)latent.data_ptr(),
        (const int*)block_table.data_ptr(), (const int*)seqlens.data_ptr(),
        nullptr, mid_acc.data_ptr<float>(), mid_m.data_ptr<float>(),
        mid_d.data_ptr<float>(), (float)sm_scale, (int)block_size, H, (int)splits, max_blocks);
    mla_stage2<<<dim3(bs, H), dim3(LANES), 0, stream>>>(
        mid_acc.data_ptr<float>(), mid_m.data_ptr<float>(), mid_d.data_ptr<float>(),
        (bf16*)o.data_ptr(), H, (int)splits);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mla_decode", &mla_decode, "block-table MLA decode (split-KV)");
}
