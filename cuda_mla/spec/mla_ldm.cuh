// ldmatrix + mma.sync.m16n16k16 MLA decode (replaces manual scattered fragment
// loads). Recipe per flashinfer prefill: GEMM1 loads K NON-transposed (K stored
// [token][dim] is already col-major B for row.col mma), GEMM2 loads V via
// ldmatrix.trans. Register-O, cp.async K, softmax in smem, split-KV.
#pragma once
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>

namespace ldm {
typedef __nv_bfloat16 bf16_t;
constexpr int HD = 576, CKV = 512;
constexpr int HDP = 584;   // padded smem row stride (576+8): 16-way bank conflict -> 2-way.
constexpr float LOG2E = 1.4426950408889634f;
// NOTE: a full XOR swizzle (conflict-free) was implemented + verified correct but
// measured SLOWER than this cheap pad (1675 vs 1898): post-register-softmax the
// kernel isn't bank-conflict-bound, and the swizzle's hot-loop address XOR costs
// more than the residual 2-way conflict. Kept the pad.

__device__ __forceinline__ void ldm_x4(uint32_t* R, const bf16_t* p) {
  uint32_t s = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(R[0]), "=r"(R[1]), "=r"(R[2]), "=r"(R[3]) : "r"(s));
}
__device__ __forceinline__ void ldm_x4_trans(uint32_t* R, const bf16_t* p) {
  uint32_t s = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.trans.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(R[0]), "=r"(R[1]), "=r"(R[2]), "=r"(R[3]) : "r"(s));
}
// m16n16k16 = two m16n8k16; C[8], A[4], B[4] (B[0,1]=n0-7, B[2,3]=n8-15)
__device__ __forceinline__ void mma16(float* C, const uint32_t* A, const uint32_t* B) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
               : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
               : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
                 "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
               : "=f"(C[4]), "=f"(C[5]), "=f"(C[6]), "=f"(C[7])
               : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]),
                 "f"(C[4]), "f"(C[5]), "f"(C[6]), "f"(C[7]));
}

template <int MTILES, int BLK, int NT, int NWARPS, int STAGES, int NACC = 1>
__global__ void stage1(const bf16_t* __restrict__ Q, const bf16_t* __restrict__ Latent,
                       const int* __restrict__ BlockTable, const int* __restrict__ Seqlens,
                       bf16_t* __restrict__ O, float* __restrict__ MidO, float* __restrict__ MidM,
                       float* __restrict__ MidL, float sm_scale, int H, int splits, int max_blocks) {
  const int b = blockIdx.x, sp = blockIdx.y;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, tid = threadIdx.x;
  const int nthreads = NWARPS * 32;
  constexpr int M = MTILES * 16;
  constexpr int ND = (CKV / 16) / NWARPS;   // dim n16-tiles per warp (32/NWARPS)
  const int seqlen = Seqlens[b];
  const int chunk = (seqlen + splits - 1) / splits;
  const int kv_start = sp * chunk, kv_end = min(kv_start + chunk, seqlen);

  extern __shared__ char smem[];
  bf16_t* Ksh = reinterpret_cast<bf16_t*>(smem);
  const int KBUF = NT * HDP;
  bf16_t* Qsh = Ksh + STAGES * KBUF;
  bf16_t* Psh = reinterpret_cast<bf16_t*>(Qsh + M * HDP);   // register softmax: no Ssh
  float* msh = reinterpret_cast<float*>(Psh + M * NT);
  float* lsh = msh + M, *ash = lsh + M;

  float Oreg[MTILES][ND][8];
#pragma unroll
  for (int mt = 0; mt < MTILES; ++mt)
#pragma unroll
    for (int t = 0; t < ND; ++t)
#pragma unroll
      for (int i = 0; i < 8; ++i) Oreg[mt][t][i] = 0.f;

  for (int i = tid; i < M * HD; i += nthreads) {
    int h = i / HD, d = i % HD;
    Qsh[h * HDP + d] = (h < H) ? Q[((long)b * H + h) * HD + d] : __float2bfloat16(0.f);
  }
  for (int i = tid; i < STAGES * KBUF; i += nthreads) Ksh[i] = __float2bfloat16(0.f);
  if (tid < M) { msh[tid] = -INFINITY; lsh[tid] = 0.f; }

  const float scale = sm_scale * LOG2E;
  const int* bt = BlockTable + (long)b * max_blocks;
  const int qrow = lane % 16, qcol = (lane / 16) * 8;   // ldmatrix lane addressing
  const int er0 = lane / 4, ec = (lane % 4) * 2;        // mma C-fragment row/col
  const int ntiles = (kv_end > kv_start) ? (kv_end - kv_start + NT - 1) / NT : 0;

#define PREFETCH(t)                                                                   \
  do {                                                                                \
    int base_ = kv_start + (t) * NT, valid_ = min(NT, kv_end - base_);                \
    bf16_t* dst_ = Ksh + ((t) % STAGES) * KBUF;                                       \
    for (int v = tid; v < valid_ * (HD / 8); v += nthreads) {                         \
      int j = v / (HD / 8), dd = (v % (HD / 8)) * 8;                                  \
      int n = base_ + j, page = bt[n / BLK];                                          \
      __pipeline_memcpy_async(dst_ + j * HDP + dd,                                    \
          Latent + (long)(page * BLK + n % BLK) * HD + dd, 16);                       \
    }                                                                                 \
    __pipeline_commit();                                                              \
  } while (0)

  __syncthreads();
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) PREFETCH(s);

  for (int t = 0; t < ntiles; ++t) {
    const int base = kv_start + t * NT, valid = min(NT, kv_end - base);
    PREFETCH(t + STAGES - 1);              // prefetch furthest (empty-commit if OOB)
    __pipeline_wait_prior(STAGES - 1);     // leave STAGES-1 in flight -> tile t ready
    __syncthreads();
    bf16_t* Kc = Ksh + (t % STAGES) * KBUF;

    // GEMM1 + REGISTER-resident online softmax. A warp owns its head-tile's full
    // token range (NTOK16 = NT/16 n16-tiles), so row max/sum is a within-warp
    // shuffle (no S smem round-trip). Q (A operand) is loaded once per k-tile and
    // reused across all NTOK16 token-tiles. NT=16 -> NTOK16=1 (low-bs winner);
    // NT=64 -> NTOK16=4 (big-tile high-bs variant).
    constexpr int NTOK16 = NT / 16;
    for (int mt = warp; mt < MTILES; mt += NWARPS) {
      float C[NTOK16][8];
#pragma unroll
      for (int n = 0; n < NTOK16; ++n)
#pragma unroll
        for (int i = 0; i < 8; ++i) C[n][i] = 0.f;
      const bf16_t* Qb = Qsh + mt * 16 * HDP;
      for (int kt = 0; kt < HD / 16; ++kt) {
        uint32_t A[4];
        ldm_x4(A, Qb + qrow * HDP + kt * 16 + qcol);          // hoisted across token-tiles
#pragma unroll
        for (int n = 0; n < NTOK16; ++n) {
          uint32_t B[4];
          ldm_x4(B, Kc + (n * 16 + qrow) * HDP + kt * 16 + qcol);
          { uint32_t t = B[1]; B[1] = B[2]; B[2] = t; }
          mma16(C[n], A, B);
        }
      }
#pragma unroll
      for (int n = 0; n < NTOK16; ++n)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          int col = n * 16 + ec + (i & 1) + (i >= 4 ? 8 : 0);
          C[n][i] = (col < valid) ? C[n][i] * scale : -INFINITY;
        }
      float rm0 = -INFINITY, rm1 = -INFINITY;
#pragma unroll
      for (int n = 0; n < NTOK16; ++n) {
        rm0 = fmaxf(rm0, fmaxf(fmaxf(C[n][0], C[n][1]), fmaxf(C[n][4], C[n][5])));
        rm1 = fmaxf(rm1, fmaxf(fmaxf(C[n][2], C[n][3]), fmaxf(C[n][6], C[n][7])));
      }
      rm0 = fmaxf(rm0, __shfl_xor_sync(0xffffffff, rm0, 1)); rm0 = fmaxf(rm0, __shfl_xor_sync(0xffffffff, rm0, 2));
      rm1 = fmaxf(rm1, __shfl_xor_sync(0xffffffff, rm1, 1)); rm1 = fmaxf(rm1, __shfl_xor_sync(0xffffffff, rm1, 2));
      int row0 = mt * 16 + er0, row1 = mt * 16 + er0 + 8;
      float mo0 = msh[row0], mn0 = fmaxf(mo0, rm0), al0 = exp2f(mo0 - mn0);
      float mo1 = msh[row1], mn1 = fmaxf(mo1, rm1), al1 = exp2f(mo1 - mn1);
      float s0 = 0.f, s1 = 0.f;
#pragma unroll
      for (int n = 0; n < NTOK16; ++n) {
        float p[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) p[i] = exp2f(C[n][i] - ((i & 2) ? mn1 : mn0));
        int cb = n * 16 + ec;
        Psh[row0 * NT + cb] = __float2bfloat16(p[0]); Psh[row0 * NT + cb + 1] = __float2bfloat16(p[1]);
        Psh[row0 * NT + cb + 8] = __float2bfloat16(p[4]); Psh[row0 * NT + cb + 9] = __float2bfloat16(p[5]);
        Psh[row1 * NT + cb] = __float2bfloat16(p[2]); Psh[row1 * NT + cb + 1] = __float2bfloat16(p[3]);
        Psh[row1 * NT + cb + 8] = __float2bfloat16(p[6]); Psh[row1 * NT + cb + 9] = __float2bfloat16(p[7]);
        s0 += p[0] + p[1] + p[4] + p[5]; s1 += p[2] + p[3] + p[6] + p[7];
      }
      s0 += __shfl_xor_sync(0xffffffff, s0, 1); s0 += __shfl_xor_sync(0xffffffff, s0, 2);
      s1 += __shfl_xor_sync(0xffffffff, s1, 1); s1 += __shfl_xor_sync(0xffffffff, s1, 2);
      if ((lane & 3) == 0) {
        msh[row0] = mn0; lsh[row0] = lsh[row0] * al0 + s0; ash[row0] = al0;
        msh[row1] = mn1; lsh[row1] = lsh[row1] * al1 + s1; ash[row1] = al1;
      }
    }
    __syncthreads();

#pragma unroll
    for (int mt = 0; mt < MTILES; ++mt) {
      float a0 = ash[mt * 16 + er0], a1 = ash[mt * 16 + er0 + 8];
#pragma unroll
      for (int t2 = 0; t2 < ND; ++t2) {
        Oreg[mt][t2][0] *= a0; Oreg[mt][t2][1] *= a0; Oreg[mt][t2][4] *= a0; Oreg[mt][t2][5] *= a0;
        Oreg[mt][t2][2] *= a1; Oreg[mt][t2][3] *= a1; Oreg[mt][t2][6] *= a1; Oreg[mt][t2][7] *= a1;
      }
    }
    // GEMM2: O[M,512] += P V. warp owns dim n16-tiles [warp*ND, +ND).
    // P (A operand) depends only on (mt, kt) -> hoist out of the dim loop
    // (loaded once per kt, reused across ND dim-tiles, not ND times).
    constexpr int KTOK = NT / 16;
#pragma unroll
    for (int mt = 0; mt < MTILES; ++mt) {
      const bf16_t* Pb = Psh + mt * 16 * NT;
#pragma unroll
      for (int kt = 0; kt < KTOK; ++kt) {
        uint32_t A[4];
        ldm_x4(A, Pb + qrow * NT + kt * 16 + qcol);
#pragma unroll
        for (int t2 = 0; t2 < ND; ++t2) {
          int dim0 = (warp * ND + t2) * 16;
          uint32_t B[4];
          ldm_x4_trans(B, Kc + (kt * 16 + qrow) * HDP + dim0 + qcol);
          mma16(Oreg[mt][t2], A, B);
        }
      }
    }
    __syncthreads();
  }
#undef PREFETCH

  const bool empty = (kv_end <= kv_start);
#pragma unroll
  for (int mt = 0; mt < MTILES; ++mt) {
    int rA = mt * 16 + er0, rB = mt * 16 + er0 + 8;
    float lA = empty ? 0.f : lsh[rA], lB = empty ? 0.f : lsh[rB];
    float iA = lA > 0.f ? 1.f / lA : 0.f, iB = lB > 0.f ? 1.f / lB : 0.f;
#pragma unroll
    for (int t2 = 0; t2 < ND; ++t2) {
      int dim0 = (warp * ND + t2) * 16;
      float* C = Oreg[mt][t2];
      // C[i]: row = er0 (+8 if i&2), col = dim0 + ec + (i&1) + (i>=4?8:0)
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        int row = (i & 2) ? rB : rA;
        float inv = (i & 2) ? iB : iA;
        int col = dim0 + ec + (i & 1) + (i >= 4 ? 8 : 0);
        if (splits == 1) {
          if (row < H) O[((long)b * H + row) * CKV + col] = __float2bfloat16(empty ? 0.f : C[i] * inv);
        } else {  // fp32 unnormalized acc (bf16-MidO tried: correct but slower here, not bandwidth-bound)
          MidO[(((long)b * splits + sp) * M + row) * CKV + col] = empty ? 0.f : C[i];
        }
      }
    }
  }
  if (splits > 1 && tid < M) {
    MidM[((long)b * splits + sp) * M + tid] = empty ? -INFINITY : msh[tid];
    MidL[((long)b * splits + sp) * M + tid] = empty ? 0.f : lsh[tid];
  }
}

__global__ void stage2(const float* __restrict__ MidO, const float* __restrict__ MidM,
                       const float* __restrict__ MidL, bf16_t* __restrict__ O, int H, int M, int splits) {
  const int b = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
  float Mx = -INFINITY;
  for (int s = 0; s < splits; ++s) Mx = fmaxf(Mx, MidM[((long)b * splits + s) * M + h]);
  float acc = 0.f, L = 0.f;
  for (int s = 0; s < splits; ++s) {
    float ms = MidM[((long)b * splits + s) * M + h];
    if (ms == -INFINITY) continue;
    float w = exp2f(ms - Mx);
    L += MidL[((long)b * splits + s) * M + h] * w;
    acc += MidO[(((long)b * splits + s) * M + h) * CKV + d] * w;
  }
  O[((long)b * H + h) * CKV + d] = __float2bfloat16(L > 0.f ? acc / L : 0.f);
}

template <int BLK, int NT = 16, int NWARPS = 4, int STAGES = 2, int NACC = 1>
void launch(torch::Tensor q, torch::Tensor latent, torch::Tensor block_table,
            torch::Tensor seqlens, torch::Tensor o, double sm_scale, int splits) {
  const int bs = q.size(0), H = q.size(1), max_blocks = block_table.size(1);
  const int MTILES = (H + 15) / 16, M = MTILES * 16;
  dim3 grid(bs, splits), block(NWARPS * 32);
  size_t sm = (size_t)STAGES * NT * HDP * 2 + M * HDP * 2 + M * NT * 2 + 3 * M * 4;  // K x STAGES, no Ssh
  auto stream = at::cuda::getCurrentCUDAStream();
  const bf16_t* Q = (const bf16_t*)q.data_ptr(); const bf16_t* L = (const bf16_t*)latent.data_ptr();
  const int* BT = (const int*)block_table.data_ptr(); const int* SL = (const int*)seqlens.data_ptr();
  bf16_t* OO = (bf16_t*)o.data_ptr();
  torch::Tensor mO, mM, mL; float *pO = nullptr, *pM = nullptr, *pL = nullptr;
  if (splits > 1) {
    auto opt = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());
    mO = torch::empty({bs, splits, M, CKV}, opt);
    mM = torch::empty({bs, splits, M}, opt); mL = torch::empty({bs, splits, M}, opt);
    pO = mO.data_ptr<float>(); pM = mM.data_ptr<float>(); pL = mL.data_ptr<float>();
  }
#define LAUNCH(MT) \
  cudaFuncSetAttribute(stage1<MT, BLK, NT, NWARPS, STAGES, NACC>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm); \
  stage1<MT, BLK, NT, NWARPS, STAGES, NACC><<<grid, block, sm, stream>>>(Q, L, BT, SL, OO, pO, pM, pL, (float)sm_scale, H, splits, max_blocks)
  if (MTILES == 1) { LAUNCH(1); } else { LAUNCH(2); }
#undef LAUNCH
  if (splits > 1) stage2<<<dim3(bs, H), dim3(CKV), 0, stream>>>(pO, pM, pL, OO, H, M, splits);
}
}  // namespace ldm
