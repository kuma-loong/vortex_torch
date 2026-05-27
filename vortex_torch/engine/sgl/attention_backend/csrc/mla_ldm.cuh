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

// MINB = min CTAs/SM target for __launch_bounds__. ncu showed this kernel is
// occupancy/latency-bound (not MMA-bound): at the natural 254 regs only 2 CTAs/SM
// fit (12.5% occ, 1.77 warps/sched, ~70% no-eligible cycles). Forcing MINB=3
// (=> ptxas reschedules to 168 regs, NO spill) lifts occupancy to 18.75% and
// beats Triton on the bs=128 flagship. MINB=2 = the old behaviour.
template <int MTILES, int BLK, int NT, int NWARPS, int STAGES, int NACC = 1, int MINB = 2>
__global__ void __launch_bounds__(NWARPS * 32, MINB) stage1(const bf16_t* __restrict__ Q, const bf16_t* __restrict__ Latent,
                       const int* __restrict__ BlockTable, const int* __restrict__ Seqlens,
                       bf16_t* __restrict__ O, float* __restrict__ MidO, float* __restrict__ MidM,
                       float* __restrict__ MidL, float sm_scale, int H, int splits, int max_blocks,
                       const int* __restrict__ NSplits = nullptr,
                       const int* __restrict__ WorkBatch = nullptr,
                       const int* __restrict__ WorkKvStart = nullptr,
                       const int* __restrict__ WorkKvEnd = nullptr) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, tid = threadIdx.x;
  const int nthreads = NWARPS * 32;
  constexpr int M = MTILES * 16;
  constexpr int ND = (CKV / 16) / NWARPS;   // dim n16-tiles per warp (32/NWARPS)

  // Resolve this CTA's work item. Two schedulers share one body:
  //  * WORK-QUEUE (WorkBatch != null): 1-D grid of exactly target_total CTAs, each
  //    a pre-balanced equal-size KV chunk (b, [kv_start,kv_end)) -> ~no idle CTAs,
  //    balanced regardless of ragged skew. MidO/M/L indexed by the flat CTA id.
  //  * 2-D (bs, splits): grid.y = fixed MAX_SPLITS; per-request NSplits[b] (~prop to
  //    seqlen) sets active splits, slots beyond it bail as no-ops.
  // Both fix the grid at launch => run() is cuda-graph-capturable.
  int b, kv_start, kv_end, seqlen; long out_idx; bool direct;
  if (WorkBatch) {
    int c = blockIdx.x;
    b = WorkBatch[c];
    if (b < 0) return;                       // padding CTA (sum nsplits <= grid)
    seqlen = 0;                              // unused in work-queue mode (KV range comes
    kv_start = WorkKvStart[c]; kv_end = WorkKvEnd[c];  // from the queue, not Seqlens)
    out_idx = c; direct = false;
  } else {
    b = blockIdx.x; int sp = blockIdx.y;
    seqlen = Seqlens[b];
    int nsp = NSplits ? NSplits[b] : splits;
    int chunk = (seqlen + nsp - 1) / nsp;
    kv_start = (sp < nsp) ? sp * chunk : seqlen;
    kv_end = min(kv_start + chunk, seqlen);
    out_idx = (long)b * splits + sp; direct = (splits == 1);
    if (NSplits && sp >= nsp) {              // no-op slot: mark inactive, bail early
      if (tid < M) { MidM[out_idx * M + tid] = -INFINITY; MidL[out_idx * M + tid] = 0.f; }
      return;
    }
  }

  extern __shared__ char smem[];
  bf16_t* Ksh = reinterpret_cast<bf16_t*>(smem);
  const int KBUF = NT * HDP;
  bf16_t* Qsh = Ksh + STAGES * KBUF;
  bf16_t* Psh = reinterpret_cast<bf16_t*>(Qsh + M * HDP);   // register softmax: no Ssh
  float* msh = reinterpret_cast<float*>(Psh + M * NT);
  float* lsh = msh + M, *ash = lsh + M;

  // bf16-PACKED O accumulator: MTILES*ND*4 regs (half the fp32 [..][8]). Halving
  // the O footprint gives ptxas register headroom at MINB=3 (no spill) and a small
  // bandwidth gain (28.4% vs 27.8% DRAM). Pairs k=(C[2k],C[2k+1]) share a head-row
  // => same online-softmax alpha, so rescale/accumulate are pure bf162 ops.
  // Accuracy vs fp32 ref ~2e-3 (<< the 3e-2 bar). To revert to fp32-O, swap this
  // for `float Oreg[MTILES][ND][8]` and the matching GEMM2/store blocks.
  __nv_bfloat162 Oreg[MTILES][ND][4];
  const __nv_bfloat162 BF162_ZERO = __floats2bfloat162_rn(0.f, 0.f);
#pragma unroll
  for (int mt = 0; mt < MTILES; ++mt)
#pragma unroll
    for (int t = 0; t < ND; ++t)
#pragma unroll
      for (int i = 0; i < 4; ++i) Oreg[mt][t][i] = BF162_ZERO;

  // Q gather global->smem in 128-bit chunks (8 bf16 = 16B; HD=576 & HDP=584 are
  // both 8-aligned) instead of scalar stores.
  for (int i = tid; i < M * (HD / 8); i += nthreads) {
    int h = i / (HD / 8), d8 = (i % (HD / 8)) * 8;
    int4 v = (h < H) ? *reinterpret_cast<const int4*>(Q + ((long)b * H + h) * HD + d8)
                     : int4{0, 0, 0, 0};
    *reinterpret_cast<int4*>(Qsh + h * HDP + d8) = v;
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
    // NOTE: a multi-accumulator GEMM1 (NACC>1, round-robin k-tiles to break the
    // 36-deep mma dependency chain) was implemented + measured here: NEGATIVE
    // (NACC1 2227 vs NACC2 2206 / NACC4 2174). __launch_bounds__(MINB=3) caps regs
    // at 168, so extra accumulators steal scheduling ILP rather than gain registers,
    // and at 3 CTAs/SM the mma latency is already hidden by the 12 resident warps.
    // Kept the single-accumulator form. (NACC stays a template knob, default 1.)
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
      __nv_bfloat162 va0 = __floats2bfloat162_rn(a0, a0), va1 = __floats2bfloat162_rn(a1, a1);
#pragma unroll
      for (int t2 = 0; t2 < ND; ++t2) {
        Oreg[mt][t2][0] = __hmul2(Oreg[mt][t2][0], va0);  // pair (C0,C1) head-row rA
        Oreg[mt][t2][1] = __hmul2(Oreg[mt][t2][1], va1);  // pair (C2,C3) head-row rB
        Oreg[mt][t2][2] = __hmul2(Oreg[mt][t2][2], va0);  // pair (C4,C5) head-row rA
        Oreg[mt][t2][3] = __hmul2(Oreg[mt][t2][3], va1);  // pair (C6,C7) head-row rB
      }
    }
    // GEMM2: O[M,512] += P V. The .f32 mma needs an fp32 accumulator, so each tile
    // mma's into a FRESH fp32 C, then folds it into the bf16 running O via bf162 add.
    constexpr int KTOK = NT / 16;
#pragma unroll
    for (int mt = 0; mt < MTILES; ++mt) {
      const bf16_t* Pb = Psh + mt * 16 * NT;
#pragma unroll
      for (int t2 = 0; t2 < ND; ++t2) {
        float C[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) C[i] = 0.f;
#pragma unroll
        for (int kt = 0; kt < KTOK; ++kt) {
          uint32_t A[4];
          ldm_x4(A, Pb + qrow * NT + kt * 16 + qcol);
          int dim0 = (warp * ND + t2) * 16;
          uint32_t B[4];
          ldm_x4_trans(B, Kc + (kt * 16 + qrow) * HDP + dim0 + qcol);
          mma16(C, A, B);
        }
        Oreg[mt][t2][0] = __hadd2(Oreg[mt][t2][0], __floats2bfloat162_rn(C[0], C[1]));
        Oreg[mt][t2][1] = __hadd2(Oreg[mt][t2][1], __floats2bfloat162_rn(C[2], C[3]));
        Oreg[mt][t2][2] = __hadd2(Oreg[mt][t2][2], __floats2bfloat162_rn(C[4], C[5]));
        Oreg[mt][t2][3] = __hadd2(Oreg[mt][t2][3], __floats2bfloat162_rn(C[6], C[7]));
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
      // VECTORIZED store: each bf162 pair p = (C[2p],C[2p+1]) lands in 2 CONTIGUOUS
      // columns of one head-row, so one 32-bit (bf16 O) / 64-bit (fp32 MidO) store
      // replaces 2 scalar ones. p=0,2 -> rA; p=1,3 -> rB; col = dim0+ec(+8 if p>=2).
      // cols are even (ec even, dim0%16) => naturally aligned for the vector store.
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        int row = (p & 1) ? rB : rA;
        float inv = (p & 1) ? iB : iA;
        int col = dim0 + ec + ((p >= 2) ? 8 : 0);
        if (row >= H) continue;                  // pad rows (>=H): stage2 only reads h<H
        float c0 = __low2float(Oreg[mt][t2][p]), c1 = __high2float(Oreg[mt][t2][p]);
        if (direct) {                            // float-precision rescale, then pack to bf162
          __nv_bfloat162 v = empty ? __floats2bfloat162_rn(0.f, 0.f)
                                   : __floats2bfloat162_rn(c0 * inv, c1 * inv);
          *reinterpret_cast<__nv_bfloat162*>(&O[((long)b * H + row) * CKV + col]) = v;
        } else {
          float2 v = empty ? make_float2(0.f, 0.f) : make_float2(c0, c1);
          *reinterpret_cast<float2*>(&MidO[(out_idx * M + row) * CKV + col]) = v;
        }
      }
    }
  }
  if (!direct && tid < M) {
    MidM[out_idx * M + tid] = empty ? -INFINITY : msh[tid];
    MidL[out_idx * M + tid] = empty ? 0.f : lsh[tid];
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

template <int BLK, int NT = 16, int NWARPS = 4, int STAGES = 2, int NACC = 1, int MINB = 2>
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
  cudaFuncSetAttribute(stage1<MT, BLK, NT, NWARPS, STAGES, NACC, MINB>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm); \
  stage1<MT, BLK, NT, NWARPS, STAGES, NACC, MINB><<<grid, block, sm, stream>>>(Q, L, BT, SL, OO, pO, pM, pL, (float)sm_scale, H, splits, max_blocks)
  if (MTILES == 1) { LAUNCH(1); } else { LAUNCH(2); }
#undef LAUNCH
  if (splits > 1) stage2<<<dim3(bs, H), dim3(CKV), 0, stream>>>(pO, pM, pL, OO, H, M, splits);
}

// ====================================================================== //
// WORK-QUEUE plan()/run() path for ragged batches.
//
// A flat 1-D grid of exactly `target_ctas` CTAs. The schedule kernel reads live
// seqlens and cuts each request into ceil-balanced equal-size KV chunks (split
// count ~proportional to its length), packing them contiguously into a work
// queue. So every CTA does a ~uniform amount of work regardless of how skewed
// the batch is (the longest request no longer serializes the wave), and there
// are ~no idle CTAs (sum of splits ~= target). vs the (bs, max_splits) grid this
// removes both the uniform-batch no-op overhead and the skew split cap, and uses
// far less scratch (target_ctas*M*CKV instead of bs*max_splits*M*CKV).
//
// init():  allocate the work-queue + MidO/M/L scratch (fixed sizes, once).
// plan():  schedule_wq populates the work queue from seqlens (one small kernel).
// run():   stage1 (work-queue mode) + stage2_wq over the queue.
// All grids fixed at init => plan() and run() are both cuda-graph-capturable.
// ====================================================================== //

// One block. Balanced split: s_b = clamp(round(target*seqlen_b/sum_seqlen), 1, cap);
// chunk_b = ceil(seqlen_b/s_b). Packs the s_b chunks of request b at WorkOffset[b]..
// and records the per-request offsets so stage2 reduces only its own chunks.
// target_ctas (grid + queue length) has +bs margin so rounding can't overflow.
__global__ void schedule_wq(const int* __restrict__ Seqlens,
                            int* __restrict__ WorkBatch, int* __restrict__ WorkKvStart,
                            int* __restrict__ WorkKvEnd, int* __restrict__ WorkOffset,
                            int bs, int target, int max_split_cap, int chunk_min, int target_ctas) {
  // Single block. Parallelized: only the bs-length prefix sum is serial (in
  // shared mem, ~bs adds); reduction, padding init, and work-item fill all run
  // across the block. (A serial single-thread fill of all work items costs ~40us
  // and dominates plan() — keep everything parallel.)
  extern __shared__ char sh[];
  int tid = threadIdx.x, nt = blockDim.x;
  float* red = reinterpret_cast<float*>(sh);     // nt floats
  int* ns = reinterpret_cast<int*>(red + nt);    // bs ints
  int* off = ns + bs;                            // bs+1 ints

  float local = 0.f;
  for (int b = tid; b < bs; b += nt) local += (float)Seqlens[b];
  red[tid] = local; __syncthreads();
  for (int s = nt >> 1; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid + s]; __syncthreads(); }
  float total = fmaxf(red[0], 1.f);

  for (int b = tid; b < bs; b += nt) {           // per-request split count (balanced)
    int seq = Seqlens[b];
    // FLOOR (not round) the proportional split: keeps total active CTAs just UNDER
    // one ctas/SM wave. Overshooting one wave costs a long tail (a few CTAs run
    // nearly alone in a 2nd wave) that is far worse than leaving some SMs idle --
    // e.g. H=16 bs=128: floor->4 splits (512 CTAs, 0.86 wave, 3023 GB/s) vs round->5
    // (640, 1.08 waves, ~2400). Also smooths the bs sawtooth at high bs.
    int s = (int)((float)target * (float)seq / total);          // ~proportional, biased down
    int by_chunk = (seq + chunk_min - 1) / chunk_min;           // don't split below chunk_min
    s = min(s, max(1, by_chunk));
    ns[b] = max(1, min(max_split_cap, s));
  }
  for (int w = tid; w < target_ctas; w += nt) WorkBatch[w] = -1;  // padding default
  __syncthreads();
  if (tid == 0) { int o = 0; for (int b = 0; b < bs; ++b) { off[b] = o; o += ns[b]; } off[bs] = o; }
  __syncthreads();
  for (int b = tid; b <= bs; b += nt) WorkOffset[b] = off[b];     // parallel offset write
  for (int b = tid; b < bs; b += nt) {           // parallel work-item fill (each thread a request)
    int seq = Seqlens[b], s = ns[b], base = off[b], chunk = (seq + s - 1) / s;
    for (int j = 0; j < s; ++j) {
      int w = base + j;
      if (w < target_ctas) { WorkBatch[w] = b; WorkKvStart[w] = j * chunk; WorkKvEnd[w] = min((j + 1) * chunk, seq); }
    }
  }
}

// stage2 over the work queue: reduce request b's own chunks [WorkOffset[b], [b+1]).
__global__ void stage2_wq(const float* __restrict__ MidO, const float* __restrict__ MidM,
                          const float* __restrict__ MidL, const int* __restrict__ WorkOffset,
                          bf16_t* __restrict__ O, int H, int M) {
  const int b = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
  const int lo = WorkOffset[b], hi = WorkOffset[b + 1];
  float Mx = -INFINITY;
  for (int c = lo; c < hi; ++c) Mx = fmaxf(Mx, MidM[(long)c * M + h]);
  float acc = 0.f, L = 0.f;
  for (int c = lo; c < hi; ++c) {
    float ms = MidM[(long)c * M + h];
    if (ms == -INFINITY) continue;
    float w = exp2f(ms - Mx);
    L += MidL[(long)c * M + h] * w;
    acc += MidO[((long)c * M + h) * CKV + d] * w;
  }
  O[((long)b * H + h) * CKV + d] = __float2bfloat16(L > 0.f ? acc / L : 0.f);
}

// plan(): populate the work queue from current seqlens (graph-safe, fixed grid).
inline void run_schedule_wq(torch::Tensor seqlens, torch::Tensor work_batch,
                            torch::Tensor work_kv_start, torch::Tensor work_kv_end,
                            torch::Tensor work_offset, int target, int max_split_cap, int chunk_min) {
  const int bs = seqlens.size(0), target_ctas = work_batch.size(0);
  auto stream = at::cuda::getCurrentCUDAStream();
  int sthreads = 256; while (sthreads > bs && sthreads > 64) sthreads >>= 1;
  size_t smem = sthreads * sizeof(float) + (2 * bs + 1) * sizeof(int);  // red[nt] + ns[bs] + off[bs+1]
  schedule_wq<<<1, sthreads, smem, stream>>>(
      seqlens.data_ptr<int>(), work_batch.data_ptr<int>(), work_kv_start.data_ptr<int>(),
      work_kv_end.data_ptr<int>(), work_offset.data_ptr<int>(), bs, target, max_split_cap, chunk_min, target_ctas);
}

// run(): the decode itself — stage1 over the work queue + stage2 reduction. Takes
// only per-layer data (q/latent/block_table/o) + the queue from plan(); does NOT
// need seqlens (plan() already consumed them). NO allocation, grids fixed =>
// cuda-graph-capturable, and one plan() feeds every layer's run().
template <int BLK, int NT = 16, int NWARPS = 4, int STAGES = 2, int NACC = 1, int MINB = 3>
void run_wq(torch::Tensor q, torch::Tensor latent, torch::Tensor block_table,
            torch::Tensor o, torch::Tensor work_batch, torch::Tensor work_kv_start,
            torch::Tensor work_kv_end, torch::Tensor work_offset,
            torch::Tensor mid_o, torch::Tensor mid_m, torch::Tensor mid_l, double sm_scale) {
  const int bs = q.size(0), H = q.size(1), max_blocks = block_table.size(1);
  const int MTILES = (H + 15) / 16, M = MTILES * 16, target_ctas = work_batch.size(0);
  auto stream = at::cuda::getCurrentCUDAStream();
  const bf16_t* Q = (const bf16_t*)q.data_ptr(); const bf16_t* L = (const bf16_t*)latent.data_ptr();
  const int* BT = (const int*)block_table.data_ptr();
  bf16_t* OO = (bf16_t*)o.data_ptr();
  const int *WB = work_batch.data_ptr<int>(), *WS = work_kv_start.data_ptr<int>(), *WE = work_kv_end.data_ptr<int>();
  float *pO = mid_o.data_ptr<float>(), *pM = mid_m.data_ptr<float>(), *pL = mid_l.data_ptr<float>();
  dim3 grid(target_ctas), block(NWARPS * 32);
  size_t sm = (size_t)STAGES * NT * HDP * 2 + M * HDP * 2 + M * NT * 2 + 3 * M * 4;
#define LAUNCHWQ(MT) \
  cudaFuncSetAttribute(stage1<MT, BLK, NT, NWARPS, STAGES, NACC, MINB>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm); \
  stage1<MT, BLK, NT, NWARPS, STAGES, NACC, MINB><<<grid, block, sm, stream>>>( \
      Q, L, BT, /*Seqlens*/nullptr, OO, pO, pM, pL, (float)sm_scale, H, /*splits*/1, max_blocks, /*NSplits*/nullptr, WB, WS, WE)
  if (MTILES == 1) { LAUNCHWQ(1); } else { LAUNCHWQ(2); }
#undef LAUNCHWQ
  stage2_wq<<<dim3(bs, H), dim3(CKV), 0, stream>>>(pO, pM, pL, work_offset.data_ptr<int>(), OO, H, M);
}
}  // namespace ldm
