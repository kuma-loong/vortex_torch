// Shared templated MLA-decode kernel for the per-(bs,block) specialized files.
// Each specialized .cu instantiates launch_mla<BLK>() with a compile-time
// block size and a baked-in `splits` tuned for its batch size.
//
// mma.sync.m16n8k16 (bf16), REGISTER-resident O accumulator, cp.async double-
// buffered K tiles, online softmax in shared memory, split-KV + stage-2 merge.
// Anonymous namespace => each TU gets private kernel copies (no dup symbols when
// many specialized files link into one extension).
#pragma once
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>

namespace {
typedef __nv_bfloat16 bf16_t;
constexpr int HD = 576, CKV = 512;
constexpr float LOG2E = 1.4426950408889634f;

__device__ __forceinline__ uint32_t pk(bf16_t a, bf16_t b) {
  return (uint32_t)__bfloat16_as_ushort(a) | ((uint32_t)__bfloat16_as_ushort(b) << 16);
}
__device__ __forceinline__ void mma_m16n8k16(float* C, const uint32_t* A, const uint32_t* B) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
        "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
}

template <int MTILES, int BLK, int NT, int NWARPS>
__global__ void stage1(const bf16_t* __restrict__ Q, const bf16_t* __restrict__ Latent,
                       const int* __restrict__ BlockTable, const int* __restrict__ Seqlens,
                       bf16_t* __restrict__ O, float* __restrict__ MidO, float* __restrict__ MidM,
                       float* __restrict__ MidL, float sm_scale, int H, int splits, int max_blocks) {
  const int b = blockIdx.x, sp = blockIdx.y;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, tid = threadIdx.x;
  const int nthreads = NWARPS * 32;
  constexpr int M = MTILES * 16;
  constexpr int OC = CKV / NWARPS, ON = OC / 8;   // output cols / n8-tiles per warp
  const int seqlen = Seqlens[b];
  const int chunk = (seqlen + splits - 1) / splits;
  const int kv_start = sp * chunk, kv_end = min(kv_start + chunk, seqlen);

  extern __shared__ char smem[];
  bf16_t* Ksh = reinterpret_cast<bf16_t*>(smem);
  const int KBUF = NT * HD;
  bf16_t* Qsh = Ksh + 2 * KBUF;
  float* Ssh = reinterpret_cast<float*>(Qsh + M * HD);
  bf16_t* Psh = reinterpret_cast<bf16_t*>(Ssh + M * NT);
  float* msh = reinterpret_cast<float*>(Psh + M * NT);
  float* lsh = msh + M;
  float* ash = lsh + M;

  float Oreg[MTILES][ON][4];
#pragma unroll
  for (int mt = 0; mt < MTILES; ++mt)
#pragma unroll
    for (int t = 0; t < ON; ++t)
#pragma unroll
      for (int i = 0; i < 4; ++i) Oreg[mt][t][i] = 0.f;

  for (int i = tid; i < M * HD; i += nthreads) {
    int h = i / HD, d = i % HD;
    Qsh[i] = (h < H) ? Q[((long)b * H + h) * HD + d] : __float2bfloat16(0.f);
  }
  for (int i = tid; i < 2 * KBUF; i += nthreads) Ksh[i] = __float2bfloat16(0.f);
  if (tid < M) { msh[tid] = -INFINITY; lsh[tid] = 0.f; }

  const float scale = sm_scale * LOG2E;
  const int* bt = BlockTable + (long)b * max_blocks;
  const int r0 = lane / 4, r1 = r0 + 8, cc = (lane % 4) * 2, nn = lane / 4, kk = (lane % 4) * 2;
  const int ntiles = (kv_end > kv_start) ? (kv_end - kv_start + NT - 1) / NT : 0;

#define PREFETCH(t)                                                                   \
  do {                                                                                \
    int base_ = kv_start + (t) * NT, valid_ = min(NT, kv_end - base_);                \
    bf16_t* dst_ = Ksh + ((t) & 1) * KBUF;                                            \
    for (int v = tid; v < valid_ * (HD / 8); v += nthreads) {                         \
      int j = v / (HD / 8), dd = (v % (HD / 8)) * 8;                                  \
      int n = base_ + j, page = bt[n / BLK];                                          \
      __pipeline_memcpy_async(dst_ + j * HD + dd,                                     \
          Latent + (long)(page * BLK + n % BLK) * HD + dd, 16);                       \
    }                                                                                 \
    __pipeline_commit();                                                              \
  } while (0)

  __syncthreads();
  if (ntiles > 0) PREFETCH(0);

  for (int t = 0; t < ntiles; ++t) {
    const int base = kv_start + t * NT, valid = min(NT, kv_end - base), cur = t & 1;
    if (t + 1 < ntiles) { PREFETCH(t + 1); __pipeline_wait_prior(1); }
    else __pipeline_wait_prior(0);
    __syncthreads();
    bf16_t* Kc = Ksh + cur * KBUF;

    for (int pair = warp; pair < MTILES * (NT / 8); pair += NWARPS) {
      int mt = pair / (NT / 8), n8 = pair % (NT / 8);
      float C[4] = {0, 0, 0, 0};
      const bf16_t* Qbase = Qsh + mt * 16 * HD;
      const bf16_t* Kbase = Kc + (n8 * 8) * HD;
      for (int ktt = 0; ktt < HD / 16; ++ktt) {
        int c = ktt * 16;
        uint32_t A[4] = {pk(Qbase[r0 * HD + c + cc], Qbase[r0 * HD + c + cc + 1]),
                         pk(Qbase[r1 * HD + c + cc], Qbase[r1 * HD + c + cc + 1]),
                         pk(Qbase[r0 * HD + c + cc + 8], Qbase[r0 * HD + c + cc + 9]),
                         pk(Qbase[r1 * HD + c + cc + 8], Qbase[r1 * HD + c + cc + 9])};
        uint32_t B[2] = {pk(Kbase[nn * HD + c + kk], Kbase[nn * HD + c + kk + 1]),
                         pk(Kbase[nn * HD + c + kk + 8], Kbase[nn * HD + c + kk + 9])};
        mma_m16n8k16(C, A, B);
      }
      Ssh[(mt * 16 + r0) * NT + n8 * 8 + cc] = C[0] * scale;
      Ssh[(mt * 16 + r0) * NT + n8 * 8 + cc + 1] = C[1] * scale;
      Ssh[(mt * 16 + r1) * NT + n8 * 8 + cc] = C[2] * scale;
      Ssh[(mt * 16 + r1) * NT + n8 * 8 + cc + 1] = C[3] * scale;
    }
    __syncthreads();

    for (int h = tid; h < M; h += nthreads) {
      float tmax = -INFINITY;
      for (int n = 0; n < valid; ++n) tmax = fmaxf(tmax, Ssh[h * NT + n]);
      float m_old = msh[h], m_new = fmaxf(m_old, tmax);
      float alpha = exp2f(m_old - m_new), lsum = 0.f;
      for (int n = 0; n < NT; ++n) {
        float p = (n < valid) ? exp2f(Ssh[h * NT + n] - m_new) : 0.f;
        Psh[h * NT + n] = __float2bfloat16(p);
        lsum += p;
      }
      lsh[h] = lsh[h] * alpha + lsum; msh[h] = m_new; ash[h] = alpha;
    }
    __syncthreads();

#pragma unroll
    for (int mt = 0; mt < MTILES; ++mt) {
      float a0 = ash[mt * 16 + r0], a1 = ash[mt * 16 + r1];
#pragma unroll
      for (int t2 = 0; t2 < ON; ++t2) {
        Oreg[mt][t2][0] *= a0; Oreg[mt][t2][1] *= a0;
        Oreg[mt][t2][2] *= a1; Oreg[mt][t2][3] *= a1;
      }
    }
#pragma unroll
    for (int mt = 0; mt < MTILES; ++mt) {
      const bf16_t* Pbase = Psh + mt * 16 * NT;
#pragma unroll
      for (int t2 = 0; t2 < ON; ++t2) {
        int dim0 = warp * OC + t2 * 8;
        float C[4] = {Oreg[mt][t2][0], Oreg[mt][t2][1], Oreg[mt][t2][2], Oreg[mt][t2][3]};
        for (int ktt = 0; ktt < NT / 16; ++ktt) {
          int c = ktt * 16;
          uint32_t A[4] = {pk(Pbase[r0 * NT + c + cc], Pbase[r0 * NT + c + cc + 1]),
                           pk(Pbase[r1 * NT + c + cc], Pbase[r1 * NT + c + cc + 1]),
                           pk(Pbase[r0 * NT + c + cc + 8], Pbase[r0 * NT + c + cc + 9]),
                           pk(Pbase[r1 * NT + c + cc + 8], Pbase[r1 * NT + c + cc + 9])};
          const bf16_t* Vb = Kc + c * HD + dim0;
          uint32_t B[2] = {pk(Vb[kk * HD + nn], Vb[(kk + 1) * HD + nn]),
                           pk(Vb[(kk + 8) * HD + nn], Vb[(kk + 9) * HD + nn])};
          mma_m16n8k16(C, A, B);
        }
        Oreg[mt][t2][0] = C[0]; Oreg[mt][t2][1] = C[1];
        Oreg[mt][t2][2] = C[2]; Oreg[mt][t2][3] = C[3];
      }
    }
    __syncthreads();
  }
#undef PREFETCH

  const bool empty = (kv_end <= kv_start);
#pragma unroll
  for (int mt = 0; mt < MTILES; ++mt) {
    int row0 = mt * 16 + r0, row1 = mt * 16 + r1;
    float l0 = empty ? 0.f : lsh[row0], l1 = empty ? 0.f : lsh[row1];
    float i0 = l0 > 0.f ? 1.f / l0 : 0.f, i1 = l1 > 0.f ? 1.f / l1 : 0.f;
#pragma unroll
    for (int t2 = 0; t2 < ON; ++t2) {
      int col = warp * OC + t2 * 8 + cc;
      if (splits == 1) {
        if (row0 < H) {
          O[((long)b * H + row0) * CKV + col] = __float2bfloat16(Oreg[mt][t2][0] * i0);
          O[((long)b * H + row0) * CKV + col + 1] = __float2bfloat16(Oreg[mt][t2][1] * i0);
        }
        if (row1 < H) {
          O[((long)b * H + row1) * CKV + col] = __float2bfloat16(Oreg[mt][t2][2] * i1);
          O[((long)b * H + row1) * CKV + col + 1] = __float2bfloat16(Oreg[mt][t2][3] * i1);
        }
      } else {
        float* mo = MidO + (((long)b * splits + sp) * M) * CKV;
        mo[row0 * CKV + col] = empty ? 0.f : Oreg[mt][t2][0];
        mo[row0 * CKV + col + 1] = empty ? 0.f : Oreg[mt][t2][1];
        mo[row1 * CKV + col] = empty ? 0.f : Oreg[mt][t2][2];
        mo[row1 * CKV + col + 1] = empty ? 0.f : Oreg[mt][t2][3];
      }
    }
  }
  if (splits > 1 && tid < M) {
    MidM[((long)b * splits + sp) * M + tid] = empty ? -INFINITY : msh[tid];
    MidL[((long)b * splits + sp) * M + tid] = empty ? 0.f : lsh[tid];
  }
}

__global__ void stage2(const float* __restrict__ MidO, const float* __restrict__ MidM,
                       const float* __restrict__ MidL, bf16_t* __restrict__ O,
                       int H, int M, int splits) {
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

template <int BLK, int NT = 32, int NWARPS = 4>
void launch_mla(torch::Tensor q, torch::Tensor latent, torch::Tensor block_table,
                torch::Tensor seqlens, torch::Tensor o, double sm_scale, int splits) {
  const int bs = q.size(0), H = q.size(1), max_blocks = block_table.size(1);
  const int MTILES = (H + 15) / 16, M = MTILES * 16;
  dim3 grid(bs, splits), block(NWARPS * 32);
  size_t sm = (size_t)2 * NT * HD * 2 + M * HD * 2 + M * NT * 4 + M * NT * 2 + 3 * M * 4;
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
  cudaFuncSetAttribute(stage1<MT, BLK, NT, NWARPS>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm); \
  stage1<MT, BLK, NT, NWARPS><<<grid, block, sm, stream>>>(Q, L, BT, SL, OO, pO, pM, pL, (float)sm_scale, H, splits, max_blocks)
  if (MTILES == 1) { LAUNCH(1); } else { LAUNCH(2); }
#undef LAUNCH
  if (splits > 1) stage2<<<dim3(bs, H), dim3(CKV), 0, stream>>>(pO, pM, pL, OO, H, M, splits);
}
}  // namespace
