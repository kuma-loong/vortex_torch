// Tensor-core MLA decode via mma.sync.m16n8k16 (bf16), REGISTER-resident O.
//
// Why mma not wmma: wmma's accumulator fragment layout is opaque, forcing the O
// accumulator into shared memory (32KB) + a load/store round-trip every tile ->
// 1 CTA/SM, ~230 GB/s wall. With mma.sync the C-fragment (row,col) layout is
// known, so O stays in registers, smem shrinks (no Osh) and >=3 CTAs/SM fit.
//
//   GEMM1  S[M,NT] = Q[M,576]·K[NT,576]^T  (mma, C scattered to smem Ssh)
//   softmax in smem -> P[M,NT] bf16, alpha[M]
//   rescale register-O by alpha[row]
//   GEMM2  O[M,512] += P[M,NT]·V[NT,512]   (mma, accumulate in registers)
// split-KV over grid.y; stage-2 merges. M = 16*MTILES (heads padded to mult 16).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>

typedef __nv_bfloat16 bf16;
#define HD 576
#define CKV 512
#define NT 32                 // KV tokens per tile
#define WARPS 4
#define OC (CKV / WARPS)      // 128 output cols per warp
#define ON (OC / 8)           // 16 n8 output tiles per warp
#define LOG2E 1.4426950408889634f

__device__ __forceinline__ uint32_t pk(bf16 a, bf16 b) {
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

template <int MTILES>
__global__ void mma_stage1(
    const bf16* __restrict__ Q, const bf16* __restrict__ Latent,
    const int* __restrict__ BlockTable, const int* __restrict__ Seqlens,
    bf16* __restrict__ O, float* __restrict__ MidO, float* __restrict__ MidM,
    float* __restrict__ MidL, float sm_scale, int block_size, int H, int splits, int max_blocks) {
  const int b = blockIdx.x, sp = blockIdx.y;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, tid = threadIdx.x;
  const int nthreads = WARPS * 32;
  constexpr int M = MTILES * 16;
  const int seqlen = Seqlens[b];
  const int chunk = (seqlen + splits - 1) / splits;
  const int kv_start = sp * chunk, kv_end = min(kv_start + chunk, seqlen);

  extern __shared__ char smem[];
  bf16*  Ksh = reinterpret_cast<bf16*>(smem);          // [2][NT][HD] double-buffered
  const int KBUF = NT * HD;
  bf16*  Qsh = Ksh + 2 * KBUF;                         // [M][HD]
  float* Ssh = reinterpret_cast<float*>(Qsh + M * HD); // [M][NT]
  bf16*  Psh = reinterpret_cast<bf16*>(Ssh + M * NT);  // [M][NT]
  float* msh = reinterpret_cast<float*>(Psh + M * NT); // [M]
  float* lsh = msh + M;                                // [M]
  float* ash = lsh + M;                                // [M]

  // register-resident O accumulator: per m-tile, ON n8-tiles, 4 fp32 each
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
  for (int i = tid; i < 2 * KBUF; i += nthreads) Ksh[i] = __float2bfloat16(0.f);  // pad
  if (tid < M) { msh[tid] = -INFINITY; lsh[tid] = 0.f; }

  const float scale = sm_scale * LOG2E;
  const int* bt = BlockTable + (long)b * max_blocks;
  const int r0 = lane / 4, r1 = r0 + 8, cc = (lane % 4) * 2, nn = lane / 4, kk = (lane % 4) * 2;
  const int ntiles = (kv_end > kv_start) ? (kv_end - kv_start + NT - 1) / NT : 0;

  // cp.async prefetch of tile `t` into buffer (t&1); valid rows only (invalid
  // rows keep zero-padded/stale-finite data, masked out in softmax).
#define PREFETCH(t)                                                                  \
  do {                                                                               \
    int base_ = kv_start + (t) * NT, valid_ = min(NT, kv_end - base_);               \
    bf16* dst_ = Ksh + ((t) & 1) * KBUF;                                             \
    for (int v = tid; v < valid_ * (HD / 8); v += nthreads) {                        \
      int j = v / (HD / 8), dd = (v % (HD / 8)) * 8;                                 \
      int n = base_ + j, page = bt[n / block_size];                                  \
      __pipeline_memcpy_async(dst_ + j * HD + dd,                                     \
          Latent + (long)(page * block_size + n % block_size) * HD + dd, 16);        \
    }                                                                                \
    __pipeline_commit();                                                             \
  } while (0)

  __syncthreads();
  if (ntiles > 0) PREFETCH(0);

  for (int t = 0; t < ntiles; ++t) {
    const int base = kv_start + t * NT;
    const int valid = min(NT, kv_end - base);
    const int cur = t & 1;
    if (t + 1 < ntiles) { PREFETCH(t + 1); __pipeline_wait_prior(1); }
    else __pipeline_wait_prior(0);
    __syncthreads();
    bf16* Kc = Ksh + cur * KBUF;

    // GEMM1: S = Q K^T. (mt, n8) pairs spread over warps.
    for (int pair = warp; pair < MTILES * (NT / 8); pair += WARPS) {
      int mt = pair / (NT / 8), n8 = pair % (NT / 8);
      float C[4] = {0, 0, 0, 0};
      const bf16* Qbase = Qsh + mt * 16 * HD;
      const bf16* Kbase = Kc + (n8 * 8) * HD;
      for (int kt = 0; kt < HD / 16; ++kt) {
        int c = kt * 16;
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

    // softmax per row -> P, alpha; update m,l
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

    // rescale register-O by alpha[row]
#pragma unroll
    for (int mt = 0; mt < MTILES; ++mt) {
      float a0 = ash[mt * 16 + r0], a1 = ash[mt * 16 + r1];
#pragma unroll
      for (int t = 0; t < ON; ++t) {
        Oreg[mt][t][0] *= a0; Oreg[mt][t][1] *= a0;
        Oreg[mt][t][2] *= a1; Oreg[mt][t][3] *= a1;
      }
    }
    // GEMM2: O += P V. each warp owns cols [warp*OC, +OC); ON n8-tiles.
#pragma unroll
    for (int mt = 0; mt < MTILES; ++mt) {
      const bf16* Pbase = Psh + mt * 16 * NT;
#pragma unroll
      for (int t = 0; t < ON; ++t) {
        int dim0 = warp * OC + t * 8;
        float C[4] = {Oreg[mt][t][0], Oreg[mt][t][1], Oreg[mt][t][2], Oreg[mt][t][3]};
        for (int kt = 0; kt < NT / 16; ++kt) {
          int c = kt * 16;
          uint32_t A[4] = {pk(Pbase[r0 * NT + c + cc], Pbase[r0 * NT + c + cc + 1]),
                           pk(Pbase[r1 * NT + c + cc], Pbase[r1 * NT + c + cc + 1]),
                           pk(Pbase[r0 * NT + c + cc + 8], Pbase[r0 * NT + c + cc + 9]),
                           pk(Pbase[r1 * NT + c + cc + 8], Pbase[r1 * NT + c + cc + 9])};
          const bf16* Vb = Kc + c * HD + dim0;   // V[token c..][dim dim0..]
          uint32_t B[2] = {pk(Vb[kk * HD + nn], Vb[(kk + 1) * HD + nn]),
                           pk(Vb[(kk + 8) * HD + nn], Vb[(kk + 9) * HD + nn])};
          mma_m16n8k16(C, A, B);
        }
        Oreg[mt][t][0] = C[0]; Oreg[mt][t][1] = C[1];
        Oreg[mt][t][2] = C[2]; Oreg[mt][t][3] = C[3];
      }
    }
    __syncthreads();
  }
#undef PREFETCH

  // epilogue: write O[row, col] (col = warp*OC + t*8 + cc + {0,1}), row r0/r1.
  const bool empty = (kv_end <= kv_start);
#pragma unroll
  for (int mt = 0; mt < MTILES; ++mt) {
    int row0 = mt * 16 + r0, row1 = mt * 16 + r1;
    float l0 = empty ? 0.f : lsh[row0], l1 = empty ? 0.f : lsh[row1];
    float inv0 = l0 > 0.f ? 1.f / l0 : 0.f, inv1 = l1 > 0.f ? 1.f / l1 : 0.f;
#pragma unroll
    for (int t = 0; t < ON; ++t) {
      int col = warp * OC + t * 8 + cc;
      if (splits == 1) {
        if (row0 < H) {
          O[((long)b * H + row0) * CKV + col] = __float2bfloat16(Oreg[mt][t][0] * inv0);
          O[((long)b * H + row0) * CKV + col + 1] = __float2bfloat16(Oreg[mt][t][1] * inv0);
        }
        if (row1 < H) {
          O[((long)b * H + row1) * CKV + col] = __float2bfloat16(Oreg[mt][t][2] * inv1);
          O[((long)b * H + row1) * CKV + col + 1] = __float2bfloat16(Oreg[mt][t][3] * inv1);
        }
      } else {
        float* mo = MidO + (((long)b * splits + sp) * M) * CKV;
        mo[row0 * CKV + col] = empty ? 0.f : Oreg[mt][t][0];
        mo[row0 * CKV + col + 1] = empty ? 0.f : Oreg[mt][t][1];
        mo[row1 * CKV + col] = empty ? 0.f : Oreg[mt][t][2];
        mo[row1 * CKV + col + 1] = empty ? 0.f : Oreg[mt][t][3];
      }
    }
  }
  if (splits > 1 && tid < M) {
    MidM[((long)b * splits + sp) * M + tid] = empty ? -INFINITY : msh[tid];
    MidL[((long)b * splits + sp) * M + tid] = empty ? 0.f : lsh[tid];
  }
}

__global__ void mma_stage2(const float* __restrict__ MidO, const float* __restrict__ MidM,
                           const float* __restrict__ MidL, bf16* __restrict__ O,
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

static size_t smem_bytes(int MTILES) {
  int M = MTILES * 16;
  return (size_t)2 * NT * HD * 2 + M * HD * 2 + M * NT * 4 + M * NT * 2 + 3 * M * 4;  // K double-buffered
}

void mla_decode_mma(torch::Tensor q, torch::Tensor latent, torch::Tensor block_table,
                    torch::Tensor seqlens, torch::Tensor o, double sm_scale,
                    int64_t block_size, int64_t splits) {
  const int bs = q.size(0), H = q.size(1), max_blocks = block_table.size(1);
  const int MTILES = (H + 15) / 16, M = MTILES * 16;
  dim3 grid(bs, splits), block(WARPS * 32);
  size_t sm = smem_bytes(MTILES);
  auto stream = at::cuda::getCurrentCUDAStream();
  const bf16* Q = (const bf16*)q.data_ptr(); const bf16* L = (const bf16*)latent.data_ptr();
  const int* BT = (const int*)block_table.data_ptr(); const int* SL = (const int*)seqlens.data_ptr();
  bf16* OO = (bf16*)o.data_ptr();
  torch::Tensor midO, midM, midL; float *pO = nullptr, *pM = nullptr, *pL = nullptr;
  if (splits > 1) {
    auto opt = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());
    midO = torch::empty({bs, (long)splits, M, CKV}, opt);
    midM = torch::empty({bs, (long)splits, M}, opt);
    midL = torch::empty({bs, (long)splits, M}, opt);
    pO = midO.data_ptr<float>(); pM = midM.data_ptr<float>(); pL = midL.data_ptr<float>();
  }
#define LAUNCH(MT) \
  cudaFuncSetAttribute(mma_stage1<MT>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm); \
  mma_stage1<MT><<<grid, block, sm, stream>>>(Q, L, BT, SL, OO, pO, pM, pL, (float)sm_scale, (int)block_size, H, (int)splits, max_blocks)
  if (MTILES == 1) { LAUNCH(1); } else { LAUNCH(2); }
#undef LAUNCH
  if (splits > 1) mma_stage2<<<dim3(bs, H), dim3(CKV), 0, stream>>>(pO, pM, pL, OO, H, M, (int)splits);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mla_decode_mma", &mla_decode_mma, "mma.sync register-O MLA decode");
}
