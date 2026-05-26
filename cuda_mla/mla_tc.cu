// Tensor-core block-table MLA decode (B200 / sm_100), self-written.
//
// FlashAttention-decode, two wmma GEMMs per KV tile (+ split-KV for parallelism):
//   GEMM1  S[M,NT] = Q[M,576] · K[NT,576]^T     (contract 576; K col-major = K^T)
//   GEMM2  O[M,512] += P[M,NT] · V[NT,512]      (V = K[:, :512], row-major)
// Online softmax in SHARED MEMORY (O accumulator in smem) so per-row rescale by
// alpha=exp2(m_old-m_new) is trivial. M = 16*MTILES (heads padded to mult of 16).
// grid = (bs, splits): one CTA per (request, KV chunk); stage-2 merges splits.
// NOTE: this is the wmma baseline — being migrated to mma.sync (m16n8k16) next.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;
typedef __nv_bfloat16 bf16;

#define HD 576
#define CKV 512
#define NT 64
#define WARPS 4
#define LOG2E 1.4426950408889634f
#define MAXSPLITS 32

template <int MTILES>
__global__ void mla_tc_stage1(
    const bf16* __restrict__ Q, const bf16* __restrict__ Latent,
    const int* __restrict__ BlockTable, const int* __restrict__ Seqlens,
    bf16* __restrict__ O,          // [bs,H,512] used iff splits==1
    float* __restrict__ MidO,      // [bs,splits,M,512]
    float* __restrict__ MidM, float* __restrict__ MidL,  // [bs,splits,M]
    float sm_scale, int block_size, int H, int splits, int max_blocks) {
  const int b = blockIdx.x, s = blockIdx.y;
  const int warp = threadIdx.x >> 5, tid = threadIdx.x, nthreads = WARPS * 32;
  const int seqlen = Seqlens[b];
  constexpr int M = MTILES * 16;
  const int chunk = (seqlen + splits - 1) / splits;
  const int kv_start = s * chunk, kv_end = min(kv_start + chunk, seqlen);

  extern __shared__ char smem[];
  bf16*  Ksh = reinterpret_cast<bf16*>(smem);
  bf16*  Qsh = Ksh + NT * HD;
  float* Ssh = reinterpret_cast<float*>(Qsh + M * HD);
  bf16*  Psh = reinterpret_cast<bf16*>(Ssh + M * NT);
  float* Osh = reinterpret_cast<float*>(Psh + M * NT);
  float* msh = Osh + M * CKV;
  float* lsh = msh + M;
  float* ash = lsh + M;

  for (int i = tid; i < M * HD; i += nthreads) {
    int h = i / HD, d = i % HD;
    Qsh[i] = (h < H) ? Q[((long)b * H + h) * HD + d] : __float2bfloat16(0.f);
  }
  for (int i = tid; i < M * CKV; i += nthreads) Osh[i] = 0.f;
  if (tid < M) { msh[tid] = -INFINITY; lsh[tid] = 0.f; }
  __syncthreads();

  const float scale = sm_scale * LOG2E;
  const int* bt = BlockTable + (long)b * max_blocks;

  for (int base = kv_start; base < kv_end; base += NT) {
    const int valid = min(NT, kv_end - base);
    for (int v = tid; v < NT * (HD / 8); v += nthreads) {
      const int j = v / (HD / 8), dd = (v % (HD / 8)) * 8;
      if (j < valid) {
        const int n = base + j, page = bt[n / block_size];
        *reinterpret_cast<int4*>(Ksh + j * HD + dd) =
            *reinterpret_cast<const int4*>(Latent + (long)(page * block_size + n % block_size) * HD + dd);
      } else {
        *reinterpret_cast<int4*>(Ksh + j * HD + dd) = make_int4(0, 0, 0, 0);
      }
    }
    __syncthreads();

    if (warp * 16 < NT) {
      for (int mt = 0; mt < MTILES; ++mt) {
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
        wmma::fill_fragment(c, 0.f);
        for (int kt = 0; kt < HD / 16; ++kt) {
          wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a;
          wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::col_major> bf;
          wmma::load_matrix_sync(a, Qsh + mt * 16 * HD + kt * 16, HD);
          wmma::load_matrix_sync(bf, Ksh + (warp * 16) * HD + kt * 16, HD);
          wmma::mma_sync(c, a, bf, c);
        }
        for (int i = 0; i < c.num_elements; ++i) c.x[i] *= scale;
        wmma::store_matrix_sync(Ssh + mt * 16 * NT + warp * 16, c, NT, wmma::mem_row_major);
      }
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
    for (int i = tid; i < M * CKV; i += nthreads) Osh[i] *= ash[i / CKV];
    __syncthreads();

    for (int mt = 0; mt < MTILES; ++mt) {
      for (int nb = 0; nb < CKV / (WARPS * 16); ++nb) {
        const int dim0 = warp * (CKV / WARPS) + nb * 16;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
        wmma::load_matrix_sync(c, Osh + mt * 16 * CKV + dim0, CKV, wmma::mem_row_major);
        for (int kt = 0; kt < NT / 16; ++kt) {
          wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a;
          wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> bf;
          wmma::load_matrix_sync(a, Psh + mt * 16 * NT + kt * 16, NT);
          wmma::load_matrix_sync(bf, Ksh + (kt * 16) * HD + dim0, HD);
          wmma::mma_sync(c, a, bf, c);
        }
        wmma::store_matrix_sync(Osh + mt * 16 * CKV + dim0, c, CKV, wmma::mem_row_major);
      }
    }
    __syncthreads();
  }

  const bool empty = (kv_end <= kv_start);
  if (splits == 1) {
    for (int i = tid; i < M * CKV; i += nthreads) {
      int h = i / CKV, d = i % CKV;
      if (h < H) O[((long)b * H + h) * CKV + d] = __float2bfloat16(empty ? 0.f : Osh[i] / lsh[h]);
    }
  } else {
    float* mo = MidO + (((long)b * splits + s) * M) * CKV;
    for (int i = tid; i < M * CKV; i += nthreads) mo[i] = empty ? 0.f : Osh[i];
    if (tid < M) {
      MidM[((long)b * splits + s) * M + tid] = empty ? -INFINITY : msh[tid];
      MidL[((long)b * splits + s) * M + tid] = empty ? 0.f : lsh[tid];
    }
  }
}

// stage-2: merge splits for one (batch, head). grid=(bs,H), blockDim=CKV/.. threads.
__global__ void mla_tc_stage2(
    const float* __restrict__ MidO, const float* __restrict__ MidM,
    const float* __restrict__ MidL, bf16* __restrict__ O,
    int H, int M, int splits) {
  const int b = blockIdx.x, h = blockIdx.y, d = threadIdx.x;  // d in [0,512)
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
  return (size_t)NT * HD * 2 + M * HD * 2 + M * NT * 4 + M * NT * 2 + M * CKV * 4 + 3 * M * 4;
}
static int g_sm = 0;

void mla_decode_tc(torch::Tensor q, torch::Tensor latent, torch::Tensor block_table,
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

  torch::Tensor midO, midM, midL;
  float *pO = nullptr, *pM = nullptr, *pL = nullptr;
  if (splits > 1) {
    auto opt = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());
    midO = torch::empty({bs, (long)splits, M, CKV}, opt);
    midM = torch::empty({bs, (long)splits, M}, opt);
    midL = torch::empty({bs, (long)splits, M}, opt);
    pO = midO.data_ptr<float>(); pM = midM.data_ptr<float>(); pL = midL.data_ptr<float>();
  }
#define LAUNCH(MT)                                                                    \
  cudaFuncSetAttribute(mla_tc_stage1<MT>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm); \
  mla_tc_stage1<MT><<<grid, block, sm, stream>>>(Q, L, BT, SL, OO, pO, pM, pL,         \
      (float)sm_scale, (int)block_size, H, (int)splits, max_blocks)
  if (MTILES == 1) { LAUNCH(1); } else { LAUNCH(2); }
#undef LAUNCH
  if (splits > 1)
    mla_tc_stage2<<<dim3(bs, H), dim3(CKV), 0, stream>>>(pO, pM, pL, OO, H, M, (int)splits);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mla_decode_tc", &mla_decode_tc, "tensor-core block-table MLA decode (split-KV)");
}
