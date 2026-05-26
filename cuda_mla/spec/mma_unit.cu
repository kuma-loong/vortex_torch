// Unit test for the ldmatrix + mma.sync.m16n8k16 fragment layout.
// One warp loads A[16,16], B[16,16] (row-major bf16) from smem via ldmatrix,
// runs the m16n16k16 (two n8) mma, scatters C[16,16] with the assumed layout.
// mma.row.col computes D[m,n] = sum_k A[m,k]*B[n,k] = (A @ B^T).
//   transB=0 (B via ldm_x4)       -> expect A @ B^T
//   transB=1 (B via ldm_x4_trans) -> expect A @ B   (trans makes B[n,k]=Braw[k,n])
#include <torch/extension.h>
#include <cuda_bf16.h>
typedef __nv_bfloat16 bf16;

__device__ __forceinline__ void ldm_x4(uint32_t* R, const bf16* p) {
  uint32_t s = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(R[0]), "=r"(R[1]), "=r"(R[2]), "=r"(R[3]) : "r"(s));
}
__device__ __forceinline__ void ldm_x4_trans(uint32_t* R, const bf16* p) {
  uint32_t s = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.trans.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(R[0]), "=r"(R[1]), "=r"(R[2]), "=r"(R[3]) : "r"(s));
}
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

__global__ void k(const bf16* A, const bf16* B, float* C, int transB, int swapB) {
  __shared__ __align__(16) bf16 As[256], Bs[256];
  int t = threadIdx.x;
  for (int i = t; i < 256; i += 32) { As[i] = A[i]; Bs[i] = B[i]; }
  __syncwarp();
  int lane = t, qrow = lane % 16, qcol = (lane / 16) * 8;
  uint32_t fa[4], fb[4];
  ldm_x4(fa, As + qrow * 16 + qcol);
  if (transB) ldm_x4_trans(fb, Bs + qrow * 16 + qcol);
  else        ldm_x4(fb, Bs + qrow * 16 + qcol);
  if (swapB) { uint32_t tmp = fb[1]; fb[1] = fb[2]; fb[2] = tmp; }
  float c[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  mma16(c, fa, fb);
  int er0 = lane / 4, ec = (lane % 4) * 2;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    int row = er0 + ((i & 2) ? 8 : 0);
    int col = ec + (i & 1) + ((i >= 4) ? 8 : 0);
    C[row * 16 + col] = c[i];
  }
}

torch::Tensor run(torch::Tensor A, torch::Tensor B, int64_t transB, int64_t swapB) {
  auto C = torch::zeros({16, 16}, torch::TensorOptions().dtype(torch::kFloat32).device(A.device()));
  k<<<1, 32>>>((const bf16*)A.data_ptr(), (const bf16*)B.data_ptr(), C.data_ptr<float>(),
               (int)transB, (int)swapB);
  return C;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
