// K2: H=20, bs=128, block_size=32. Inherits K1's optimized kernel (ldmatrix +
// mma.sync.m16n16k16 + register-O + register softmax + 576->584 pad), BLK=32.
#include "mla_ldm.cuh"
#define V(nm, NW, ST) \
  void nm(torch::Tensor q, torch::Tensor l, torch::Tensor bt, torch::Tensor sl, \
          torch::Tensor o, double s, int sp) { ldm::launch<32,16,NW,ST>(q,l,bt,sl,o,s,sp); }
V(r_w4_s2,4,2) V(r_w4_s3,4,3) V(r_w8_s2,8,2)
void run(torch::Tensor q, torch::Tensor l, torch::Tensor bt, torch::Tensor sl,
         torch::Tensor o, double s) { ldm::launch<32,16,4,2>(q,l,bt,sl,o,s,4); }
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("run",&run); m.def("r_w4_s2",&r_w4_s2); m.def("r_w4_s3",&r_w4_s3); m.def("r_w8_s2",&r_w8_s2);
}
