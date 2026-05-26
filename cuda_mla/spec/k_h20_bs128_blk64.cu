// H=20, bs=128, block_size=64. ldmatrix + reg-O + reg-softmax + cp.async + split-KV.
#include "mla_ldm.cuh"
#define V(nm, NW, ST) \
  void nm(torch::Tensor q, torch::Tensor l, torch::Tensor bt, torch::Tensor sl, \
          torch::Tensor o, double s, int sp) { ldm::launch<64,16,NW,ST>(q,l,bt,sl,o,s,sp); }
V(r_w4_s1,4,1) V(r_w4_s2,4,2) V(r_w8_s1,8,1) V(r_w8_s2,8,2)
void run(torch::Tensor q, torch::Tensor l, torch::Tensor bt, torch::Tensor sl,
         torch::Tensor o, double s) { ldm::launch<64,16,4,2>(q,l,bt,sl,o,s,2); }
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run",&run); m.def("r_w4_s1",&r_w4_s1); m.def("r_w4_s2",&r_w4_s2);
  m.def("r_w8_s1",&r_w8_s1); m.def("r_w8_s2",&r_w8_s2);
}
