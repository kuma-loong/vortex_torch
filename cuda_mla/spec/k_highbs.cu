// High-bs specialized: NT=64 (big tiles), single-pass (splits=1) or low splits, deep cp.async.
#include "mla_ldm.cuh"
#define V(nm,BLK,NW,ST) \
  void nm(torch::Tensor q, torch::Tensor l, torch::Tensor bt, torch::Tensor sl, \
          torch::Tensor o, double s, int sp){ ldm::launch<BLK,64,NW,ST>(q,l,bt,sl,o,s,sp); }
V(b32_w4_s2,32,4,2) V(b32_w8_s2,32,8,2) V(b64_w4_s2,64,4,2) V(b64_w8_s2,64,8,2)
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("b32_w4_s2",&b32_w4_s2); m.def("b32_w8_s2",&b32_w8_s2);
  m.def("b64_w4_s2",&b64_w4_s2); m.def("b64_w8_s2",&b64_w8_s2);
}
