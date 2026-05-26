// Full H=20 campaign: optimized kernel for blk in {16,32,64}, splits as arg.
#include "mla_ldm.cuh"
#define V(nm, BLK) \
  void nm(torch::Tensor q, torch::Tensor l, torch::Tensor bt, torch::Tensor sl, \
          torch::Tensor o, double s, int sp) { ldm::launch<BLK,16,4,2>(q,l,bt,sl,o,s,sp); }
V(b16,16) V(b32,32) V(b64,64)
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){ m.def("b16",&b16); m.def("b32",&b32); m.def("b64",&b64); }
