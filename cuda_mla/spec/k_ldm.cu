#include "mla_ldm.cuh"
#define V(nm, BLK, NT, NW) \
  void nm(torch::Tensor q, torch::Tensor l, torch::Tensor bt, torch::Tensor sl, \
          torch::Tensor o, double s, int sp) { ldm::launch<BLK, NT, NW>(q, l, bt, sl, o, s, sp); }
V(r_nt16_w4, 64, 16, 4) V(r_nt32_w4, 64, 32, 4)
V(r_nt16_w8, 64, 16, 8) V(r_nt32_w8, 64, 32, 8)
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("r_nt16_w4", &r_nt16_w4); m.def("r_nt32_w4", &r_nt32_w4);
  m.def("r_nt16_w8", &r_nt16_w8); m.def("r_nt32_w8", &r_nt32_w8);
}
