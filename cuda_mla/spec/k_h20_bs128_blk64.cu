// H=20, bs=128, block_size=64. ldmatrix + bf16-packed reg-O + reg-softmax +
// cp.async + split-KV. WINNING CONFIG (ncu-tuned): NWARPS=4, STAGES=2, MINB=3
// (3 CTAs/SM => 18.75% occ, the occupancy wall the bandwidth-bound decode hit),
// splits=3 (populates the 3-block capacity). 2059 GB/s @ sel=2048 vs Triton 1971.
#include "mla_ldm.cuh"
#define V(nm, NW, ST, MB) \
  void nm(torch::Tensor q, torch::Tensor l, torch::Tensor bt, torch::Tensor sl, \
          torch::Tensor o, double s, int sp) { ldm::launch<64,16,NW,ST,1,MB>(q,l,bt,sl,o,s,sp); }
V(r_w4_s2_b2,4,2,2) V(r_w4_s2_b3,4,2,3) V(r_w8_s2_b2,8,2,2) V(r_w8_s2_b3,8,2,3)
// default: the measured optimum for h20/bs128/blk64 (MINB=3, splits=3).
void run(torch::Tensor q, torch::Tensor l, torch::Tensor bt, torch::Tensor sl,
         torch::Tensor o, double s) { ldm::launch<64,16,4,2,1,3>(q,l,bt,sl,o,s,3); }
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run",&run); m.def("r_w4_s2_b2",&r_w4_s2_b2); m.def("r_w4_s2_b3",&r_w4_s2_b3);
  m.def("r_w8_s2_b2",&r_w8_s2_b2); m.def("r_w8_s2_b3",&r_w8_s2_b3);
}
