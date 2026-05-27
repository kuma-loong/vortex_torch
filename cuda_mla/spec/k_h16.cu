// H=16 standalone flagship (MTILES=1, no head padding). vs H=20: M=16 halves the Q
// smem (~18.7KB) and Oreg (32 regs) => ~55KB smem => 4 CTAs/SM (vs 3) => 44% DRAM.
// Optimum: NWARPS=4, STAGES=2, MINB=5, splits=4 (~3000-3023 GB/s @ bs128, 1.1-1.9x Triton).
// (MINB=5: smem caps occupancy at 4 CTAs, but the lower reg target schedules ~1% better
// than MINB=4; the tiny M=16 Oreg keeps it spill-free.) BLK template must match block_size.
#include "mla_ldm.cuh"
#define V(nm, BLK, MB) \
  void nm(torch::Tensor q, torch::Tensor l, torch::Tensor bt, torch::Tensor sl, \
          torch::Tensor o, double s, int sp) { ldm::launch<BLK,16,4,2,1,MB>(q,l,bt,sl,o,s,sp); }
V(b64_b4,64,4) V(b64_b5,64,5) V(b32_b4,32,4) V(b32_b5,32,5) V(b16_b4,16,4) V(b16_b5,16,5)
// defaults: the measured optimum per block size (MINB=5, splits=4).
void run64(torch::Tensor q,torch::Tensor l,torch::Tensor bt,torch::Tensor sl,torch::Tensor o,double s){ ldm::launch<64,16,4,2,1,5>(q,l,bt,sl,o,s,4); }
void run32(torch::Tensor q,torch::Tensor l,torch::Tensor bt,torch::Tensor sl,torch::Tensor o,double s){ ldm::launch<32,16,4,2,1,5>(q,l,bt,sl,o,s,4); }
void run16(torch::Tensor q,torch::Tensor l,torch::Tensor bt,torch::Tensor sl,torch::Tensor o,double s){ ldm::launch<16,16,4,2,1,5>(q,l,bt,sl,o,s,4); }
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run64",&run64); m.def("run32",&run32); m.def("run16",&run16);
  m.def("b64_b4",&b64_b4); m.def("b64_b5",&b64_b5); m.def("b32_b4",&b32_b4);
  m.def("b32_b5",&b32_b5); m.def("b16_b4",&b16_b4); m.def("b16_b5",&b16_b5);
}
