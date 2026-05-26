// Experiment: H=20 bs128 blk64, vary NACC (GEMM1 accumulators for MMA-latency ILP).
#include "mla_ldm.cuh"
#define V(nm, NACC) \
  void nm(torch::Tensor q, torch::Tensor l, torch::Tensor bt, torch::Tensor sl, \
          torch::Tensor o, double s, int sp) { ldm::launch<64,16,4,2,NACC>(q,l,bt,sl,o,s,sp); }
V(acc1,1) V(acc2,2) V(acc3,3) V(acc4,4) V(acc6,6)
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("acc1",&acc1); m.def("acc2",&acc2); m.def("acc3",&acc3); m.def("acc4",&acc4); m.def("acc6",&acc6);
}
