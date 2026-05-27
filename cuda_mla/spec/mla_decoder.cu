// MLADecoder: flashinfer-style init/plan/run for ragged-batch MLA decode,
// general across batch size and block size.
//
//   __init__(bs, H, block_size, max_blocks, ...)  -- ALLOCATE once + fix geometry.
//        A bs-aware policy sets the schedule knobs: target #active CTAs (one
//        MINB=3 wave ~ 3*SM, so LOW bs auto-gets many splits/request and HIGH bs
//        gets few), a chunk_min floor (avoid tiny-chunk overhead), a per-request
//        split cap, and the MINB to launch run() with.
//   plan(seqlens)  -- POPULATE the load-balanced work queue from live seqlens.
//   run(q, latent, block_table, o, sm_scale)  -- EXECUTE; dispatches the decode
//        kernel by (block_size, MINB). No seqlens => one plan() feeds all layers.
//
// Both plan() and run() are fixed-grid launches on the current stream with the
// pre-allocated buffers => both CUDA-graph-capturable.
#include "mla_ldm.cuh"
#include <torch/extension.h>
#include <algorithm>

struct MLADecoder {
  int bs = 0, H = 0, block_size = 0, max_blocks = 0;
  int target = 0, target_ctas = 0, max_split_cap = 0, chunk_min = 0, minb = 3;
  int MTILES = 0, M = 0, sm_count = 0;
  torch::Tensor work_batch, work_kv_start, work_kv_end, work_offset, mid_o, mid_m, mid_l;

  // ---- init: bs-aware schedule policy + allocation. Negative knob args => auto. ----
  MLADecoder(int bs_, int H_, int block_size_, int max_blocks_,
             int max_split_cap_ = -1, int chunk_min_ = -1, int minb_ = -1) {
    bs = bs_; H = H_; block_size = block_size_; max_blocks = max_blocks_;
    MTILES = (H + 15) / 16; M = MTILES * 16;
    int dev; cudaGetDevice(&dev);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
    sm_count = prop.multiProcessorCount;

    // Achievable CTAs/SM is set by smem (the run_wq footprint, STAGES=2/NT=16), which
    // scales with M = MTILES*16: H<=16 (M=16) => ~55KB => 4 CTAs/SM; H<=32 (M=32) =>
    // ~74KB => 3. We fill exactly one such wave: target active CTAs = ctas*SM, and
    // MINB=ctas (launch_bounds forces that occupancy; the small-M Oreg keeps it spill-
    // free). So H=16 auto-uses 4 CTAs/splits~4, H=20 uses 3 CTAs/splits~3.
    int smem_cta = 2 * 16 * ldm::HDP * 2 + M * ldm::HDP * 2 + M * 16 * 2 + 3 * M * 4;
    int smem_sm = (int)prop.sharedMemPerMultiprocessor;
    int ctas = std::max(1, std::min(6, smem_sm / smem_cta));
    minb = (minb_ > 0) ? minb_ : ctas;                        // CTAs/SM occupancy target
    target = ctas * sm_count;                                 // active CTAs to fill one wave
    chunk_min = (chunk_min_ > 0) ? chunk_min_ : 128;          // don't split below 128 tokens
    // per-request cap: enough for low bs to fill a wave, bounded so one request can't
    // starve others on skew. ~ceil(target/bs) headroom, clamped to [ctas, target].
    int auto_cap = std::min(target, std::max(ctas, 2 * ((target + bs - 1) / bs)));
    max_split_cap = (max_split_cap_ > 0) ? max_split_cap_ : auto_cap;
    // queue length: safe upper bound on sum(nsplits) (rounding + the >=1 clamp).
    target_ctas = std::max(target, bs) + bs;

    auto i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    work_batch = torch::empty({target_ctas}, i32);
    work_kv_start = torch::empty({target_ctas}, i32);
    work_kv_end = torch::empty({target_ctas}, i32);
    work_offset = torch::empty({bs + 1}, i32);
    mid_o = torch::empty({target_ctas, M, ldm::CKV}, f32);
    mid_m = torch::empty({target_ctas, M}, f32);
    mid_l = torch::empty({target_ctas, M}, f32);
  }

  // ---- plan: populate the work queue from current seqlens. ----
  void plan(torch::Tensor seqlens) {
    TORCH_CHECK(seqlens.size(0) == bs, "plan: seqlens batch ", seqlens.size(0), " != init bs ", bs);
    ldm::run_schedule_wq(seqlens, work_batch, work_kv_start, work_kv_end, work_offset,
                         target, max_split_cap, chunk_min);
  }

  // ---- run: dispatch the decode kernel by (block_size, MINB). ----
  void run(torch::Tensor q, torch::Tensor latent, torch::Tensor block_table,
           torch::Tensor o, double sm_scale) {
#define RUN(BLK, MB) ldm::run_wq<BLK, 16, 4, 2, 1, MB>(q, latent, block_table, o, work_batch, \
        work_kv_start, work_kv_end, work_offset, mid_o, mid_m, mid_l, sm_scale)
// MINB is the occupancy target chosen in init from M (3 for H<=32, 4 for H<=16).
#define DISPATCH_MB(BLK) do { \
    if (minb <= 2) { RUN(BLK, 2); } else if (minb == 3) { RUN(BLK, 3); } \
    else if (minb == 4) { RUN(BLK, 4); } else { RUN(BLK, 5); } } while (0)
    if (block_size == 64) { DISPATCH_MB(64); }
    else if (block_size == 32) { DISPATCH_MB(32); }
    else if (block_size == 16) { DISPATCH_MB(16); }
    else TORCH_CHECK(false, "MLADecoder: unsupported block_size ", block_size, " (need 16/32/64)");
#undef DISPATCH_MB
#undef RUN
  }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind11::class_<MLADecoder>(m, "MLADecoder")
      .def(pybind11::init<int, int, int, int, int, int, int>(),
           pybind11::arg("bs"), pybind11::arg("H"), pybind11::arg("block_size"),
           pybind11::arg("max_blocks"), pybind11::arg("max_split_cap") = -1,
           pybind11::arg("chunk_min") = -1, pybind11::arg("minb") = -1)
      .def("plan", &MLADecoder::plan, pybind11::arg("seqlens"))
      .def("run", &MLADecoder::run,
           pybind11::arg("q"), pybind11::arg("latent"), pybind11::arg("block_table"),
           pybind11::arg("o"), pybind11::arg("sm_scale"))
      .def_readonly("target", &MLADecoder::target)
      .def_readonly("target_ctas", &MLADecoder::target_ctas)
      .def_readonly("max_split_cap", &MLADecoder::max_split_cap)
      .def_readonly("chunk_min", &MLADecoder::chunk_min)
      .def_readonly("minb", &MLADecoder::minb);
}
