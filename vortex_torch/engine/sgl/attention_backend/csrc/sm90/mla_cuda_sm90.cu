// SM90 arch-only copy of the hand-written CUDA block-table MLA decode kernel.
// Keep algorithm and launch parameters matched to the original until the
// arch-only benchmark is complete.
//
// Exposes two interfaces over the same kernel (mla_ldm_sm90.cuh: ldmatrix + bf16-packed
// register-O + register softmax + split-KV):
//
//   decode(...)          -- stateless drop-in for the Triton decode_blocktable_mla.
//   class MLADecoder     -- flashinfer-style init / plan / run for the backend:
//        __init__(bs,H,block_size,max_blocks,...)  ALLOCATE the work-queue + split
//             scratch once and fix the launch geometry (occupancy target MINB and
//             active-CTA target derived from M = ceil(H/16)*16 via smem footprint).
//        plan(seqlens)   POPULATE a load-balanced work queue from the (sparse)
//             seqlens — ONE plan() per decode step, shared across all layers.
//        run(q,latent,block_table,o,sm_scale)  the per-layer decode (stage1 over the
//             queue + stage2 reduction). No seqlens (plan owns the schedule).
//
// All launches are fixed-grid on the current stream with pre-allocated buffers =>
// plan() (eager, in init_forward_metadata) + run() (captured, in forward_decode)
// are CUDA-graph-safe: capture once per bs, replay as the seqlens change in place.
#include "mla_ldm_sm90.cuh"
#include <torch/extension.h>
#include <algorithm>

// ---- stateless drop-in (matches Triton decode_blocktable_mla contract) ---------
void decode(torch::Tensor q, torch::Tensor latent, torch::Tensor block_table,
            torch::Tensor seqlens, torch::Tensor o, double sm_scale,
            int64_t block_size, int64_t splits) {
  const int H = q.size(1);
  const int MTILES = (H + 15) / 16;
  const int sp = (int)splits;
#define CALL(BLK, MB) ldm::launch<BLK, 16, 4, 2, 1, MB>(q, latent, block_table, seqlens, o, sm_scale, sp)
  if (block_size == 64) { if (MTILES == 1) CALL(64, 5); else CALL(64, 3); }
  else if (block_size == 32) { if (MTILES == 1) CALL(32, 5); else CALL(32, 3); }
  else if (block_size == 16) { if (MTILES == 1) CALL(16, 5); else CALL(16, 3); }
  else TORCH_CHECK(false, "cuda_mla decode: block_size must be 16/32/64, got ", block_size);
#undef CALL
}

// ---- launch geometry (host-only; shared by the python allocator + the ctor) ----
// All of (target, minb, chunk_min, max_split_cap, M) depend only on (H, block_size,
// max_blocks) — constant across decode bs. Only `target_ctas` (the run_wq grid /
// work-queue length) grows with bs, and it is MONOTONE in bs:
//   target_ctas(bs) = min(target, bs*max_split_cap) + bs.
// => buffers sized at the max decode bs cover every smaller bs. The allocator
// (python) calls compute_geom(max_bs, ...) once to size the shared scratch; each
// per-bs decoder calls compute_geom(bs, ...) and slices that scratch to its grid.
struct MLAGeom { int target_ctas, M, minb, target, max_split_cap, chunk_min; };

static MLAGeom compute_geom(int bs, int H, int block_size, int max_blocks,
                            int max_split_cap_, int chunk_min_, int minb_) {
  MLAGeom g{};
  int MTILES = (H + 15) / 16; g.M = MTILES * 16;
  int dev; cudaGetDevice(&dev);
  cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
  int sm_count = prop.multiProcessorCount;

  // MINB (launch_bounds CTAs/SM occupancy) is set by smem (run_wq footprint,
  // STAGES=2/NT=16), scaling with M=MTILES*16: H<=16 (M=16) ~55KB => 4; else 3.
  int smem_cta = 2 * 16 * ldm::HDP * 2 + g.M * ldm::HDP * 2 + g.M * 16 * 2 + 3 * g.M * 4;
  int smem_sm = (int)prop.sharedMemPerMultiprocessor;
  int ctas = std::max(1, std::min(6, smem_sm / smem_cta));
  g.minb = (minb_ > 0) ? minb_ : ctas;
  // `target` = #active CTAs to fill one MINB-deep wave (=> ~target/bs splits per
  // request). Microbench (isolated kernel) shows MORE splits cut latency at LOW bs
  // (the decode tail, where attention dominates and the GPU is under-filled): at
  // bs=1, sparse=1024, splits 4->8 is 28.5->19.3us. The earlier "splits are slow
  // end-to-end" was the GRID-padding bug (target_ctas), not the split count — fixed
  // below. So fill ctas*SM (=> ~8 splits at bs=1, ~4 at bs=128), capped by chunk_min.
  g.target = ctas * sm_count;
  // chunk_min = min tokens/split. 64 (4 NT-tiles) lets the tail reach ~16 splits
  // (bs=1, 1024-tok budget: sp16=16.2us vs sp8=19.3us); the fill target caps splits
  // at higher bs so this only matters in the low-bs tail. (Smaller would over-split:
  // sp32=17.5us.)
  g.chunk_min = (chunk_min_ > 0) ? chunk_min_ : 64;
  // cap = max splits a request can take = ceil(max_sparse_seqlen / chunk_min), where
  // max_sparse_seqlen = max_blocks*block_size (the sparse budget). This also keeps
  // the grid tight (target_ctas below): for the RULER 1024-tok budget cap=8.
  int cap_from_chunk = std::max(1, (max_blocks * block_size + g.chunk_min - 1) / g.chunk_min);
  g.max_split_cap = (max_split_cap_ > 0) ? max_split_cap_ : cap_from_chunk;
  // queue length = the run_wq grid. Upper-bounds sum(nsplits) but stays TIGHT: sum
  // <= min(target, bs*cap) (proportional sums to target; each request <= cap), + bs
  // for the >=1-clamp/rounding margin. (The old max(target,bs)+bs left ~SM padding
  // CTAs at EVERY bs => at the bs=1 tail the grid was ~150 CTAs, ~146 no-ops.)
  g.target_ctas = std::min(g.target, bs * g.max_split_cap) + bs;
  return g;
}

// Exposed to python so the backend can size the shared scratch at the max decode bs
// WITHOUT allocating a decoder. Returns [target_ctas, M, minb, target, max_split_cap,
// chunk_min].
std::vector<int64_t> mla_decoder_geometry(int64_t bs, int64_t H, int64_t block_size,
                                          int64_t max_blocks, int64_t max_split_cap,
                                          int64_t chunk_min, int64_t minb) {
  MLAGeom g = compute_geom((int)bs, (int)H, (int)block_size, (int)max_blocks,
                           (int)max_split_cap, (int)chunk_min, (int)minb);
  return {g.target_ctas, g.M, g.minb, g.target, g.max_split_cap, g.chunk_min};
}

// ---- flashinfer-style plan/run decoder -----------------------------------------
struct MLADecoder {
  int bs = 0, H = 0, block_size = 0, max_blocks = 0;
  int target = 0, target_ctas = 0, max_split_cap = 0, chunk_min = 0, minb = 3;
  int MTILES = 0, M = 0;
  torch::Tensor work_batch, work_kv_start, work_kv_end, work_offset, mid_o, mid_m, mid_l;

  // The work-queue + split-reduction scratch is NO LONGER allocated here — the caller
  // (python) allocates it once at the max decode bs and passes it in. We slice each
  // buffer to this bs's geometry (a dim-0 prefix view => same base pointer, fixed
  // address => cuda-graph-capturable; decode runs one bs/step so sharing is safe).
  MLADecoder(int bs_, int H_, int block_size_, int max_blocks_,
             torch::Tensor work_batch_, torch::Tensor work_kv_start_, torch::Tensor work_kv_end_,
             torch::Tensor work_offset_, torch::Tensor mid_o_, torch::Tensor mid_m_,
             torch::Tensor mid_l_, int max_split_cap_ = -1, int chunk_min_ = -1, int minb_ = -1) {
    bs = bs_; H = H_; block_size = block_size_; max_blocks = max_blocks_;
    MTILES = (H + 15) / 16; M = MTILES * 16;
    MLAGeom g = compute_geom(bs, H, block_size, max_blocks, max_split_cap_, chunk_min_, minb_);
    target = g.target; minb = g.minb; chunk_min = g.chunk_min;
    max_split_cap = g.max_split_cap; target_ctas = g.target_ctas;

    TORCH_CHECK(work_batch_.dim() == 1 && work_batch_.size(0) >= target_ctas,
                "work_batch too small: ", work_batch_.size(0), " < target_ctas ", target_ctas);
    TORCH_CHECK(work_kv_start_.size(0) >= target_ctas && work_kv_end_.size(0) >= target_ctas,
                "work_kv_{start,end} too small for target_ctas ", target_ctas);
    TORCH_CHECK(work_offset_.size(0) >= bs + 1, "work_offset too small: ",
                work_offset_.size(0), " < bs+1 ", bs + 1);
    TORCH_CHECK(mid_o_.dim() == 3 && mid_o_.size(0) >= target_ctas && mid_o_.size(1) == M
                && mid_o_.size(2) == ldm::CKV,
                "mid_o shape mismatch: need [>=", target_ctas, ",", M, ",", ldm::CKV,
                "], got [", mid_o_.size(0), ",", mid_o_.size(1), ",", mid_o_.size(2), "]");
    TORCH_CHECK(mid_m_.size(0) >= target_ctas && mid_m_.size(1) == M
                && mid_l_.size(0) >= target_ctas && mid_l_.size(1) == M,
                "mid_{m,l} shape mismatch (need [>=", target_ctas, ",", M, "])");

    work_batch = work_batch_.narrow(0, 0, target_ctas);
    work_kv_start = work_kv_start_.narrow(0, 0, target_ctas);
    work_kv_end = work_kv_end_.narrow(0, 0, target_ctas);
    work_offset = work_offset_.narrow(0, 0, bs + 1);
    mid_o = mid_o_.narrow(0, 0, target_ctas);
    mid_m = mid_m_.narrow(0, 0, target_ctas);
    mid_l = mid_l_.narrow(0, 0, target_ctas);
  }

  void plan(torch::Tensor seqlens) {
    TORCH_CHECK(seqlens.size(0) == bs, "plan: seqlens batch ", seqlens.size(0), " != init bs ", bs);
    ldm::run_schedule_wq(seqlens, work_batch, work_kv_start, work_kv_end, work_offset,
                         target, max_split_cap, chunk_min);
  }

  void run(torch::Tensor q, torch::Tensor latent, torch::Tensor block_table,
           torch::Tensor o, double sm_scale) {
#define RUN(BLK, MB) ldm::run_wq<BLK, 16, 4, 2, 1, MB>(q, latent, block_table, o, work_batch, \
        work_kv_start, work_kv_end, work_offset, mid_o, mid_m, mid_l, sm_scale)
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
  m.def("decode", &decode, "block-table MLA decode (CUDA, stateless)");
  m.def("mla_decoder_geometry", &mla_decoder_geometry,
        "host-only launch geometry [target_ctas, M, minb, target, max_split_cap, chunk_min]",
        pybind11::arg("bs"), pybind11::arg("H"), pybind11::arg("block_size"),
        pybind11::arg("max_blocks"), pybind11::arg("max_split_cap") = -1,
        pybind11::arg("chunk_min") = -1, pybind11::arg("minb") = -1);
  pybind11::class_<MLADecoder>(m, "MLADecoder")
      .def(pybind11::init<int, int, int, int, torch::Tensor, torch::Tensor, torch::Tensor,
                          torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                          int, int, int>(),
           pybind11::arg("bs"), pybind11::arg("H"), pybind11::arg("block_size"),
           pybind11::arg("max_blocks"), pybind11::arg("work_batch"),
           pybind11::arg("work_kv_start"), pybind11::arg("work_kv_end"),
           pybind11::arg("work_offset"), pybind11::arg("mid_o"), pybind11::arg("mid_m"),
           pybind11::arg("mid_l"), pybind11::arg("max_split_cap") = -1,
           pybind11::arg("chunk_min") = -1, pybind11::arg("minb") = -1)
      .def("plan", &MLADecoder::plan, pybind11::arg("seqlens"))
      .def("run", &MLADecoder::run,
           pybind11::arg("q"), pybind11::arg("latent"), pybind11::arg("block_table"),
           pybind11::arg("o"), pybind11::arg("sm_scale"))
      .def_readonly("target_ctas", &MLADecoder::target_ctas)
      .def_readonly("minb", &MLADecoder::minb);
}
