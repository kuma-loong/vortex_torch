# Hand-written CUDA MLA-decode kernel for B200 (sm_100) — final report

A from-scratch CUDA kernel for the **vortex block-table MLA decode** workload,
built to compete with the existing Triton kernels (`triton_mla_kernel.py`) on a
**B200 (sm_100)**. Focus: **H = 20 query heads** (the real configuration);
flashinfer used only as a technique reference, not called.

Workload per request `b`: gather its `seqlen_b` selected latent rows via
`block_table` (page size = `block_size`), attend `H` heads against the fused
576-d latent (ckv=512 doubles as V, kpe=64 rope), write `o[b,H,512]`. Bandwidth
metric = useful HBM = `Σ_b seqlen_b · 576 · 2 bytes / time`. Peak ≈ 8 TB/s.

---

## TL;DR

A single kernel (`spec/mla_ldm.cuh`) — `ldmatrix` + `mma.sync.m16n16k16` +
register-resident O + register-resident online softmax + a shared-memory
bank-conflict pad + split-KV with a wave-aware split heuristic — that is
**correct for any seqlen (incl. ragged/partial)** and, vs Triton (best of
`kv_split_opt`/`single_pass_opt`) at H=20:

- **Wins 54 / 69 (bs × block) cells** at sel=2048 — by **2–5×** at low batch,
  **1.2–1.6×** at mid batch.
- **Wins all 23 block=16 cells** (Triton degrades at small blocks; this kernel
  is block-size-invariant).
- **Parity (0.93–0.99×)** only in the high-batch (bs≥64) × large-block (32/64)
  corner.
- Holds across **sel ∈ {1024, 2048, 4096}** and **non-multiple / ragged** seqlens.

---

## Kernel design (`spec/mla_ldm.cuh`)

FlashAttention-decode, split-KV (grid `(bs, splits)`, stage-2 merge):

- **GEMM1** `S = Q·Kᵀ` and **GEMM2** `O += P·V` via **`mma.sync.m16n8k16` (bf16)**,
  loaded with **`ldmatrix.x4`** (and `ldmatrix.x4.trans` for V). Fragment layout
  was pinned with a standalone unit test (`spec/mma_unit.cu`): GEMM1 loads K
  non-transposed with a `B[1]↔B[2]` swap; GEMM2 loads V transposed, no swap.
- **Register-resident O accumulator** (fp32) — `mma.sync`'s known C-fragment
  layout lets O live in registers (no shared-memory round-trip), which `wmma`
  could not.
- **Register-resident online softmax** — for NT=16 a warp owns its head-tile's
  full token range, so row max/sum is a within-warp `__shfl_xor` reduction; no S
  written to shared memory. (Generalized to `NTOK16 = NT/16` token-tiles.)
- **Shared-memory stride pad 576 → 584** — `ldmatrix`'s 16 rows at stride 576
  all alias the same banks (16-way conflict); padding to 584 → 2-way. **This one
  change was +135%** (the single biggest lever).
- **`cp.async` double-buffered** K tiles; **fp32** mid-buffer (`MidO`) — matching
  flashinfer's fp32 cross-block accumulator (bf16 there was tried, measured
  slower in this occupancy-bound regime).
- **Heads padded to a multiple of 16** (H=20 → MTILES=2), padded rows masked at
  store.

### Wave-aware split selection (`spec/wave_splits.py`)
`splits = clamp(⌊2·SM_count / bs⌋, 1, 32)` — host-side, cuda-graph-safe (uses
`bs` + SM count only). Derived from a fine sweep showing the optimum is **one
wave of ~2 CTAs/SM** (`bs·splits ≈ 296`); it matched the per-bs fine-sweep
optimum exactly for every bs ≥ 32 and removed the wave-quantization dips
(e.g. bs=80: 1266 → 1614, bs=96: 1436 → 1917).

---

## Optimization journey (flagship: H=20, bs=128, blk=64, sel=2048)

| step | GB/s | % of Triton (1944) |
|---|---|---|
| scalar warp-per-head (FP32 GEMV) | 442 | 23% |
| `ldmatrix` + `mma.sync.m16n16k16` | 567 | 29% |
| **+ smem stride pad (kill 16-way bank conflict)** | **1330** | 68% |
| **+ register-resident online softmax** | **1898** | **98%** |

Negatives measured and discarded (the kernel sits at a sharp optimum): deeper
`cp.async` pipeline, STAGES=1, full XOR swizzle (correct but slower — not
bank-conflict-bound after the pad), P-load hoist, GEMM1 K-split across idle
warps, multi-accumulator, warp count, bf16 mid-buffer, and NT=64 single-pass.

---

## Correctness

bf16-vs-fp32 reference, all configs (bs, blk, sel, H, splits), **including
non-multiple seqlens (1000/1023/4000) and fully ragged batches** (per-request
seqlens incl. a 1-token request): **max error ≈ 1–2.3e-3** — the bf16 floor.
The kernel reads `Seqlens[b]` per request, clips each request's split range to
its own length, masks the partial tail, and guards page loads to `n < seqlen_b`.

---

## Benchmarks (H=20, vs Triton best)

**sel=2048, full bs sweep, wave-aware splits — 54/69 cells win:**
- blk=16: **23/23** win (1.2–3.0×).
- blk=32: 16/23 win (lose only bs∈{64,72,80,104,112,120,128}).
- blk=64: 15/23 win (lose only bs∈{64,72,80,88,104,112,120,128}).
- Low batch is decisive: bs=1 → **5.1×**, bs=8 → 2.3×, bs=16 → 1.9×.

**Other sequence lengths** (blk=64): sel=1024 → win 10/11; sel=4096 → win 9/11.
GB/s scales with seqlen (longer = better overhead amortization; bs=96 hits
~2230 GB/s = 28% peak at sel=4096).

---

## Honest limits / what's left

1. **High-batch × large-block parity.** At bs≥64, blk∈{32,64} the kernel is
   0.93–0.99× of Triton. Root cause (evidence: ~12 negative levers + profiling):
   the MLA score GEMM is M=20, K=576 — very low arithmetic intensity — and at
   ~24% of peak HBM the limiter is MMA issue, not memory. Triton wins there
   because `tl.dot` **distributes the score MMA across all warps**; this kernel's
   per-warp register-softmax can't without a cross-warp reduction (measured
   net-negative).
2. **`tcgen05` (Blackwell 5th-gen TC + tensor memory) is the wrong tool for
   H=20.** Its MMA requires **M ∈ {64,128}**, so 20 heads → 31% row utilization
   (worse than m16's 62.5%); and it needs the full CUTLASS sm100 collective.
3. **Ragged batches at high bs.** Correct, but uniform per-batch `splits` causes
   a load-imbalance tail (longest request dominates). Bumping splits recovers
   much of it (bs=64 ragged → 1.15× win); the full fix is a **work-balanced
   scheduler** (per-request `splits ∝ seqlen_b`, flashinfer-style) — not yet
   implemented.
4. **No `ncu`.** GPU performance counters are admin-locked in this container, so
   the high-batch stall was reasoned-about, not profiled. With `ncu` access the
   last 2–3% in the high-batch corner could likely be found precisely.

---

## Files

- `spec/mla_ldm.cuh` — the kernel (templated on `BLK, NT, NWARPS, STAGES, NACC`).
- `spec/wave_splits.py` — wave-aware split heuristic (deployable).
- `spec/mma_unit.cu` — `ldmatrix`/`mma` fragment-layout unit test.
- `spec/campaign.cu`, `spec/k_h20_bs128_blk{32,64}.cu`, `spec/k_highbs.cu`,
  `spec/k_acc.cu` — per-config instantiations / experiments.
- `PROGRESS.md` — full chronological log (every lever, every measurement).
- Older scalar baseline + flashinfer eval: `mla_decode.cu`, `bench_flashinfer.py`.

### Build / run
```bash
# all kernels compile via torch JIT; needs:
SGLANG_ENABLE_TORCH_COMPILE=0 PYTHONPATH=. CUDA_VISIBLE_DEVICES=<g> \
  TORCH_CUDA_ARCH_LIST=10.0 python <bench script>
# kernel build flags: -arch=sm_100a --use_fast_math
```
Build dirs (`build_*/`) are git-ignored (JIT artifacts).
