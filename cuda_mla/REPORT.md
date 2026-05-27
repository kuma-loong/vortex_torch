# Hand-written CUDA MLA-decode kernel for B200 (sm_100) — report

A from-scratch CUDA kernel + a flashinfer-style `plan()/run()` decoder for the
**vortex block-table MLA decode** workload, built to beat the existing Triton
kernels (`triton_mla_kernel.py`) on a **B200 (sm_100)**. Covers **H = 20** (GLM-4.7-Flash)
and **H = 16**, fused 576-d latent (`ckv=512` doubles as V, `kpe=64` rope). Bandwidth
metric = useful HBM = `Σ_b seqlen_b · 576 · 2 B / time`; peak ≈ 8 TB/s.

**H=16 vs H=20 in one line:** H=16 → `MTILES=1` (no head padding), so M=16 halves the
Q smem and bf16 Oreg → **4 CTAs/SM (44% DRAM)** vs H=20's 3 (30%). The decoder auto-tunes
the occupancy target from M. Flagship bs128/sel2048: **H=16 ≈ 3000–3021 GB/s** (1.11–1.91×
Triton across blk), **H=20 ≈ 2229 GB/s** (1.08–2.42×). Detail in the "H=16" section below.

---

## TL;DR

- **Flagship h20/bs128/blk64, sel=2048: 2229 GB/s vs Triton 1963 (+13%)**; +19% at
  sel=4096. The prior session plateaued at 1898 (98% of Triton) and concluded the
  high-batch corner was unbeatable without `tcgen05`. With `ncu` available, that
  diagnosis was **wrong**: the kernel was **occupancy/latency-bound, not MMA-bound**,
  and a launch-bounds + bf16-accumulator + scheduling change beat Triton cleanly.
- **Wins across the entire bs × block grid** (table below): bs 1–128, blk 16/32/64,
  uniform **1.0–2.4×**, ragged **1.5–3.8×**. Only 2 cells dip to 0.97–0.98× (parity).
- A **`MLADecoder` (init/plan/run) class** load-balances ragged batches via a
  work-queue scheduler, is **bs-general**, and is **CUDA-graph-capturable** (one
  `plan()` drives every layer's `run()`).
- Correctness vs fp32 ref ≤ **8e-3** everywhere (incl. ragged / 1-token / non-multiple),
  well under the 3e-2 bar.

---

## Kernel design (`cuda_mla/spec/mla_ldm.cuh`)

FlashAttention-decode, split-KV (stage-1 partials → stage-2 merge):

- **GEMM1** `S=Q·Kᵀ` and **GEMM2** `O+=P·V` via **`mma.sync.m16n8k16` (bf16)**, loaded
  with **`ldmatrix.x4`** (+`.trans` for V). Fragment layout pinned with a unit test
  (`spec/mma_unit.cu`).
- **Register-resident online softmax** — for NT=16 a warp owns its head-tile's full
  token range, so row max/sum is a within-warp `__shfl` (no S smem round-trip).
- **bf16-packed O accumulator** (`__nv_bfloat162 Oreg[..][4]`, 64 regs vs fp32's 128) —
  halving O's register footprint is what gives ptxas headroom at the forced occupancy.
- **`__launch_bounds__(128, MINB=3)`** — forces **3 CTAs/SM** (ptxas reschedules to 168
  regs, *no spill*); the single biggest lever in this session.
- **`cp.async` double-buffered K tiles** (STAGES=2); **128-bit vectorized Q gather** and
  **vectorized (`bf162`/`float2`) O/MidO epilogue store**.
- **smem stride pad 576→584** kills `ldmatrix`'s 16-way bank conflict (a +135% lever
  historically).

### The turn (once `ncu` was available)
At the 1898 plateau the kernel ran at **12.5% occupancy** (2 CTAs/SM, capped by 254 regs
*and* 76 KB smem), **1.77 active warps/scheduler**, **70% no-eligible cycles**, **DRAM 26%**
— pure latency/occupancy starvation, and Triton sat in the *same* hole. Forcing **3 CTAs/SM**
+ **bf16-packed O** + **splits=3** lifted occupancy to 18.75%, DRAM to 30%, duration 156→133 µs.

### Optimization journey (flagship h20/bs128/blk64, sel=2048)
| step | GB/s | vs Triton |
|---|---|---|
| scalar warp-per-head (fp32 GEMV) | 442 | 0.23× |
| `ldmatrix` + `mma.sync` | 567 | 0.29× |
| + smem pad (kill 16-way bank conflict) | 1330 | 0.68× |
| + register-resident softmax | 1898 | 0.98× |
| + **MINB=3 launch_bounds + splits=3** (ncu: occupancy) | 2031 | 1.04× |
| + **bf16-packed O accumulator** | 2059 | 1.06× |
| + **vectorized Q gather + O/MidO store** | **2229** | **1.13×** |

---

## `MLADecoder` — plan/run for ragged batches (`spec/mla_decoder.cu`)

Uniform per-request splits let the *longest* request serialize the wave on ragged
batches. Fixed with a flashinfer-style load-balanced **work queue**, split so one
`plan()` drives every layer's `run()`:

- **`__init__(bs, H, block_size, max_blocks, …)`** — allocate the work queue
  (`work_batch/kv_start/kv_end[target_ctas]`, `work_offset[bs+1]`) + split scratch
  (`mid_o/m/l`) once. A **bs-general policy** sets `target = 3·SM` active CTAs (one
  3-CTA wave), so **low bs auto-gets many splits/request** and **high bs gets ~1**; a
  `chunk_min` floor avoids tiny-chunk overhead.
- **`plan(seqlens)`** — one parallel `schedule_wq` kernel cuts each request into
  balanced equal-size KV chunks packed into the queue (**5.4 µs**; a serial pack was 40 µs).
- **`run(q, latent, block_table, o, sm_scale)`** — `stage1` over the flat queue (1-D grid,
  ~no idle CTAs, balanced regardless of skew) + `stage2_wq` (reduces each request's own
  chunks). Takes **no seqlens** — `plan()` owns the schedule.
- **CUDA graph**: both are fixed-grid launches, no host sync. Verified — capture
  `[plan(); run()×N_layers]`, change seqlens in place, replay → correct (≤7e-3).

Ragged wins (bs=128, blk=64): ragged-2× **1.5×**, heavy-skew (16 long, 112 short) **3.8×**
vs Triton single-pass (which is dominated by its longest request).

---

## H=16 (MTILES=1) — the decoder auto-tunes occupancy from head count

H=16 makes `MTILES = ceil(16/16) = 1` (vs 2 for H=20) → **no head padding**. M=16 halves
both the Q smem (37→18.7KB) and the bf16 O accumulator (64→32 regs), so smem/CTA ≈ 55KB →
**4 CTAs/SM** (smem-limited) at 25% occupancy and **DRAM 44%** — well above H=20's 3 CTAs/30%.
GEMM1 then runs on only 1 warp, but the extra occupancy hides it. Optimum: NWARPS=4, STAGES=2,
**MINB=5, splits=4**.

Two changes made this automatic and general:
- **`MLADecoder` policy**: `__init__` computes achievable CTAs/SM from the run-queue smem
  footprint (M-dependent) → `minb = ctas`, `target = ctas·SM`. So H=16 auto-uses 4 CTAs / ~4
  splits, H=20 uses 3 / ~3 — no per-head tuning.
- **Floor (not round) the proportional split** in the scheduler: total active CTAs must stay
  *just under* one wave; overshooting costs a 2nd-wave tail far worse than idle SMs. (round
  put H=16/bs128 on 5 splits = a LOSS ~2450; floor → 4 splits = 2934.) This also turned the
  H=20 bs112/120 sawtooth *losses* (0.97–0.98×) into 1.08× wins.

**H=16 bs128, sel=2048 (clean B200, GB/s, ratio vs Triton best):**
| block | standalone | decoder | Triton best | decoder ratio |
|---|---|---|---|---|
| 16 | 3021 | 2912 | 1574 (ks) | **1.85×** |
| 32 | 3015 | 2928 | 2472 (ks) | **1.18×** |
| 64 | 3023 | 2934 | 2722 (sp) | **1.08×** |

(The work-queue decoder runs ~3% under the 2D standalone peak — plan + flat-grid overhead,
amortized across layers. Triton collapses at blk=16.)

---

## Full benchmark table — H=20 (sel=2048 uniform, one empty B200; GB/s, ratio vs Triton best)

`mine` = `MLADecoder` (plan+run). `tri_sp` = single_pass_opt, `tri_ks` = kv_split_opt.

### block_size = 16
| bs | mine | tri_sp | tri_ks | ratio |   | bs | mine | tri_sp | tri_ks | ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 86 | 7 | 45 | 1.89× |   | 48 | 1670 | 292 | 830 | 2.01× |
| 2 | 165 | 14 | 90 | 1.83× |   | 56 | 1417 | 340 | 892 | 1.59× |
| 3 | 247 | 21 | 146 | 1.69× |   | 64 | 1433 | 389 | 812 | 1.76× |
| 4 | 339 | 28 | 180 | 1.89× |   | 72 | 1960 | 437 | 902 | 2.17× |
| 5 | 408 | 35 | 226 | 1.81× |   | 80 | 1570 | 485 | 933 | 1.68× |
| 6 | 486 | 41 | 266 | 1.83× |   | 88 | 2123 | 533 | 872 | 2.43× |
| 7 | 483 | 48 | 326 | 1.48× |   | 96 | 1660 | 582 | 933 | 1.78× |
| 8 | 547 | 55 | 345 | 1.59× |   | 104 | 2200 | 630 | 1236 | 1.78× |
| 16 | 907 | 110 | 695 | 1.30× |   | 112 | 1685 | 678 | 887 | 1.90× |
| 24 | 1286 | 165 | 789 | 1.63× |   | 120 | 1782 | 726 | 843 | 2.11× |
| 32 | 1107 | 218 | 786 | 1.41× |   | 128 | 2099 | 774 | 870 | 2.41× |
| 40 | 1553 | 248 | 815 | 1.91× |   |    |      |     |     |       |

### block_size = 32
| bs | mine | tri_sp | tri_ks | ratio |   | bs | mine | tri_sp | tri_ks | ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 86 | 11 | 45 | 1.92× |   | 48 | 1678 | 505 | 1226 | 1.37× |
| 2 | 169 | 23 | 90 | 1.89× |   | 56 | 1419 | 588 | 1261 | 1.12× |
| 3 | 253 | 34 | 135 | 1.88× |   | 64 | 1438 | 673 | 1253 | 1.15× |
| 4 | 327 | 45 | 177 | 1.84× |   | 72 | 1985 | 755 | 1374 | 1.44× |
| 5 | 407 | 56 | 225 | 1.81× |   | 80 | 1587 | 838 | 1251 | 1.27× |
| 6 | 489 | 67 | 267 | 1.83× |   | 88 | 2132 | 923 | 1344 | 1.59× |
| 7 | 482 | 79 | 307 | 1.57× |   | 96 | 1672 | 1005 | 1450 | 1.15× |
| 8 | 548 | 90 | 357 | 1.53× |   | 104 | 2191 | 1086 | 1112 | 1.97× |
| 16 | 908 | 180 | 712 | 1.28× |   | 112 | 1696 | 1168 | 1199 | 1.41× |
| 24 | 1282 | 269 | 1023 | 1.25× |   | 120 | 1790 | 1250 | 1277 | 1.40× |
| 32 | 1119 | 359 | 1100 | 1.02× |   | 128 | 2109 | 1335 | 1359 | 1.55× |
| 40 | 1569 | 426 | 1172 | 1.34× |   |    |      |     |     |       |

### block_size = 64
| bs | mine | tri_sp | tri_ks | ratio |   | bs | mine | tri_sp | tri_ks | ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 86 | 18 | 45 | 1.92× |   | 48 | 1684 | 761 | 1048 | 1.61× |
| 2 | 170 | 35 | 89 | 1.91× |   | 56 | 1417 | 885 | 1101 | 1.29× |
| 3 | 247 | 53 | 141 | 1.75× |   | 64 | 1446 | 1008 | 1099 | 1.32× |
| 4 | 326 | 70 | 178 | 1.83× |   | 72 | 1982 | 1132 | 1209 | 1.64× |
| 5 | 422 | 88 | 224 | 1.88× |   | 80 | 1587 | 1253 | 1060 | 1.27× |
| 6 | 489 | 105 | 263 | 1.86× |   | 88 | 2135 | 1372 | 1140 | 1.56× |
| 7 | 495 | 123 | 264 | 1.87× |   | 96 | 1678 | 1491 | 1229 | 1.13× |
| 8 | 548 | 140 | 293 | 1.87× |   | 104 | 2201 | 1608 | 1223 | 1.37× |
| 16 | 906 | 281 | 678 | 1.34× |   | 112 | 1695 | 1726 | 1026 | 0.98× |
| 24 | 1287 | 420 | 881 | 1.46× |   | 120 | 1794 | 1844 | 1084 | 0.97× |
| 32 | 1147 | 559 | 937 | 1.22× |   | 128 | 2118 | 1963 | 1124 | 1.08× |
| 40 | 1559 | 657 | 991 | 1.57× |   |    |      |     |     |       |

**Reading the table.** Win in 67/69 cells; the two 0.97–0.98× cells (blk64, bs112/120)
are Triton's strongest corner *and* a sawtooth dip of ours. **Low bs is decisive**
(1.5–1.9× — Triton can't fill the GPU; the work-queue splits each request to ~3 CTAs/SM).
The **sawtooth in `mine`** (e.g. blk64: bs64=1446 → bs72=1982 → bs80=1587 → bs88=2135) is
wave-quantization: `nsplits=round(3·SM/bs)` makes the active-CTA count fill the 3-CTA wave
well at some bs and poorly at others — a known, smoothable artifact (variable splits to hit
exactly 3·SM, currently uniform-rounded). Triton's `tri_ks` is consistently weak here; `tri_sp`
only becomes competitive at high bs × large block.

---

## What was tried and rejected (all measured)

- **NACC multi-accumulator GEMM1** (break the 36-deep mma chain): NACC1 2227 > NACC2/4
  (2206/2174). `__launch_bounds__` pins regs at 168, so extra accumulators steal scheduling
  ILP rather than gain registers; at 3 CTAs the mma latency is already hidden.
- **Warp specialization** (producer/consumer): targets global-load latency = `long_scoreboard`,
  which is only **0.49** (cp.async already hides it); the real stalls are `barrier 2.1` +
  `wait 1.8`. Needs ≥6–8 warps/CTA → fewer CTAs (NWARPS=8 already measured 2086 < 2229).
- **K-dim (latent) tiling to cut smem**: gives **no occupancy gain** — `ncu` shows
  occupancy is **co-limited by registers AND smem, both =3 CTAs**; cutting smem leaves the
  register cap at 3. Also the K tile **doubles as V** (must stay resident for GEMM2; tiling
  forces a V reload that doubles the dominant HBM read).
- **NWARPS=8** (24 warps, 37.5% occ): 2086 < 2229 — GEMM1 only feeds 2 warps, the rest add
  barrier/overhead. **STAGES=1** (4 CTAs): 1819 — lost pipelining > the 4th CTA.
  **STAGES=3** (pipelining): 1943 (2 CTAs). **Variable-split wave-fill**, **full XOR swizzle**,
  **K-split GEMM1**, **bf16 MidO**: all net-negative.

## Limits / next step

The kernel sits at a **robust optimum for the bf16-O / 3-CTA / STAGES=2 structure** — every
in-structure lever is now exhausted. Occupancy is **co-limited by registers (168) and smem
(75 KB) at 3 CTAs/SM**; 4 CTAs needs *both* ≤128 regs (spills) and ≤56 KB smem. The only
step-change levers are: **(a)** shrink the register working set below 128 without spilling
(hard — bf16 Oreg already cut 128→64; the rest is fragments + softmax state), or **(b)** fp8
latent (smem limit → 4 and 2× mma throughput vs `wait`, though it does *not* lift the register
cap, and at 30% DRAM the halved HBM bytes help less than in a memory-bound kernel).

---

## Files
- `spec/mla_ldm.cuh` — kernel (stage1 with 2D + work-queue paths, stage2/stage2_wq,
  schedule_wq, launchers). Templated on `BLK, NT, NWARPS, STAGES, NACC, MINB`.
- `spec/mla_decoder.cu` — `MLADecoder` (init/plan/run) pybind class.
- `spec/k_h20_bs128_blk64.cu` — standalone flagship (`run()` = the delivered default).
- `spec/{full_sweep,bs_sweep,test_decoder}.py` — the table, the bs×blk sweep, and the
  correctness/graph/ragged tests.
- `PROGRESS.md` — full chronological log (every lever + measurement, incl. ncu data).

### Build / run
```bash
SGLANG_ENABLE_TORCH_COMPILE=0 PYTHONPATH=. CUDA_VISIBLE_DEVICES=<g> TORCH_CUDA_ARCH_LIST=10.0 \
  python cuda_mla/spec/full_sweep.py      # the table above
# build flags: -O3 -arch=sm_100a --use_fast_math -lineinfo
```
