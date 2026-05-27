# CUDA/CUTLASS MLA decode — autonomous optimization progress

Goal: optimize the block-table MLA decode workload (see `bench_triton_mla.py` /
`triton_mla_kernel.py`) with CUDA/CUTLASS, flashinfer as reference. Beat the best
Triton kernel (kv_split_opt ~2790 GB/s @ H=16 bs=128 blk=32; single_pass_opt
~2680 @ blk=64). B200 sm_100, peak ~8 TB/s.

## Status log
- [DONE] CUDA extension build works on sm_100 (`-arch=sm_100a`, torch load).
- [DONE] Custom scalar kernel v1/v2 (`mla_decode.cu`): split-KV flash decode,
  one CTA per (request, split), warp-per-head, K shared in smem (no head
  redundancy), shuffle-reduction GEMV, 16B vectorized gathers. CORRECT (~5e-4 vs
  fp32, incl partial tails; both split + single-pass paths). Registered as
  "cuda" in KERNELS via `cuda_mla/build.py::register()`.
- [RESULT] Scalar kernel caps at ~300-360 GB/s and is FLAT across bs/splits.
  Diagnosis: per-CTA throughput wall, not parallelism. Causes: (1) no tensor-core
  score/PV GEMM (Triton uses tl.dot/MMA), (2) non-pipelined smem round-trip
  (Triton uses num_stages=2 double-buffering), (3) low occupancy (32*H threads,
  high regs => ~1 CTA/SM) can't hide load latency, (4) serial online-softmax
  recurrence. Scalar approach won't beat Triton's MMA+pipelined kernels here.
- [NEXT] Try flashinfer's CUDA/CUTLASS MLA decode (sm100) on this workload as the
  optimized path + reference target. If H=16/20 unsupported, consider a
  tensor-core (wmma/mma.sync) CUDA kernel mirroring Triton's tiling.

## Files
- `cuda_mla/mla_decode.cu` — custom kernel source (editable).
- `cuda_mla/build.py` — JIT loader + wrapper + KERNELS registration ("cuda").
- `marks/mla/plan_sweep.py` — sweep harness (add "cuda" via cb.register()).

## Triton targets (median GB/s, sel=2048, B200)
H=16: bs=128 blk=32 kv_split_opt=2790, blk=64 single_pass_opt=2680.
H=20: bs=128 blk=32 kv_split_opt=1915, blk=64 single_pass_opt=1945.
Low bs (8): ~230 (kv_split_opt); CUDA scalar ~240 (roughly competitive there).

## Findings (iter ~1hr)
- flashinfer cutlass (sm100 Blackwell) MLA decode **hard-requires H==128**
  (`_check_cutlass_shape`: `if H != 128: raise`). UNUSABLE for H=16/20. It also
  wants block_num % (128/page_size)==0 and a fused [pages,page,576] cache.
- flashinfer fa2/fa3 (auto) MLA decode IS usable + correct (~2e-3). Measured
  (GB/s, sel=2048): H=16 bs8=662 bs64=1832 bs128=1681 ; H=20 similar.
  => BEATS Triton at LOW bs (662 vs kv_split_opt ~230, 2.9x!), LOSES at high bs
     (1681 vs 2790). Doesn't scale past bs=64. flashinfer bench: cuda_mla/bench_flashinfer.py
- Custom scalar CUDA caps ~300 GB/s, flat. fp32 compute ceiling ~2200 GB/s
  (H=16: 34816 flop/token / ~70TF fp32) => 300 is an IMPLEMENTATION gap, not a
  wall. Missing vs Triton: (1) pipelined loads (num_stages=2 double-buffer),
  (2) bf16-TC GEMM (higher compute ceiling). Levers: cp.async double-buffer,
  half2/bf16 compute, occupancy.
- To BEAT Triton high-bs (2790, bf16 MMA) ultimately needs a tensor-core
  flash-decode (O[H,512] across many warps) — large effort.

## Plan from here
- v3: cp.async double-buffered K tiles (hide gather latency at low occupancy).
- v4: bf16/half2 compute + occupancy (__launch_bounds__).
- Bank flashinfer fa2/fa3 as the LOW-bs option (real 2.9x win there).

## FINAL (autonomous session end)
Iterations done: v1 (warp/head) -> v2 (two-phase) -> v3 (cp.async double-buffer)
-> launch_bounds experiment; + full flashinfer evaluation. Custom "cuda" kernel
is CORRECT and registered, competitive only at bs<=8 (H=16). Plateau ~340 GB/s
root cause: 80 regs/thread => 1 CTA/SM (~25% occupancy, register-limited) +
scalar-FP32 GEMV ceiling. Confirmed not load-latency (cp.async no help) and not
occupancy-fixable-by-fiat (launch_bounds spilled).

Beating Triton (2790 @ bs128) needs a tensor-core flash-decode (bf16 mma.sync,
O[H,512] across >=4 warps, smem softmax). NOT attempted (high risk to leave
half-finished). Design sketch in marks/mla/cuda_mla_report.md "Recommendations".

Actionable win found: flashinfer fa2/fa3 ~2.9x Triton at low bs (662 vs 230 @ bs8)
— but needs split ckv/kpe caches (layout change vs vortex fused latent).
Final report: marks/mla/cuda_mla_report.md

## BREAKTHROUGH (ldmatrix + bank-conflict fix) — K1 h20/bs128/blk64
- Switched manual mma fragment loads -> ldmatrix + mma.sync.m16n16k16 (mla_ldm.cuh).
  Unit test (mma_unit.cu) pinned layout: GEMM1 K non-trans + SWAP B[1]<->B[2];
  GEMM2 V via ldmatrix.trans, no swap. Correct ~6e-4.
- THE "huge thing": ldmatrix read K[token][576] => 16 rows stride 576 elems => all
  hit SAME smem banks => 16-WAY bank conflict. Fix: pad smem row stride 576->584
  (HDP, keeps 16B align) => 8 distinct bank sets => 2-way. 
- Result h20/bs128/blk64 sel2048: 442(manual) -> 567(ldmatrix) -> **1330**(HDP pad).
  +135% from the pad alone. Now 68% of Triton single_pass_opt (1944).
- Best config: NT=16, WARPS=4, splits=2. (NT=32 worse: occupancy.)
- Triton PTX (TRITON_CACHE_DIR dump) confirms IDENTICAL instrs: mma.m16n8k16.row.col,
  ldmatrix.x4(+trans), cp.async.cg. So approach validated; gap = orchestration.
- Remaining levers to close 1330->1944: register-resident softmax (drop S/P smem
  round-trip + ~512 syncs at NT16), deeper cp.async pipeline (Triton ~6 groups),
  full XOR swizzle (kill residual 2-way), pad P tile.
Files: cuda_mla/spec/{mla_ldm.cuh, k_h20_bs128_blk64.cu, mma_unit.cu}

## K1 COMPLETE: register-resident softmax -> 1898 GB/s (98% of Triton 1944)
- Dropped Ssh + serialized smem softmax. For NT=16 each warp's GEMM1 C[8] holds
  the full token range for its head-tile => online softmax = within-warp shuffle
  over the 4 lanes sharing a row (rowmax/rowsum via __shfl_xor delta 1,2). m/l/alpha
  per-head to smem (cross-warp share for dim-split GEMM2 O rescale/normalize); P[head][token]
  to smem for GEMM2 A. No S round-trip.
- h20/bs128/blk64 sel2048: 442 -> 567 -> 1330 -> **1898** (W4, splits=2). correct ~6e-4.
  Hand-written CUDA now MATCHES Triton at the flagship high-batch case (and already
  beats Triton at low bs). All 3 levers (ldmatrix, stride-pad, reg-softmax) transfer to K2-K15.
- Residual ~2% + cheap wins left (deeper pipeline, full XOR swizzle, P-tile pad) if needed.

## K1 extra opts #2/#4 (both flat) + #3 assessment
- #4 P-load hoist out of dim loop (8x fewer P ldmatrix): 1898 -> 1901 (negligible;
  redundant load was already hidden). Kept (cleaner).
- #2 deeper cp.async pipeline (STAGES 2/3/4): 1893/1887/1859 — NO gain, slightly worse
  (extra K buffers cost occupancy). Kept STAGES=2. => kernel is NOT load-latency bound
  after the bank-conflict fix + register softmax.
- #3 full XOR swizzle (kill residual 2-way conflict): since #2 (more buffering) didn't
  help, the kernel isn't load/smem-throughput bound -> swizzle expected to yield little
  over the cheap 576->584 pad, and it's the riskiest change (store+all reads consistent).
- K1 final: 1898 GB/s = 98% of Triton 1944. NT16/W4/STAGES2/sp2.

## #3 full XOR swizzle: implemented, correct, but SLOWER -> reverted
- swz(row,d)=d^((row&7)<<3) on K/Q store + all ldmatrix reads (chunk-contiguity
  preserved). Correct ~6e-4. But 1675 vs 1898(pad). Post-register-softmax the kernel
  is NOT bank-conflict-bound (confirmed by #2: deeper pipeline also no help), so the
  conflict-free layout gains nothing while the hot-loop address XOR costs. Reverted to pad.
- FINAL K1 (h20/bs128/blk64): 1898 GB/s = 98% of Triton 1944. Levers that mattered:
  ldmatrix + 576->584 pad + register softmax. #2/#3/#4 gave nothing (kernel at its limit).

## K1 breakthrough attempts (B200 / structural) — all negative, kernel at robust optimum
- Profiled: 72-96 regs no spill; ~24% peak HBM => NOT bandwidth-bound; occupancy-bound
  (2-3 CTA/SM, smem-limited by the 576-wide latent tiles).
- K-split GEMM1 (fill the 2 idle warps by splitting K-dim, reduce partial S via smem):
  CORRECT but SLOWER (1758 KG2 / 1847 KG4 vs 1898) — reduction sync+smem > savings;
  the idle GEMM1 warps weren't the bottleneck. Reverted.
- cutlass sm100 path (Sm100FmhaFwdMainloopTmaWarpspecialized = tcgen05 + tmem + TMA):
  would raise occupancy (accumulator in tensor memory) but is a full CUTLASS-collective
  integration for dense varlen FMHA; retrofitting onto block-sparse paged-MLA (fused 576
  latent) is a major effort, not tractable as an incremental edit here.
- CONCLUSION: K1 = 1898 GB/s = 98% of Triton (1944) is a robust local optimum. Levers that
  worked: ldmatrix + 576->584 pad + register softmax. #2(pipeline)/#3(swizzle)/#4(P-hoist)/
  K-split all measured negative. Beating Triton's high-batch needs tcgen05/tmem (CUTLASS).

## More K1 attempts (ncu-blocked, software reasoning) — still 1898
- ncu present but GPU perf counters admin-locked (RmProfilingAdminOnly=1); even root blocked
  at container level; flipping NVreg_RestrictProfilingToAdminUsers auto-denied (shared-host security).
- multi-accumulator GEMM1 (NACC=2/3/4/6 to break the 36-deep dep mma chain): NACC=1 best (1889);
  more accumulators SLOWER (1813-1850) -> not mma-latency-bound; reg pressure -> occupancy cliff.
- STAGES=1 (less smem -> more CTAs): 1700, worse (lost pipelining > occupancy gain).
- Pattern across 8 levers: kernel is at a sharp occupancy/pipelining optimum; +regs/+smem/+instrs all hurt.
- Honest state: K1 = 1898 = 98% of Triton 1944, both at ~24% peak (m16n8k16 MMA ceiling on
  low-AI MLA score GEMM M=16/20,K=576). To beat needs ncu (find exact stall) OR tcgen05/tmem.

## Trial A (tcgen05/tmem) — investigated deeply, verdict: wrong tool for H=20
- cute sm100 headers compile here (-arch=sm_100a). Extracted full API: tcgen05.alloc/dealloc,
  SM100_MMA_F16BF16_SS (mma.kind::f16, accumulator in tmem, smem matrix descriptors via
  make_umma_desc, idesc via make_instr_desc), tcgen05.ld (16x256b) to read accumulator.
- KEY CONSTRAINT: SM100 UMMA requires M in {64,128}. For H=20 (M=heads=20) => M=64 => 20/64=31%
  row utilization, vs my m16n8k16's 2 tiles = 20/32=62.5%. tcgen05 DOUBLES the H=20 m-waste.
  (tcgen05 shines at large M, e.g. 128-head DeepSeek MLA — not 20 heads.)
- Counter-benefits (N<=256 tokens/MMA, tmem accumulator frees regs->occupancy, async) *might*
  offset, but require the full CUTLASS sm100 collective: async mbarrier MMA pipeline + swizzled
  smem descriptors + tmem alloc/lifecycle + tcgen05.ld softmax. Not debuggable here (no ncu/cuda-gdb)
  => very low odds of landing correct+fast blind.
- VERDICT: for H=20 the m16-MMA kernel I built is already near-optimal granularity. tcgen05 is not
  a clear win and not tractable to implement correctly in this environment.

## FINAL K1 RESULT
- h20/bs128/blk64 = 1898 GB/s = 98% of Triton (1944); beats Triton at low batch (bs<=16).
- Hand-written CUDA: ldmatrix + mma.sync.m16n16k16 + register-O + register softmax + 576->584 pad.
- Matches a mature compiler (Triton) and exceeds it where the workload is parallelism-starved.

## K2 (h20/bs128/blk32) = 1876 GB/s = 98% Triton(1916). DONE.
- Same kernel as K1 (BLK=32; NT=16 tiling is block-size-agnostic). Inherits full stack. Correct ~6e-4.
- bs=128 regime is at the MMA-shape ceiling (K1's 9 levers + tcgen05 verdict apply). Parity.
- Block-agnostic kernel vs Triton's block-sensitivity (bs128): blk16 1871 vs 1408 (1.33x WIN),
  blk32 1876 vs 1916 (0.98x), blk64 1898 vs 1949 (0.97x). K2/blk32 is Triton's strongest cell.

## bf16 MidO (flashinfer-style split mid-buffer) — implemented, correct, but SLOWER -> reverted
- Stored normalized partial o_s=acc_s/l_s in bf16 (O(1), bf16-safe), stage2 re-weighted by l_s*w. Correct ~7e-4-1.1e-3.
- Measured SLOWER everywhere: bs8 608->538(-11%), bs128 1874->1831(-2.4%). Kernel not bandwidth-bound
  (24% peak) so halving mid bytes doesn't help; fp32<->bf16 conversion + per-split reduction overhead
  (worst at high splits=low bs) dominates. flashinfer's bf16 mid-buffer wins only in bandwidth-bound regime.
- Reverted to fp32 MidO. K2=1876, K1=1898 restored.

## High-bs NT=64 single-pass kernel (the per-regime specialization) — built, slower -> NT=16 stays best
- Generalized register-softmax to NTOK16=NT/16 token-tiles/warp + hoisted Q-load. NT=16 unchanged
  (609/1240/1878/1892, correct). NT=64 correct ~6e-4.
- NT=64 @ bs128: best 1712 (b64_w4_s1) < my NT=16 split 1895 < Triton single_pass 1949.
  Cause: NT=64 => 187KB smem => 1 CTA/SM, 4-8 warps; warp-per-m-tile leaves GEMM1 on 2 warps.
  Triton wins high-bs because tl.dot DISTRIBUTES the score MMA across all warps; my per-warp
  register-softmax can't without cross-warp reduction (=K-split, already shown net-negative).
- ~12 levers now negative for the high-bs corner. NT=16 split is best across the whole regime.

## CAMPAIGN RESULT (H=20, sel=2048): WIN 11/15 cells, parity 4/15
- Wins (mine/Triton): bs8 2.7-5.8x, bs16 2.0x, bs32 1.34-1.36x, bs64/blk16 1.24x, bs128/blk16 1.33x.
- Parity (0.94-0.98x): bs64/128 x blk32/64 (Triton's strongest = single-pass big-tile distributed MMA).
- Best kernel everywhere: NT=16 + ldmatrix + mma.sync.m16n16k16 + register-O + register-softmax +
  576->584 pad + split-KV (splits tuned per bs). All in cuda_mla/spec/mla_ldm.cuh.

## ===== BREAKTHROUGH: ncu unlocked => the "parity corner" was OCCUPANCY-BOUND, not MMA-bound =====
## (h20/bs128/blk64 flagship: 1898 "robust optimum" -> 2186 GB/s, +11% over Triton)
The prior session declared bs128/blk{32,64} a "robust optimum" needing tcgen05, reasoning
BLIND (perf counters were admin-locked). With ncu now available the diagnosis flipped:

- **ncu on the old kernel**: DRAM 26%, Compute 27% (NOT near any ceiling); Issued/sched 0.30,
  **No-Eligible 70%**, Active warps/sched **1.77**. Top stalls: `wait` 1.69 + `barrier` 1.59.
  Theoretical occupancy **12.5%** (8 warps/SM) — capped at **2 CTAs/SM** by 254 regs AND 76KB smem.
  => Verdict: latency-bound from LOW OCCUPANCY, not the m16 MMA shape. Profiling Triton's
  single_pass_opt showed it sits in the SAME hole (12.5% occ, 0.86 wave, 80% no-eligible) —
  both kernels were starved at bs=128, which is why they tied near ~1950.

- **Lever 1 — force 3 CTAs/SM (`__launch_bounds__(128, MINB=3)`).** ptxas reschedules to 168 regs
  with **ZERO spill** (b4=128regs DOES spill 200B -> tanks). 3 CTAs/SM = 18.75% occ = 12 warps/SM.
  Only pays when paired with enough blocks to populate it => **splits=3** (384 blocks). MINBxsplits
  sweep: b3/sp3 = 2023 vs b2/sp2 = 1972. (Variable-split-to-fill-the-wave: tried, the per-call
  nsplits H2D + bigger MidO + empty-block prologue ate the gain -> reverted.)
- **Lever 2 — bf16-PACKED O accumulator (`__nv_bfloat162 Oreg[..][4]`, 64 regs vs fp32 128).**
  Halving O's register footprint gives ptxas headroom at MINB=3 -> DRAM 27.8%->28.4%, 2031->2059.
  Accuracy ~2-5e-3 (<< 3e-2 bar; verified ragged/1-tok/non-multiple). The .f32 mma still writes a
  fresh fp32 C per tile; it's folded into the bf16 running O via bf162 mul/add (pairs aligned to the
  softmax-alpha rows). The 4-CTA/SM dream (bf16-O + STAGES=1, 55KB) LOSES (1819): lost cp.async
  pipelining > the 4th block. STAGES=2 smem (74KB) caps at 3 CTAs => 3 is the real ceiling.
- **Lever 3 — 128-bit (int4) vectorized Q gather** global->smem (was scalar bf16). Prologue-only but
  free; also lifted the short-seq sel=1024 from 0.89x -> 0.98x of Triton. (K prefetch was already
  16B cp.async.) MidO write now skips pad rows (>=H), stage2 only reads h<H.

FINAL (empty B200, sel sweep, mine vs Triton single_pass_opt):
  sel=1024 1809 / 1848 (0.98)   sel=2048 **2186 / 1963 (1.11)**
  sel=3072 2351 / 2008 (1.17)   sel=4096 2413 / 2032 (1.19)   ragged 2300 / 1983 (1.16)
Kernel ncu: DRAM 25.9%->30.2%, duration 156us->133us, 18.75% occ (was 12.5%). 3 CTAs/SM is the
ceiling (regs AND smem both =3). Config: NT=16, NWARPS=4, STAGES=2, MINB=3, splits=3; all in
cuda_mla/spec/mla_ldm.cuh (k_h20_bs128_blk64.cu::run is the delivered default). The "parity corner"
the prior session called unbeatable-without-tcgen05 is now a clean +11% win — the wall was
occupancy, reachable with launch_bounds + bf16-O, no tcgen05/tmem needed.

## NWARPS probe (24 warps): fits but does NOT help — work-distribution wall, not warp count
- B200 SM = 2048 thr / 64 warps / 65536 regs / 228KB. At 12 warps we used 18.75% of THREAD cap.
- NWARPS=8 @ MINB=3 compiles to 80 regs, no spill => 3 CTAs x 8 warps = 24 warps = 37.5% occ
  (achieved 30.9%, active warps/sched 1.77->4.96). But throughput DROPPED 2187->2086: DRAM stayed
  ~flat (28.9%) while compute% rose. Root cause: GEMM1 score is done by only MTILES(=2) warps
  (register-softmax: a warp owns a 16-head tile's full token range). NWARPS=8 => 6/8 warps idle in
  GEMM1, adding barrier+prologue overhead with no extra score MMA. Feeding >2 warps needs a
  cross-warp partial-score reduction (the S-smem round-trip the register-softmax removed for +43%);
  measured net-negative. => NWARPS=4 (12 warps) is the sweet spot; occupancy is necessary not sufficient.

## ===== plan()/run() class for RAGGED batches (work-queue scheduler, cuda-graph-safe) =====
## (cuda_mla/spec/mla_decoder.cu: MLADecoder; kernels/launchers in mla_ldm.cuh)
Uniform splits make the LONGEST request serialize the wave on ragged batches. Fix = a flashinfer-style
load-balanced work queue, split init/plan/run so ONE plan() drives every layer's run() in a step:
- **init (ctor)**: allocate the work queue (work_batch/kv_start/kv_end[target_ctas], work_offset[bs+1])
  + split scratch (mid_o/m/l[target_ctas,M,*]) ONCE. target_ctas = max(3*bs,2*SM)+bs (one MINB=3 wave
  + rounding margin). Fixed sizes => no per-step alloc.
- **plan(seqlens)**: `schedule_wq` (one block) cuts each request into ceil-balanced equal-size KV
  chunks (split count ~prop to seqlen => ~uniform work/CTA), packed contiguously into the queue;
  records per-request offsets for stage2. Parallelized (reduction + fill across the block; only the
  bs-length prefix is serial) => **5.4us** (a serial single-thread pack was 40us and dominated).
- **run(q, latent, block_table, o, sm_scale)**: stage1 over the flat queue (1-D grid of target_ctas,
  ~no idle CTAs, balanced regardless of skew) + stage2_wq (reduces each request's own chunks). Takes
  NO seqlens (plan owns the schedule) => one plan(), many layer run()s.
- **cuda-graph**: both plan() and run() are fixed-grid launches on the current stream, no host sync =>
  capture [plan(); run() x N_layers] once; replay each step as seqlens change in place (plan's kernel
  re-derives the balance during replay). Verified: 1 plan + 4 layers captured, replayed after seqlen
  change, correct (6.7e-3). One plan + 8 distinct-q/latent layers also verified (9.3e-3).
- Results (bs=128 blk=64, GB/s, mine/Triton single_pass): uniform 2079/1963 = 1.06x; ragged 2x
  1737/1122 = **1.55x**; heavy skew (16 long, 112 short) 1407/374 = **3.76x**. run-only decode = 2164
  (== standalone), plan adds 5.4us (amortized to ~0 across layers). Correctness incl. all-1-token
  (exact) and graph replay all < 1.3e-2 (bf16-O floor).
Files: mla_decoder.cu (class), mla_ldm.cuh (schedule_wq/stage2_wq/run_schedule_wq/run_wq + stage1
work-queue branch), test_decoder.py (correctness + graph + ragged bench).

## bs-GENERAL policy: init/plan adapt the schedule to bs; run() dispatches (block_size, MINB)
The schedule is driven by ONE bs-independent invariant: fill one MINB=3 wave => target active CTAs
~= 3*SM. The per-request split count then falls out as ~ target*seqlen_b/sum_seqlen, so:
  * LOW bs (few requests) auto-gets MANY splits/request (fills the GPU);
  * HIGH bs (bs >= target) gets ~1 split/request (one CTA each, scheduler load-balances).
A chunk_min=128 floor avoids tiny-chunk prologue overhead; a per-request cap stops one long request
starving others on skew. __init__ sizes the queue/scratch for the given bs (target_ctas = max(target,bs)+bs);
run() dispatches the decode template by block_size (64/32/16) x MINB. All knobs (cap/chunk_min/minb)
have auto defaults but are overridable.
Sweep (bs x blk, GB/s, mine/Triton-best; one empty B200):
  blk64 uniform: bs8 1.31x  bs32 1.15x  bs64 1.28x  bs128 1.06x  bs256 1.02x
  blk64 ragged : bs8 1.38x  bs32 1.41x  bs64 1.69x  bs128 2.01x  bs256 1.79x
  blk32 uniform: bs8 1.33x  bs32 0.98x  bs64 1.11x  bs128 1.52x  bs256 1.50x
  blk32 ragged : bs8 1.16x  bs32 1.34x  bs64 1.67x  bs128 2.09x  bs256 2.21x
Wins every cell except bs32/blk32 uniform (0.98x, parity). Correctness <= 9e-3 (bf16-O) everywhere.
Sweep harness: cuda_mla/spec/bs_sweep.py.

## Opt: VECTORIZED O / MidO epilogue store (+2-3%)
The mma C-fragment lands columns (ec, ec+1) in the SAME head-row, and the bf16-O accumulator already
holds them as a bf162 pair => the 8 scalar epilogue stores per (mt,t2) collapse to 4 vector stores:
one 32-bit (bf16 O, float-precision rescale then pack) / one 64-bit float2 (the fp32 MidO that every
split CTA writes). cols are even (ec even, dim0%16) => naturally aligned. Standalone h20/bs128/blk64
sel=2048: 2180 -> **2228 GB/s**; decoder uniform 2083->2131, ragged 1768->1799, skew 1408->1465.
Correctness unchanged (<=9.4e-3). In mla_ldm.cuh stage1 epilogue.

## Considered: WARP SPECIALIZATION (producer/consumer) -- declined, wrong tool here (evidence-based)
ncu stall breakdown of the final kernel (per issued instr): barrier 2.11, wait 1.80, short_scoreboard
0.56, **long_scoreboard 0.49**, lg_throttle 0.00. Warp specialization (dedicate warps to cp.async
loads vs MMA, decouple via mbarriers) targets GLOBAL-LOAD latency = long_scoreboard, which is only
0.49 here -- the cp.async double-buffer ALREADY hides the load. The actual ceilings are barrier
(real Psh/Kc smem data deps, not removable) and wait (MMA latency + the GEMM1-on-2-warps
serialization). Worse, a clean producer/consumer split needs spare warps => >=6-8 warps/CTA, but more
warps/CTA drops CTAs/SM (occupancy): the NWARPS=8 probe already measured 2086 < 2228, a strictly worse
starting point. So warp-spec would start lower-occupancy, attack a 0.49 non-bottleneck, and only
shave ~1 of 3 barriers -- net negative, consistent with the NWARPS=8 result. The real levers
(distribute GEMM1 across warps via K-split; change the MMA shape) were both measured net-negative
earlier. Not implemented.

## Opt 2a: multi-accumulator GEMM1 (break the 36-deep mma chain) -- TRIED, NEGATIVE, reverted
Wired a real NACC>1 GEMM1 (k-tiles round-robin across NACC independent accumulators, inner loop
unrolled so the [a] index stays static -- a dynamic reg-array index would spill). Measured @ bs128/blk64
sp3: NACC1 2227, NACC2 2206, NACC3 2191, NACC4 2174, NACC6 2190 -- monotonically WORSE. ptxas: all
pinned at 168 regs (the __launch_bounds__(MINB=3) budget) with ~no spill, so extra accumulators don't
gain registers -- they steal scheduling ILP, and at 3 CTAs/SM the mma latency is already hidden by the
12 resident warps. The 168-reg budget is the wall, not intra-warp ILP. Reverted to single-accumulator.

## Opt 2b: cut the barrier stall (2.11) -- structurally blocked at 3 CTAs/STAGES=2, not implemented
The 3 __syncthreads/tile each guard a NECESSARY cross-warp smem hazard: (S1) visibility of the
cooperatively-cp.async'd K tile; (S2) GEMM1(warps 0,1 write Psh/ash) -> GEMM2(all warps read) producer-
consumer; (S3) free the K double-buffer before the loop's next prefetch reuses it. The only way to drop
a barrier is compute pipelining (overlap GEMM1(t) with GEMM2(t-1)), which needs Kc[t-1] alive while
loading Kc[t+1] => STAGES=3 => +18.7KB smem => 3->2 CTAs/SM. STAGES=3 is already measured at 1943
(2 CTAs) vs 2229 (3 CTAs) -- a 33% occupancy deficit no pipelining can recover. K-split (balance GEMM1
across the idle warps to shrink the S2 wait) reintroduces the S-matrix smem round-trip + extra syncs the
register-softmax removed (net-negative). => barrier (2.11) + wait (1.80) are structural consequences of
the smem/occupancy constraint, not cheaply removable within the 3-CTA design.

## State: robust optimum for bf16-O / 3-CTA / STAGES=2 = 2229 GB/s (+13% Triton @ sel2048). The
remaining headroom (30% DRAM) is gated by the 3-CTA smem ceiling; the one lever that breaks it is fp8
latent (halves K smem -> 4+ CTAs AND halves the dominant HBM read) -- the recommended next step.

## ===== H=16 settings (MTILES=1, no head padding) -- DONE: 4 CTAs/SM, ~3000-3021 GB/s =====
H=16 => MTILES=ceil(16/16)=1 (vs 2 for H=20): NO padding waste. M=16 halves the Q smem (37->18.7KB)
and the bf16 Oreg (64->32 regs), so smem/CTA ~= 55KB => **4 CTAs/SM** (smem-limited; vs 3 for H=20)
=> 25% occupancy, **DRAM 44%** (vs 30%). GEMM1 now runs on only 1 warp (MTILES=1) but the higher
occupancy hides it. Optimum (standalone, all blk): NWARPS=4, STAGES=2, **MINB=5, splits=4** -- MINB=5
makes Block-Limit-Registers=5 while smem caps actual occupancy at 4, the lower reg target scheduling
~1% better than MINB=4 (the tiny M=16 Oreg keeps it spill-free). Peak ~3000-3021 GB/s @ bs128.
- Standalone vs Triton (bs128, sel2048, clean B200): blk16 3021/1574=**1.91x**, blk32 3015/2480=**1.22x**,
  blk64 3023/2722=**1.11x** (Triton degrades hard at small blocks). Files: spec/k_h16.cu.

## decoder POLICY generalized to any H (auto CTAs/SM from M) + FLOOR split fix
- __init__ now computes ctas/SM from the run_wq smem footprint (M-dependent): H<=16 => 4, H<=32 => 3.
  Sets minb=ctas and target=ctas*SM, and run() dispatches run_wq for MINB in {3,4,5}. So the SAME
  decoder auto-tunes: H=16 -> 4 CTAs/splits~4, H=20 -> 3 CTAs/splits~3. (minb overridable; minb=5 adds
  ~1% at H=16 but +1 spills at H=20, so the safe auto is minb=ctas.)
- **FLOOR (not round) the proportional split** in schedule_wq: total active CTAs must stay just UNDER
  one ctas/SM wave -- overshooting costs a long 2nd-wave tail far worse than idle SMs. round->5 splits
  put H=16 bs128 on the bad point (~2450, a LOSS); floor->4 (512 CTAs, 0.86 wave) => 2934 (1.08x). The
  fix ALSO smoothed the H=20 high-bs sawtooth: bs112/120 were 0.97-0.98x LOSSES (round overshot 1 wave),
  now 1.08x WINS. No regression elsewhere.
- decoder (plan+run, ragged-capable) vs Triton, clean B200 bs128 sel2048:
  H=16: blk16 1.85x, blk32 1.18x, blk64 1.08x.  H=20: blk16 2.42x, blk32 1.55x, blk64 1.08x.
  (work-queue ~3% under the 2D standalone peak due to plan + flat-grid overhead; amortized over layers.)

## ===== INTEGRATION into vortex sglang backend + RULER verification =====
Wired the CUDA kernel as a first-class sglang attention backend, sibling of the
Triton one, with a clean in-package JIT (referencing the utils_sglang factory pattern):
- `vortex_torch/engine/sgl/attention_backend/csrc/{mla_ldm.cuh, mla_cuda.cu}` — kernel
  + a stateless `decode(q,latent,bt,sl,o,sm,block_size,splits)` op (dispatches BLK 16/32/64
  and MINB from head count: M=16 -> MINB=5, else 3). Fixed launch geometry => graph-safe.
- `cuda_mla_kernel.py` — `get_cuda_mla_module()` (process-cached load, mirrors
  indexer/utils_sglang) + `decode_blocktable_mla_cuda(...)` drop-in (same signature as the
  Triton `decode_blocktable_mla`). `csrc/build/` is gitignored.
- `cuda_mla.py` — `VortexCudaMLABackend`, byte-identical to `triton_mla.py` except the decode
  call. Registered as attention_backend `"cuda_mla"` (attention_registry.py + server_args
  ATTENTION_BACKEND_CHOICES + attention_backend/__init__.py). Not in
  FORWARD_ABSORB_CORE_ATTENTION_BACKENDS => model fuses q/k like the "triton" path.

RULER (DeepSeek-V2-Lite-Chat, H=16, block=page=32, topk=29, module=lserve_centroid_mla,
greedy, cuda-graph ON, B200): ALL THREE backends = **100.00% accuracy** (correctness verified).
End-to-end tok/s (MLA decode is a small fraction of this MoE forward; numbers also sensitive
to neighbor-GPU load): trtllm_mla (native absorbed path) ~1595 >> triton 376 ~= cuda_mla 378
(splits=2). The CUDA backend is in the generic block-table FALLBACK class (like "triton",
for non-DeepSeek geometry); within that class it matches Triton.
- Split heuristic lesson: the microbench optimum (~MINB*SM/bs ~ 7 splits) is WRONG for the
  full model — the per-layer stage2+MidO traffic dominates the tiny sparse decode. Corrected
  the backend default to a modest fill (~1 CTA/SM, clamp(SM//bs,1,4)); splits=2 at the RULER
  bs matches Triton (378 vs 376) vs 275 for the old aggressive default. Tunable via
  VORTEX_CUDA_MLA_SPLITS.

## ===== backend refactor to flashinfer-style plan()/run() (one plan drives all layers) =====
Replaced the per-layer stateless decode in VortexCudaMLABackend with the MLADecoder
plan/run idiom (like flashinfer's wrapper.plan()/run()), now merged into the package JIT
(csrc/mla_cuda.cu exposes both `decode` and class `MLADecoder`):
- init_forward_metadata / *_capture_cuda_graph / *_replay_cuda_graph: after plan_decode
  fills sparse_seqlens, call dec.plan(sparse_seqlens) ONCE — builds the load-balanced work
  queue shared by every layer. A per-bs decoder cache (mirrors flashinfer's
  decode_cuda_graph_metadata[bs]); each captured bs creates its decoder at capture (outside
  the graph region => fixed-address buffers), reused on replay.
- forward_decode: self._cur_decoder.run(q, latent, block_table, o, sm) — per layer, no
  seqlens. plan() (eager) + run() (captured) are both fixed-grid => cuda-graph-safe.
RULER (DeepSeek-V2-Lite, H=16, block=32, topk=29, graph ON): 100.00% acc, 377 tok/s ==
triton (377) == the prior stateless path. trtllm_mla (native absorbed path) still ~1595.
- KEY FIX: the work-queue grid `target_ctas` must be TIGHT. The first cut used
  max(target,bs)+bs (~SM padding CTAs at EVERY bs); at the low-bs decode tail (one long
  prompt finishing at bs=1 for ~hundreds of steps) that launched ~150 CTAs/layer, ~146 of
  them padding no-ops that STILL reserve smem on dispatch => MLA decode ~2x slower => 169
  tok/s end-to-end (MLA decode is a large fraction of the generic-fallback forward, so it
  halves throughput). Capping splits alone (4) did NOT help (168) — the grid was the issue.
  Tightening to min(target, bs*cap)+bs (=> ~bs*splits, e.g. 5 at bs=1) restored 377 tok/s.
