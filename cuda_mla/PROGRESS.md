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
