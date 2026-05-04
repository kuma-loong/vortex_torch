# Algorithm-scientist memory

Persistent working notebook for the vortex_torch sparse-attention
submission workflow. **Every agent run reads this file at start and
updates it before stopping.** It survives across sessions; the
conversation does not.

**Hard constraints** (restate every time you open this file):

- Submissions **always run in a batch sized to the free local GPUs**.
  Detect with `algorithm_scientist/free_gpus.sh` (returns space-
  separated indices of GPUs with no compute process and
  memory.used < 1024 MiB); `N = ${#FREE_GPUS[@]}`. Never launch a
  single variant — if you have only one idea, fill the other `N - 1`
  slots with orthogonal knob sweeps.
- **One batch at a time** on the free GPUs. Concurrent batches contend
  for memory and OOM/thrash. The cadence is: detect free → launch
  N-variant batch → `wait` → analyse → re-detect → next batch.
- **At least one off-catalog variant per batch** (`papers/guide.md §16`):
  paper combinations, knob inversions, untried-knob experiments, or
  first-principles answers. Pure parameter sweeps and direct paper
  replications do not count. Pre-register the off-catalog hypothesis
  in §3 the moment you launch.
- **File layout.** Submissions live under `submissions/<tag>/`, where
  `<tag>` is your sanitized model name (e.g. `claude_opus_4_7`).
  Batched runs use `submissions/<tag>/batch_<x>_id<y>.{py,json}`
  (`<x>` = batch index, `<y>` = variant slot 0…N-1). Summaries land
  at `summary_submissions/<tag>/batch_<x>_id<y>/latest.json` (the
  runner mirrors the source path).
- **Objective**: maximise AIME24 `throughput` while keeping `mean@16`
  at or above the quality floor set for this session.

---

## 1. In-flight batches  (at most 1 — every targeted GPU is consumed)

> Append one row the moment you launch a batch; remove it once all
> `N` finished rows land in §2. A batch counts as "in-flight" from
> the `wait` start until the last child writes its `latest.json`.

| tag | batch_x | launched_at | logdir | gpus | submissions | off-cat hyp § | status |
|---|---|---|---|---|---|---|---|
| _none_ |  |  |  |  |  |  |  |

---

## 2. Completed batches

> Oldest first. When a batch completes, copy headline metrics from
> `summary_submissions/<tag>/<stem>/latest.json` for each variant
> and distil 1-3 sentences of takeaway — what moved, what didn't,
> why. Always cite the off-catalog variant and what its result said
> about the §3 hypothesis it pre-registered.

<!--
### <tag>/batch_<x> — <one-line theme>  (launched YYYY-MM-DD HH:MM, free GPUs: 0,3,5,7)

| variant         | content_hash | mean@16 | pass@16 | throughput (tok/s) | e2e_time (s) | notes |
|---|---|---|---|---|---|---|
| batch_<x>_id0   |  |  |  |  |  |  |
| batch_<x>_id1   |  |  |  |  |  |  |
| …               |  |  |  |  |  |  |
| batch_<x>_idN-1 |  |  |  |  |  | ← off-catalog (§3 hyp #?) |

**Knob matrix:** id0=…, id1=…, …, idN-1=… (off-catalog: …)

**Off-catalog hypothesis tested:** §3 row #? — verdict: confirmed/refuted/inconclusive.

**What moved:** …

**Takeaway:** …
-->

### claude_opus_4_7/batch_0 — establish baselines + orthogonal knob sweeps + Prism×Keyformer off-catalog (launched 2026-05-04 08:22, free GPUs: 0-9)

| variant         | content_hash | mean@16 | pass@16 | throughput (tok/s) | e2e_time (s) | notes |
|---|---|---|---|---|---|---|
| batch_0_id0 (baseline kimi-style: CMean centroid, exact topK, block=32, topk=29/0.0625, layers_skip=[0], kv=auto, mfs=0.8) | 248baba6a214 | 0.3854 | 0.7000 | 985.6 | 7541 | anchor |
| batch_0_id1 (id0 + approxTopK(0.1)) | c4a2dc60c0c3 | 0.4125 | 0.6000 | 995.7 | 7441 | mean@16 +0.027 vs anchor (likely noise); thr +1% — **approxTopK(0.1) is free** |
| batch_0_id2 (id0 + mem_fraction_static=0.9) | 12c23335acb9 | 0.3917 | 0.7000 | 1079.2 | 6903 | **+9.5% thr at zero acc cost** |
| batch_0_id3 (id0 + kv_cache_dtype=fp8_e4m3) | a048e0068dc1 | 0.3854 | 0.6667 | **1483.8** | 5017 | **+50.5% thr at zero mean@16 cost (pass@16 −0.033 — small)** |
| batch_0_id4 (id0 + tighter budget topk=21/0.05) | 9caa333498e7 | 0.3479 | 0.6000 | 1034.1 | 7181 | −0.038 acc, +5% thr — bad trade-off |
| batch_0_id5 (id0 + looser budget topk=64/0.125) | bacfbc27172a | 0.4604 | 0.7667 | 834.0 | 8791 | best mean@16 in batch (+0.075 vs anchor); −15% thr |
| batch_0_id6 (id0 + layers_skip=[0,1]) | 43e251a7a17e | 0.3771 | 0.7000 | 951.3 | 7826 | thr *worse* than [0] (counter-intuitive); inconclusive |
| batch_0_id7 (GQA-Quest envelope CMin+CMax score) | 29c16135422a | 0.3458 | 0.6000 | 950.9 | 7857 | −0.040 acc, −3.5% thr — Quest envelope is *worse* than CMean here |
| batch_0_id8 ← **off-catalog** (Prism CL2Norm dual-band × Keyformer Save/Load heavy-hitter; §3 #1) | 0c8ef28ac9f4 | **0.1167** | 0.2667 | 927.2 | 8471 | **REFUTED** — accuracy collapsed; momentum poisoned the picker |
| batch_0_id9 (id0 + block_size=16) | c410f788acf8 | 0.3583 | 0.6000 | 1006.7 | 7474 | −0.027 acc, +2% thr — block=32 dominates block=16 here |

**Knob matrix:** id0=anchor, id1=approxTopK(0.1), id2=mfs=0.9, id3=fp8_e4m3, id4=tight budget, id5=loose budget, id6=skip[0,1], id7=Quest envelope, id8=Prism×Keyformer (off-cat), id9=block=16

**Off-catalog hypothesis tested:** §3 #1 — Prism CL2Norm dual-band score wrapped in a Save/Load heavy-hitter accumulator. **Verdict: REFUTED.** mean@16 = 0.117 vs 0.385 anchor — accuracy fell by ~70% relative. Two suspected mechanisms: (a) the per-feature `q · k_norm` term has wildly different scale than `q · centroid`, so the α=β=0.5 blend is dominated by whichever is larger and the weighted sum is noisy; (b) the heavy-hitter accumulator (decay=0.9) preserves and amplifies that noise across all decode steps, so even briefly-noisy pages dominate the topK once accumulated. Either mechanism alone would degrade quality; together they collapse it.

**What moved (in throughput-friendly direction):** fp8_e4m3 (+50.5%), mfs=0.9 (+9.5%) are the two clean throughput wins, and they should stack independently. approxTopK(0.1) is throughput-neutral but quality-neutral too (drop-in).
**What didn't move (or hurt):** wider layer skip, smaller block size, tighter budget, Quest envelope, heavy-hitter accumulator. Some of these hint at the model's structure (Qwen3-4B may have only ~36 layers; layer 0 is the dominant skip target).

**Takeaway:** Stack `fp8_e4m3 + mfs=0.9` as the new throughput-baseline. Expected ~1620 tok/s at mean@16 ~0.385 (+64% over batch-0 anchor). Batch 1 should explore: fp8_e5m2 (more aggressive fp8), mfs=0.95 (push memory harder), block_size=64 (less cache overhead per token), oai-style wider layer skip, and a multi-resolution Prism off-catalog (`CMeanInterleave(k=block_size/4)` + `Kron(dim=1)` fold) — without the heavy-hitter accumulator that broke id8.

---

## 3. Design hypotheses (running)

> Keep one row per idea under investigation. Close a row with a
> verdict when you have enough evidence. A refuted hypothesis is
> just as valuable as a confirmed one — record it. Number rows so
> the §1 "off-cat hyp §" column and §2 "off-catalog hypothesis"
> notes can cite them.

| # | hypothesis | source | evidence for (batches) | evidence against | verdict |
|---|---|---|---|---|---|
| _ex_ | _"fp8_e5m2 kv cache keeps mean@16 within 5% of bf16 on AIME24"_ | _§16.4 / Double Sparsity §4_ | _e.g. <tag>/batch_3, batch_5_ | _e.g. batch_4_ | _open / confirmed / refuted_ |
| 1 | "Prism (high-frequency CL2Norm summary) wrapped in a Keyformer-style Save/Load heavy-hitter accumulator picks pages that neither plain CMean+topK nor a pure heavy-hitter alone catches; effective on position-sensitive AIME24 reasoning steps." | papers/guide.md §16.1 (Prism × Heavy-hitter combination) | — | claude_opus_4_7/batch_0 id8 (mean@16=0.117 vs 0.385 anchor) | **refuted** — see batch 0 takeaway: scale mismatch between `q·centroid` and `Σ q⊙k_norm` makes the α=β=0.5 blend noisy, then the heavy-hitter momentum amplifies the noise across decode steps |

---

## 4. Anti-patterns (things tried that did not work)

> One line each. Brief. Future-you will thank you for the crisp list.

- **Heavy-hitter Save/Load momentum on a multi-term blended score** (batch_0/id8): mean@16 collapsed to 0.117. Either skip the blend or reset/normalise the accumulator each step.
- **Tightening `topk_val=21, ratio=0.05`** at fixed CMean centroid + bf16 KV (batch_0/id4): −0.038 mean@16 for only +5% throughput. Bad trade.
- **Block size 16 over 32** at fixed everything else (batch_0/id9): −0.027 mean@16 for +2% throughput. block=32 dominates.
- **Layer skip [0, 1] vs [0]** (batch_0/id6): no throughput win, even slightly *slower* (−3.5%). Skipping more layers is not automatically faster — likely because dense vs sparse path costs near-balance for layer 1 of Qwen3-4B.
- **GQA-Quest envelope (CMin+CMax) score function** (batch_0/id7): −0.040 mean@16 vs CMean centroid, no throughput win. CMean dominates Quest on AIME24/Qwen3-4B.

---

## 5. Patterns that worked (reuse these)

> Confirmed throughput / accuracy wins worth carrying forward.

- **`kv_cache_dtype: "fp8_e4m3"`** (batch_0/id3): +50.5% throughput at zero mean@16 cost. Single biggest lever observed. Use as the default in all subsequent variants.
- **`mem_fraction_static: 0.9`** (batch_0/id2): +9.5% throughput at zero mean@16 cost. Stack with fp8.
- **`approxTopK(tolerate_ratio=0.1)`** (batch_0/id1): throughput-neutral, quality-neutral. Free drop-in for `topK()` — keep when iterating.
- **block_size=32 + topk_val=29 + topk_ratio=0.0625 + layers_skip=[0]** as the kimi-style anchor: clean baseline, mean@16=0.385 at 985 tok/s. Quality floor reference.

---

## 6. Open questions / next directions

> The agent's own backlog of ideas it hasn't tested. When the
> current batch finishes (no in-flight rows in §1), pick from the
> top of this list. At least one slot in every new batch must be
> off-catalog — promote items here into pre-registered §3
> hypotheses on launch.

- **Batch 1 sketch (post-batch-0):** carry forward batch 0's best variant; sweep `approxTopK(tolerate_ratio ∈ {0.05, 0.15, 0.30})`, `topk_val` tighter by 30%, `block_size=64`, `layers_skip=[0,4,8,12,16,20,24,28,32]` (oai_v0 style), `mem_fraction_static=0.95`. Fill the off-catalog slot with **multi-resolution Prism dual-band** (papers/guide.md §14.1 + §16.1): `CMeanInterleave(dim=1, k=block_size/4)` → `[1, 4, D]` mini-centroids; indexer-side `Kron(dim=1)(q, mini_centroids) → [S, H_q*4, D]` then `Sum(dim=2)+Max(dim=1)`. Structurally identical to LServe (vortex_torch/flow/algorithms.py:501) but with *mean-pooled* mini-centroids instead of max/min envelopes — recovers high-frequency RoPE info that pure CMean destroys.
- **§16.4 first-principles** — what's the smallest cache footprint that still clears the floor? After batch 0 if id3 (fp8) or id4 (tight budget) doesn't crash quality, push fp8_e5m2 + tighter budget + smaller block_size simultaneously to find the brittle edge.
- **§16.3 inversion of "BOS is necessary"** — try `vortex_block_reserved_bos=2` (current) vs `vortex_block_reserved_bos=4` and explicit score downweighting of the first 4 pages via MaskSlice. Does explicit downweighting buy throughput at fixed quality (the framework keeps BOS reserved anyway, freeing topk slots for genuinely informative pages)?

---

## 7. Reading log

> Timestamped notes from tutorials / developer_guides / source code.
> One bullet per file read, with the single most useful insight.

- 2026-05-04 [vortex_torch/indexer/__init__.py + vortex_torch/cache/__init__.py] **Asymmetric op coverage** between indexer and cache: indexer exports `Sum` and `Conv1d` but cache does NOT (cache reductions are Mean/Max/Min/L2Norm only — `Sum` must be done indexer-side, often via `Mean` + a constant scaling); cache exports the four `*Interleave` reductions (Mean/Max/Min/L2NormInterleave) but indexer has none — so multi-resolution mini-summaries are *constructed* on the cache side and *consumed* on the indexer side, typically via `Kron(dim=1)`. Also: indexer has `GeMV` separate from `GeMM` (potentially cheaper for `[1, 1, D] × [S, 1, D]` scoring patterns) — worth swapping into id0/id8 for a "GeMV vs GeMM" knob in batch 1.
- 2026-05-04 [vortex_torch/flow/algorithms.py] `LServeSparseAttention` is a concrete reference for **Kron + MaxInterleave/MinInterleave** as a multi-resolution selector. It declares `cache["max"]: (block_size//LSERVE_BLOCK_SIZE, head_dim)` (i.e. b/k mini-envelopes per block) and uses `Kron(dim=1)(q[1,H_q,D], cache["max"][S, b/k, D]) → [S, H_q*(b/k), D]` then `Sum(dim=2)+Max(dim=1)` to fold. This is the structural template for any "Prism dual-band" off-catalog variant — mini-centroids via `CMeanInterleave`, indexer-side fold via `Kron(dim=1)+Sum+Max`. Note: `LSERVE_BLOCK_SIZE=16` is the *sub-block* size, so for `block_size=32` you get 2 mini-summaries; for `block_size=64`, 4. Configurable knob worth a sweep.

---

## 8. Session notes

> Freeform per-session summary: what the agent did, what was surprising,
> what's left for next time. Append — do not overwrite.

- 2026-05-04 — claude_opus_4_7 session start. Bootstrap reads done (CLAUDE.md, AGENTS.md, 6 tutorials, papers/guide.md, memory.md). Designed batch 0 with 10 variants on 10 free GPUs (0-9). id0 = kimi-style baseline; id1-id7,id9 = orthogonal knob sweeps (terminal op, mem_fraction, kv dtype, topk budget, layers_skip, score function, block_size); id8 = off-catalog Prism×Keyformer (§16.1) with Save/Load heavy-hitter on a CMean+CL2Norm dual-band score. All 10 pre-flights passed CPU compile sweep. Using vortex_new conda env (/home/zhuominc/anaconda3/envs/vortex_new/bin/python) per user instruction.
