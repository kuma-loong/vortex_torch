# Sparse-attention reading guide for vortex_torch

> **Read this section first.** The papers below are *seeds*, not
> recipes. Every one of them was the product of someone noticing a
> gap that the previous papers had not filled — StreamingLLM noticed
> the sink, MagicPIG noticed that "attention is sparse" is mostly an
> artifact of the sink, Prism noticed that mean-pooling silently
> destroys high-frequency RoPE patterns. The most valuable thing you
> can do in this codebase is **find the next gap**, not faithfully
> reimplement the previous one. Use §14 as scaffolding for "what's
> already known to work"; use §16 as a prompt for "what nobody has
> tried yet"; and treat any framework-compatible idea you can defend
> as a legitimate batch variant *even if you can't cite it*. A
> winning flow that nobody published is the goal, not an
> embarrassment.

This folder collects ten papers on KV-cache compression, sparse
attention, and long-context LLM serving. Each one proposes a different
way to **avoid touching every key/value during decoding**, which is
exactly the design space `vortex_torch` operates in: a per-block
`forward_cache` builds compact summaries; a per-step `forward_indexer`
scores pages from those summaries and hands a top-k list to the
attention kernel.

The guide is organized so you can:

1. Skim §1–§10 for what each paper actually does, in vortex-relevant
   terms.
2. Read §11 for the patterns that recur across papers — these are the
   levers you have when designing a flow.
3. Read §12 for a direct mapping from paper ideas → `vortex_torch`
   primitives (`Mean`, `Max`, `L2Norm`, `MeanInterleave`, `GeMM`,
   `Kron`, `Save`/`Load`, `MaskSlice`, …).
4. Read §13 for what the framework *cannot* do (CPU offload, ANNS,
   per-head static masking, post-hoc retraining), so you don't waste
   batches chasing those ideas.
5. Use §14 as a backlog of concrete, framework-compatible submission
   ideas inspired by each paper.
6. **Use §16 as the prompt for inventing flows that no paper here
   explores.** Every batch should include at least one off-catalog
   variant — see §15 for the workflow.

## 1. StreamingLLM — attention sinks for infinite-length streaming
*Xiao, Tian, Chen, Han, Lewis. ICLR 2024. arXiv:2309.17453.*

**Claim.** Pure window attention (drop KVs older than `W`) collapses
the moment the first token is evicted, even when the first tokens
carry no semantic content. The reason is the softmax: scores must sum
to 1, so the network parks "unwanted" attention mass on the first few
positions, which become **attention sinks**. Keeping just **4 sink
tokens + a recent window** matches dense quality on 4M-token streams
with 22× speedup over sliding-window-with-recompute.

**Why it matters here.** This is the empirical justification for
`vortex_block_reserved_bos` (and `..._eos`) — the framework already
hard-reserves the first BOS pages of every sequence regardless of
score, so a flow does not need to add an additional sink-bias term
to its score. The other lesson is operational: do not let your scoring
op down-weight the leading pages; if your `topK` budget is `k` and the
sink budget is `bos`, the effective sparse budget is `k + bos`.

**Submission angle.** None directly — the sink primitive is built in.
But it explains why a "drop the first 8 pages" experiment will tank
quality even if those pages look uninformative.

## 2. Keyformer — recent + dynamically selected "key" tokens
*Adnan, Arunkumar, Jain, Nair, Soloveychik, Kamath. MLSys 2024. arXiv:2403.09054.*

**Claim.** ~90% of attention weight in generative inference
concentrates on a small subset of "key" tokens. Keyformer keeps
recent tokens **plus** a dynamically-identified key set, scored by a
novel score function that mixes attention magnitudes with positional
priors. 50% KV reduction with no accuracy loss; 2.1× latency / 2.4×
throughput.

**Why it matters here.** Keyformer's "score = recency_window +
running_attention_mass" is buildable inside `forward_indexer` as a
two-term score. The cumulative attention mass needs to survive across
decode steps, so a `Save`/`Load` pair on a per-page accumulator is
required (see §12).

**Submission angle.** A flow that maintains a per-page running
attention-score sum (`Load → Add(α=decay, β=score) → Save`) and adds
a positional-recency bonus via `MaskSlice` over the last-N pages.

## 3. IceFormer — k-NN attention on CPU, no retraining
*Mao, Ester, Li. ICLR 2024. arXiv:2405.02842.*

**Claim.** Frozen pretrained Transformers can be accelerated 2.7–7.6×
on CPU by replacing the dense attention matrix with a **k-NN search
over keys**, fetching only the top-k V vectors per query. 98–99.6%
quality retention.

**Why it matters here.** This is the cleanest statement of the core
intuition vortex_torch is built on: per-query, you only need a small
top-k of pages. Where vortex_torch differs is **granularity** — pages
not tokens — so the relevant similarity metric is `q · centroid`
rather than `q · token_key`. The IceFormer evaluation is a useful
sanity check that this lossy step is acceptable on long-context
benchmarks.

**Submission angle.** None new — every vortex flow is already an
IceFormer-style approximation at the page granularity. But it's the
right baseline citation.

## 4. Double Sparsity — token + channel sparsity, channel via calibration
*Yang, Sheng, Gonzalez, Stoica, Zheng. arXiv:2408.07092.*

**Claim.** Combine token sparsity (only attend to important tokens)
with **channel sparsity** (only use a subset of feature channels for
the score). The crucial observation: channel importance is **largely
static** across queries, so it can be picked once via offline
calibration; the dynamic part is just "which tokens." With 1/16 token
× 1/16 channel sparsity, attention op runs 14.1× faster, end-to-end
1.9×; 16.3× decoding throughput with offloading.

**Why it matters here.** Channel sparsity is a real lever vortex_torch
can pull, *if* the "important channel" set is encoded into the flow at
write time. Two ways to do this:

- **Static MaskSlice on dim=2.** Build a `[1, 1, D]` 0/1 mask
  indicating "use these `D/16` features"; pre-multiply both `q` and
  `cache["centroids"]` against it before the dot product. The mask
  itself is a constant — `MaskSlice(start=0, end=D/16, dim=2,
  alpha=1.0, beta=0.0)` only works if the important channels are
  contiguous, which they generally are not.
- **A learned permutation baked into the cache field.** Compute a
  reduced-dim centroid `[1, 1, D/r]` in `forward_cache` using
  `CGeMM` against a fixed projection matrix you store as a constant.
  Same idea as a learned linear probe.

The "calibration" step is offline though — vortex_torch has no hook
for it. You'd have to bake the calibrated channels/projection into
the flow's `__init__`.

**Submission angle.** Variant that scores via a fixed linear
projection of the centroid (see §14, "channel-projected centroid").

## 5. RetrievalAttention — ANNS on CPU, OOD-aware
*Liu et al., Microsoft. arXiv:2409.10516.*

**Claim.** Build an ANNS (approximate nearest-neighbor search) index
over KV vectors in CPU memory; at decode time, retrieve only the
1–3% most relevant. The catch: queries and keys come from
different projection matrices and lie in *different distributions* —
off-the-shelf ANNS indexes assume they don't, so they degrade. The
paper's contribution is an **OOD-aware index** that adapts to the
query distribution. Single RTX 4090 serves 128K tokens for an 8B
model.

**Why it matters here.** The OOD finding (Q and K geometry differ
substantially) is a **universal warning** for vortex flows: any
similarity metric you build on `q` and `cache[…]` should not assume
they are drawn from the same distribution. Practical implication:
score with `GeMM(q, centroid)` followed by a *learned* normalizer
(e.g. `Normalize` or fp32 z-score) rather than raw cosine.

**Submission angle.** Not the ANNS itself (vortex_torch is
single-kernel, no CPU offload), but a flow that **post-processes** the
raw `q · centroid` score with `Normalize(dim=0)` (across pages within
a sequence) to compensate for the distribution mismatch. See MagicPIG
below for the same warning from a different angle.

## 6. MagicPIG — LSH sampling instead of TopK
*Chen et al., CMU/UW/NYU/Meta. arXiv:2410.16179.*

**Claim.** Three findings that together kill the naive "top-k by
attention score" recipe:
1. *Attention is not always sparse.* On aggregation tasks the top
   20% of keys cover only ~70% of attention mass — so dropping the
   bottom 80% is genuinely lossy.
2. *Apparent sparsity is mostly the attention-sink artifact.* Once
   the sink is set aside, the rest of the distribution is much
   flatter than it looks.
3. *Q and K live in narrow opposite cones.* Standard nearest-neighbor
   search in this geometry is intrinsically hard — the assumptions
   it makes are violated.

The fix: **LSH sampling** (with bias-correction) on CPU instead of
exact-or-approximate top-k. 5× decode throughput; 54ms latency on a
single RTX 4090 for Llama-3.1-8B with 96K context.

**Why it matters here.** This is the strongest theoretical pushback
against the entire premise of `topK(score)`. The takeaways are:

- **Don't over-tighten `vortex_topk_val` on aggregation tasks.** A
  flow that's perfect on retrieval can collapse on summarization /
  reasoning if `topk_val` is squeezed too aggressively.
- **Sink-handling is non-optional.** Whatever you do, do not include
  the sink pages in the top-k budget — the framework's
  `vortex_block_reserved_bos` already takes care of this.
- **The "narrow opposite cones" geometry** is the same OOD warning as
  RetrievalAttention. A flow that just does `q · centroid` is doing
  inner-product retrieval in a regime where inner-product retrieval
  is borderline. Pair with a magnitude term (`L2Norm`) or a per-page
  z-score to compensate.

**Submission angle.** A "robust score" flow: `q · centroid` plus
`L2Norm(dim=2)(centroid)` plus `L2Norm(dim=2)(value_summary)` —
combined via `Add` — which down-weights pages whose keys *or*
values are small even if the dot product happens to be large. This is
the spirit of MagicPIG (account for the full distribution, not just
the top of it) without doing actual LSH.

## 7. ShadowKV — low-rank pre-RoPE keys + offloaded values
*Sun et al., ByteDance/CMU. arXiv:2410.21465.*

**Claim.** Two structural facts:
1. **Pre-RoPE keys are low-rank.** Truncated SVD (rank 256 on
   Llama-3.1-8B with 1024-dim head) loses essentially nothing.
2. The KV cache is the bottleneck for memory, but pre-RoPE keys can
   be SVD-compressed on GPU while values are offloaded to CPU.

System-level: GPU stores the low-rank key cache; CPU stores values;
sparse KV pairs are **reconstructed on-the-fly** for the selected
pages. Up to 6× larger batch size and 3.04× throughput.

**Why it matters here.** vortex_torch can't replicate the
GPU/CPU split, but the **first** finding — pre-RoPE keys live in a
low-rank subspace — is directly actionable. If you can build a
per-page low-rank summary (analogue of pre-RoPE-key SVD truncation)
during `forward_cache`, your indexer score can use a much smaller
feature dimension than `D`, saving bandwidth on every step.

**Submission angle.** A **rank-r centroid**: in `forward_cache`,
compute `CGeMM(cache["k"], W_lowrank)` with a fixed projection
`W_lowrank: [r, D]` (baked in `__init__`) to produce `[1, 1, r]`
centroids in low-rank space. The indexer then projects `q` through the
same `W_lowrank` and scores in `r`-dim. Smaller cache field, smaller
score-time GeMM, same selection. Pairs naturally with §4 (Double
Sparsity).

## 8. LServe — unified static + dynamic block-sparse attention
*Yang et al., MIT/SJTU/NVIDIA. MLSys 2025. arXiv:2502.14866.*

**Claim.** Convert half the attention heads into **streaming heads**
(static Λ-shaped masks, à la StreamingLLM) and leave the other half
**dense**. Add **dynamic** block-level KV page selection on top
(query-centric similarity, hierarchical page selector that reuses
selection results across nearby tokens to amortize the picker cost).
The two sparsity forms are **orthogonal** — combining them
multiplies. 2.9× prefill, 1.3–2.1× decode.

**Why it matters here.** Three concrete lessons:

- **Static and dynamic sparsity are orthogonal.** vortex_torch's
  `vortex_layers_skip` is one form of static sparsity (skip whole
  layers); it stacks cleanly on top of `topK`-based dynamic page
  selection. Combining them aggressively is the throughput playbook.
- **A constant number of selected pages is enough** (the paper
  reports 4096 KV tokens regardless of context length on long-context
  tasks). This is the rationale for `vortex_topk_val` being
  context-length-independent.
- **Page-selector reuse across nearby query tokens** — vortex_torch's
  decoding loop already runs one selector per decode token, but the
  framework caches the selection within a single forward pass per
  layer. This is built-in; the paper validates it is the right
  approach.

**Submission angle.** A flow that's deliberately *cheap* on the
indexer side (single `Multiply → Sum(dim=2)` from a low-rank
centroid, no fancy compositions) — relying on the picker simplicity
itself to be the throughput win, the way LServe's hierarchical
selector does. Combine with aggressive `vortex_layers_skip`.

## 9. Prism — spectral-aware block-sparse attention
*Wang, Wang, Liu, Liu, Chu, Song, Qiu. arXiv:2602.08426 (Feb 2026).*

**Claim.** Block-sparse prefilling needs cheap **block importance
estimation**. The standard recipe — mean-pool tokens within a block,
compute attention against the mean — is *theoretically broken* on
RoPE-encoded keys: mean-pooling acts as a low-pass filter, and the
high-frequency dimensions of RoPE rotate fast enough that pooling
**destructively interferes** them. The result is a "blind spot" that
erases fine-grained positional patterns (slash diagonals). Prism
fixes this by **decomposing** block importance into a low-frequency
branch (semantic) and a high-frequency branch (positional / local
slash patterns) with energy-based temperature calibration. Up to
5.1× speedup at 128K with full-attention parity.

**Why it matters here.** This is the **single most actionable result
in the folder for vortex_torch.** The cache-side `CMean(dim=1)`
centroid that nearly every flow in `vortex_torch/flow/algorithms.py`
relies on is exactly the mean-pool that Prism shows is broken on the
high-RoPE dimensions. Three implications:

1. A pure-`CMean` centroid is doing attention scoring blind to slash
   patterns. Pages with strong local-position signal will look
   uninformative.
2. **Pair `CMean` with `CL2Norm` and/or `CMax`** — these are not
   linear, so they survive RoPE rotation and preserve the
   high-frequency energy that mean-pool kills. This is
   already a vortex_torch idiom for "QUEST envelope" (CMin+CMax) and
   is justified post-hoc by Prism.
3. **Multi-resolution centroids** (now possible via the new
   `CMeanInterleave(dim=1, k=g)` op) are a partial fix: instead of one
   `[1, 1, D]` mean over the whole block, emit `[1, g, D]` of
   `block_size/g`-token sub-block means. Sub-block size of ~4 keeps
   enough of the high-frequency content because the RoPE rotation
   over 4 adjacent tokens is small. The indexer then scores against
   each sub-centroid and folds them with `Max(dim=1)`.

**Submission angle.** The most important entry in §14. A **dual-band
centroid** flow: `CMean` (low-pass, semantic) plus `CMeanInterleave(k=4)`
(preserves high-frequency / slash) plus `CL2Norm` (preserves
magnitude). Indexer combines them with `Add`/`Max` before `topK`.
Predicted to clear quality floor on `niah-multikey` and similar
position-sensitive tasks where pure-`CMean` flows wobble.

## 10. SparQ Attention — sparse query, then top-k full keys
*Ribar, Chelombiev, Hudlass-Galley, Blake, Luschi, Orr. Graphcore. ICLR PML4LRS 2024. arXiv:2312.04985 (referenced as 64_SparQ_…).*

**Claim.** Three-step recipe per attention head:
1. Take the `r` largest-magnitude components of `q` (sparse query).
2. Compute approximate scores using only those `r` channels of `K`.
3. Top-`k` by approximate score, then fetch full `K`/`V` only for
   those `k` positions; compensate for the missing mass by
   interpolating with a running mean-V vector.

Up to 8× attention data-transfer savings, no accuracy drop.

**Why it matters here.** SparQ is the cleanest example of **two-level
sparsification** that maps directly onto vortex_torch:

- The "sparse query → approximate score" step is exactly what `q ·
  centroid` already does, just at page granularity instead of channel
  granularity.
- The "full attention only on top-k" step is exactly what
  `topK(score)` triggers.
- The "running mean-V residual to recover missing mass" step has
  **no analogue in vortex_torch today** — once you've top-k'd, the
  framework discards the dropped pages. A `Save`/`Load`-backed
  running-V cache field could in principle be added, but the engine
  doesn't currently consume it during attention.

**Submission angle.** Within vortex_torch you can implement steps 1
and 2 of SparQ at the channel level (sparse-query channel selection
for the score) by using `MaskSlice(dim=2)` to keep only the largest-r
channels of `q` before the `GeMM` against a centroid. This is
weaker than the original (channels are picked statically rather than
per-query), but it's the channel-sparsity story of §4 (Double
Sparsity) restated.

---

## 11. Cross-cutting themes

Read in aggregate, the ten papers stake out four orthogonal axes that
every sparse-attention design has to make a choice on. **Vortex_torch
flows that vary along these axes give you good orthogonal-batch
candidates.**

### Axis A — Granularity of the selected unit

| paper | unit |
|---|---|
| StreamingLLM, Keyformer, H2O, IceFormer, SparQ, MagicPIG, RetrievalAttention | **token / KV-pair** |
| Quest, ShadowKV, LServe, Prism, vortex_torch | **block / page** |

Page-level selection is strictly cheaper (the picker runs over `S`
pages, not `S × block_size` tokens). The cost is selectivity — you
can't drop a hot token within a cold page, only the whole page. The
framework's design is firmly in the page-level camp; do not try to
fight it.

### Axis B — Token sparsity vs channel sparsity vs both

| paper | token sparsity | channel sparsity |
|---|---|---|
| StreamingLLM, Keyformer, H2O, MagicPIG, LServe, Quest, IceFormer, RetrievalAttention | ✓ | — |
| Double Sparsity, SparQ | ✓ | ✓ |
| ShadowKV (low-rank K) | — | low-rank ≈ channel sparsity |

**vortex_torch is token-sparse out of the box (page-level, but same
spirit) and has *channel-sparsity hooks* you usually don't use:**
`MaskSlice(dim=2)` for static channel selection, `CGeMM` against a
small projection matrix for low-rank summaries (§4, §7). This is the
single underexplored axis in the existing flow library.

### Axis C — Score function: dot product, magnitude, both, or learned

- **Pure dot product** (`GeMM(q, centroid)`) — every vanilla flow.
- **Magnitude** (`L2Norm` of `cache["k"]` or `cache["v"]`) — used as
  a *bias* in QUEST and various magnitude-aware variants.
- **Both** (combined via `Add` / `Multiply`) — robust to OOD
  query/key geometry per MagicPIG, RetrievalAttention.
- **Learned** (a fixed linear probe baked into the flow) — Double
  Sparsity, partly ShadowKV.

The MagicPIG/RetrievalAttention OOD finding says that pure dot-product
scoring is *fragile*: the `q` and `centroid` distributions differ
enough that occasional pages will have inflated dot products from
geometry alone. Adding a magnitude term as an `Add(α=1, β=γ)` post-
processing step costs essentially nothing and fixes this.

### Axis D — Static vs dynamic selection

LServe makes the cleanest statement: **static** patterns (Λ-shaped
masks, BOS sinks, layer skips) and **dynamic** selection (top-k by
score) are *orthogonal*; combining them multiplies the speedup.
vortex_torch already supports both — `vortex_layers_skip`,
`vortex_block_reserved_bos`/`_eos` are static; `topK(score)` is
dynamic. The orthogonal-batch design pattern is to vary one axis at a
time.

### Axis E — Where the heavy work lives

| paper | location |
|---|---|
| RetrievalAttention, MagicPIG, ShadowKV (V offload), Double Sparsity (offload mode) | **CPU offload** |
| StreamingLLM, Keyformer, IceFormer, LServe, Prism, Quest, SparQ, vortex_torch | **GPU only** |

vortex_torch is GPU-only by design. Every CPU-offload paper is a
"how would the system look if we had infinite host memory and a
fast PCIe interconnect?" exploration. Their findings about *what to
keep* are still applicable to a GPU-only flow even though the
mechanism isn't.

---

## 12. Mapping paper ideas → vortex_torch primitives

Concrete rosetta stone. Left column is a recurring concept from the
papers; right column is the cheapest vortex_torch realization.

| paper concept | vortex realization |
|---|---|
| Attention sink (StreamingLLM) | `vortex_block_reserved_bos` (already on by default) |
| Recent-window bias (Keyformer, StreamingLLM) | `MaskSlice(dim=0)` over the last-N pages, added to the score (`Add(α=1, β=γ)`) |
| Heavy-hitter / running attention mass (H2O, Keyformer) | per-page `Save`/`Load` accumulator: `Load(F) → Add(α=decay, β=score) → Save(F)` |
| Channel sparsity / Double Sparsity | static `MaskSlice(dim=2)` mask, *or* a fixed `[r, D]` projection encoded as a `CGeMM` weight |
| Sparse-query top-r channels (SparQ) | static `MaskSlice(dim=2)` on `q` before the score-GeMM (channels picked offline, not per-query) |
| Low-rank pre-RoPE key (ShadowKV) | `CGeMM(cache["k"], W_lowrank)` in `forward_cache` to write `[1, 1, r]` low-rank centroids |
| Magnitude / L2-norm bias (MagicPIG, RetrievalAttention) | `CL2Norm(dim=2)(cache["k"])` written as a `[1, block_size, 1]` per-token field, then reduced or used as a multiplier |
| QUEST envelope | `CMin(dim=1)` + `CMax(dim=1)` of `cache["k"]` → indexer combines with `q` via `Multiply`/`Maximum`/`Sum` |
| Static streaming heads (LServe Λ-mask) | `vortex_layers_skip` (closest available analogue at layer granularity; head granularity not supported) |
| Layer-level static sparsity | `vortex_layers_skip = [0]` or `[0, 4, 8, …]` |
| Block-importance via mean (universal recipe) | `CMean(dim=1)(cache["k"], cache["centroids"])` |
| **Multi-resolution block importance (Prism fix for mean+RoPE)** | `CMeanInterleave(dim=1, k=block_size/g)` → `[1, g, D]` mini-centroids, indexer folds via `Max(dim=1)` |
| **High-frequency-preserving summary (Prism fix)** | pair `CMean` with `CL2Norm` and `CMax`; combine in indexer with `Add`/`Maximum` |
| Approximate top-k (MagicPIG, IceFormer) | `approxTopK(tolerate_ratio=0.05–0.15)` instead of `topK` |
| Cross-page distribution normalization | `Normalize(dim=0)` after the score `GeMM` to undo OOD inflation |
| Cartesian / multi-head × multi-centroid score grid | `Kron(dim=(1,2))` on a per-head q-summary × per-page multi-centroid field, then `Max(dim=1)` |
| Running-V residual (SparQ step 3) | not implementable in current framework — engine consumes only the topK indices |
| ANNS / LSH on CPU (RetrievalAttention, MagicPIG) | not implementable — no CPU offload, single fused kernel |

## 13. Things vortex_torch can't do (don't waste batches on these)

- **Per-token selection finer than a page.** Page granularity is
  baked in; there is no API to drop tokens within a page.
- **Per-head static masking** (LServe's "streaming heads" /
  StreamingLLM applied per-head). Layer skipping exists; head
  skipping does not.
- **CPU offload** of K or V (RetrievalAttention, ShadowKV-V,
  Double-Sparsity-offload). The compiled kernel reads from GPU
  memory only.
- **ANNS / LSH retrieval.** Same reason — no host-side kernel
  participation.
- **Online channel-importance calibration** (Double Sparsity's
  offline step). Calibration data has no place in the flow's
  `__init__`; you can only bake in a *constant* projection / mask.
- **Per-query channel selection for `q`** (SparQ step 1). The score-
  side `MaskSlice` is static. The dynamic version would need
  query-conditional indexing into channels, which isn't an op.
- **Running-V residual reconstruction** (SparQ step 3). The engine
  uses the top-k indices only — there is no place to inject a
  reconstructed-mass value.
- **Fine-tuning anything.** Every flow is post-training only.
- **Adding new ops to the kernel codegen during a session.** Ops are
  fixed at framework build time; the legal toolset is what's
  exported from `vortex_torch.indexer` / `vortex_torch.cache`.

If a submission idea requires any of the above, it is a
*framework-level proposal*, not a submission. File it as backlog and
focus the batch on something the engine can actually compile.

## 14. Submission idea catalogue

A backlog of orthogonal-knob ideas you can pull from when designing
the next batch. Each entry names the paper that motivates it and
sketches the cache + indexer ops involved.

### 14.1 Dual-band centroid (Prism — high priority)

**Hypothesis.** Pure-`CMean(dim=1)` centroids are blind to RoPE
high-frequency / slash patterns. A flow that pairs `CMean` with a
high-frequency-preserving summary will keep quality on
position-sensitive aggregation while losing nothing on retrieval.

**Cache.**
```python
self.k_mean = CMean(dim=1)
self.k_mini = CMeanInterleave(dim=1, k=block_size // 4)   # [1, 4, D]
self.k_norm = CL2Norm(dim=1)                              # [1, 1, D]
```

**Indexer.**
```python
score_mean = self.gemm_mean(q_mean, cache["centroids"], ctx=ctx)        # [S, 1, 1]
score_mini = self.gemm_mini(q_mean, cache["mini_centroids"], ctx=ctx)   # [S, 4, 1]
score_mini_max = self.max_over_mini(score_mini, ctx=ctx)                # [S, 1, 1]
score_norm = self.mul_norm(q_mean, cache["k_norm"], ctx=ctx)            # [S, 1, D]
score_norm_sum = self.sum_norm(score_norm, ctx=ctx)                     # [S, 1, 1]
score = self.add_three(self.add_two(score_mean, score_mini_max),
                       score_norm_sum, ctx=ctx)
self.topk(score, o, ctx=ctx)
```

### 14.2 Channel-projected centroid (Double Sparsity, ShadowKV)

**Hypothesis.** Scoring against a low-rank `r`-dim centroid is as
accurate as full-`D` scoring while saving bandwidth on every step.
Drop `r` from `D` to `D/4`, `D/8`, `D/16` across the batch.

**Cache.**
```python
self.W_proj = ...   # [r, D] constant baked into __init__
self.proj = CGeMM()
# proj-centroid:  [1, r, D] @ [1, 1, D] → [1, r, 1] per block? — see signature
```

**Indexer.** Project `q` through the same `W_proj`, then dot product
in `r`-dim. Variants in the batch differ in `r` only.

### 14.3 QUEST-plus-magnitude (MagicPIG, RetrievalAttention)

**Hypothesis.** Augment QUEST's `(min, max)` envelope with `L2Norm`
to penalize pages whose values are small even when their key envelope
clips `q` favorably. Robust to Q/K OOD.

**Cache.** `CMin(dim=1)`, `CMax(dim=1)` on `k`; `CL2Norm(dim=1)` on
`v`.

**Indexer.** Standard QUEST score, then `Multiply` with a normalized
v-norm summary, then `Sum(dim=2)`.

### 14.4 Multi-head × multi-centroid score grid (Kron)

**Hypothesis.** Maintain `C` centroids per page (multi-mode keys,
e.g. via `CMeanInterleave`) and score them against all `H` query
heads simultaneously via `Kron(dim=(1,2))`, then `Max(dim=1)` over
heads to pick the best-aligned head per page.

**Cache.** `CMeanInterleave(dim=1, k=block_size // C)` → `[1, C, D]`.

**Indexer.**
```python
kron_score = self.kron(q_per_head, cache["multi_centroids"], ctx=ctx)  # [S, H, C]
score_per_page = self.max_over_h(kron_score, ctx=ctx)                  # [S, 1, C]
score = self.max_over_c(score_per_page, ctx=ctx)                       # [S, 1, 1]
```

### 14.5 Running attention-mass accumulator (H2O, Keyformer)

**Hypothesis.** A page's *cumulative* attention score across decode
steps is a better signal than its score on the current step alone.

**Cache.** `CFill(0.0)` on a `[1, 1, 1]` accumulator field at each
block completion (the indexer reads-then-writes it).

**Indexer.**
```python
prev = self.load(cache["accum"], ctx=ctx)                              # [S, 1, 1]
new  = self.score_step(...)                                            # [S, 1, 1]
mixed = self.add_decay(prev, new, ctx=ctx)                             # Add(α=0.9, β=1.0)
self.save(mixed, cache["accum"], ctx=ctx)
self.topk(mixed, o, ctx=ctx)
```

**Don't forget:** `disable_radix_cache: true` in the engine JSON
because of `Save`. Pre-flight will reject a violation.

### 14.6 Per-page magnitude correction (RetrievalAttention / MagicPIG OOD note)

**Hypothesis.** Raw `q · centroid` over-rewards pages whose centroid
happens to have large magnitude — an OOD artifact, not real
relevance. Compensate by **dividing** the dot-product by the
centroid's L2 norm (i.e. score via cosine similarity instead of
inner product). Unlike subtracting the per-sequence mean, this
genuinely re-orders pages because the magnitude varies per page.

**Cache.**
```python
self.k_mean = CMean(dim=1)                  # [1, block_size, D] → [1, 1, D]
self.k_norm = CL2Norm(dim=2)                # [1, 1, D] → [1, 1, 1]   (norm of centroid)
# Apply CL2Norm to the centroid output, not to cache["k"] directly.
```

**Indexer.**
```python
dot     = self.gemm(q_mean, cache["centroids"], ctx=ctx)               # [S, 1, 1]
# Division isn't directly an op; do reciprocal-multiply via a fixed scale,
# or — equivalently and cheaper — multiply the score by 1 / norm using
# a Multiply against a precomputed inverse-norm field (compute the inverse
# in forward_cache as Exp(Log(norm) * -1.0) if no Reciprocal op exists).
inv_norm = self.inv_norm(cache["centroid_inv_norm"], ctx=ctx)          # [S, 1, 1]
cos_sim  = self.scale(dot, inv_norm, ctx=ctx)                          # Multiply
self.topk(cos_sim, o, ctx=ctx)
```

**Caveat — no monotonic-only post-processing.** Anything you put
between `gemm` and `topk` must change the **relative ordering** of
pages, not just shift them by a constant or scale them all by the
same factor. Subtracting `Reduce(dim=0)(score)` (a single scalar
per sequence) does *not* qualify and gives the same picked set as
no centering at all. Multiplying by a per-page signal (norm,
heavy-hitter accumulator, value mass) does qualify because the
multiplier varies across pages.

### 14.7 Approximate top-k sweep (MagicPIG / IceFormer rationale)

**Hypothesis.** With the right score function, `approxTopK` matches
`topK` quality at lower cost. Sweep `tolerate_ratio` along
`{0.0, 0.05, 0.1, 0.15, 0.25, 0.4, 0.6, 0.8}` (one per GPU) at fixed
score function and budget.

A pure throughput-vs-quality knob; cheapest possible orthogonal-batch
to design.

### 14.8 Static layer-skip × dynamic page-select (LServe)

**Hypothesis.** Layer-skip and page-select speedups multiply. A
batch that varies `vortex_layers_skip` ∈ `{[0], [0,1], [0,2,4],
[0,1,2,3,4,5,6,7], …}` at fixed flow + topk should show monotonic
throughput improvement until `mean@16` snaps below floor.

Pure config-axis batch, no Python changes per variant — just JSON.

### 14.9 BOS-budget × topk-budget grid (StreamingLLM × dynamic)

**Hypothesis.** Increasing `vortex_block_reserved_bos` past 1 page
buys quality on long-context tasks; decreasing `vortex_topk_val`
buys throughput. The optimal is somewhere on the diagonal.

Two-axis sweep: BOS ∈ `{1, 2, 4, 8}`, topk ∈ `{32, 48, 64, 96}` —
take the 8 cells that bracket the current best.

---

## 15. How to use this guide in a batch design

§14 is a *floor*, not a ceiling. The protocol for a batch is:

1. **Pick an axis from §11** that the current best flow doesn't
   exploit (channel sparsity is the most underused).
2. Either:
   - Pick a paper from §1–§10 and translate it via §12 (then pull a
     sketch from §14), **or**
   - Pick a prompt from §16 and invent your own answer.
3. If the idea lives in §13's "can't do" list, file it as
   framework-level backlog and try again.
4. Build `N` orthogonal variants. **Required composition:**
   - **At least 1 off-catalog variant** — an idea you cannot
     attribute to any single paper here. Could be a combination
     nobody has explored, a knob nobody has tried, an inversion of
     a paper's claim, or a flow you derived from first principles.
     Document the hypothesis in one sentence in `algorithm_scientist/memory.md`
     §3 alongside the batch row.
   - The remaining `N - 1` slots should sweep one knob — rank,
     tolerate_ratio, layers_skip, decay, group size `k`, projection
     dimension, etc. — to localize the effect.
5. Pre-flight all `N`, launch with `/batch-benchmark`, and record the
   batch in `algorithm_scientist/memory.md` §1.

The shortest path to a *measurable* win, given the current flow
library, is **§14.1 (Prism dual-band centroid)** — its underlying
fix is theoretically forced rather than empirically observed. The
shortest path to a *novel* win is somewhere in §16.

---

## 16. Going beyond the catalog — prompts for inventing new flows

Every paper in §1–§10 was built on a gap the previous literature
hadn't named. The framework you're working with — page-level
selection, fused per-block kernel, Save/Load across decode steps,
and now (recently added) `Kron` and `MeanInterleave` — opens
combinations the public literature does not cover. The questions
below are not rhetorical; each one is a research direction you can
answer with a batch.

### 16.1 Combinations the literature has not bothered to publish

- **Prism (high-freq pooling) × Heavy-hitter accumulator (Keyformer/H2O).**
  Prism shows that mean-pool centroids miss slash patterns;
  heavy-hitter scores accumulate cross-step signal. Nobody has
  combined a slash-aware centroid with a running attention-mass
  accumulator. Does the accumulator pick up *different* pages than
  the dual-band centroid?
- **Channel sparsity (Double Sparsity) × Multi-resolution centroids
  (Prism).** Reduce the centroid's feature dimension *and* split it
  into mini-centroids — the cost goes down twice; does the quality?
- **Kron-grid score (multi-head × multi-centroid) × low-rank K
  (ShadowKV).** The grid is `[S, H, C]`; if the centroid is a
  rank-`r` projection, the GeMM dimension drops from `D` to `r`.
  Multiplicative throughput win, untested at this granularity in
  any paper here.
- **Static Λ-mask (StreamingLLM/LServe at the head level) × dynamic
  page-select.** vortex_torch can't do per-head static masking, but
  it *can* do per-layer static masking — does running an
  `vortex_layers_skip = [0, 1]` flow with a *deeper* dynamic
  selector recover the per-head lossiness?

### 16.2 Knobs no paper has explored

- **Per-page temperature.** Every score function in the literature
  treats all pages identically. Does scaling each page's contribution
  by its `L2Norm(v_summary)` *before* the topk — i.e. softmaxing
  the relevance against the page's own value-mass budget — change
  the picked set?
- **Adaptive top-k budget per layer.** Layer skip says "first few
  layers run dense, the rest run sparse." Nobody publishes per-layer
  *budgets* (e.g. layer 4 picks 96 pages, layer 12 picks 32). The
  framework's per-layer compilation makes this trivial to express.
- **Pre-RoPE vs post-RoPE centroid comparison.** ShadowKV exploits
  the low-rank structure of pre-RoPE keys; Prism's blind-spot is on
  post-RoPE. What if `forward_cache` builds *both* a pre-RoPE
  proxy (e.g. `CMean` of `cache["k"]` plus a `CL2Norm` correction)
  and a post-RoPE one, and the indexer picks based on the
  disagreement between them?
- **Multi-stage selection.** Two cheap top-`k1` selectors with
  different signals (centroid score vs heavy-hitter accumulator)
  feeding into a final top-`k2 < k1` chooser. Cuts the work the
  expensive selector does without dropping recall.

### 16.3 Inversions — claims worth testing for falsification

- "Attention is *not* always sparse" (MagicPIG). Try the inverse:
  flows that *deliberately keep* a noisy second-tier of pages
  (e.g. top-k from a less-sharp score) instead of relying on a
  single sharp picker. Does this rescue accuracy on aggregation
  tasks while staying cheap?
- "Channel importance is largely static" (Double Sparsity). Try a
  flow where the channel mask **drifts** across decode steps via a
  `Save`/`Load`-tracked counter of which channels contributed most
  recently. Empirically: does it actually drift, or does Double
  Sparsity's "static" claim survive on this engine?
- "BOS sinks are necessary" (StreamingLLM). The framework reserves
  them by default — but what if the score function explicitly
  *down-weights* the sink (knowing the framework will keep it
  anyway), so the rest of the topk budget goes to genuinely
  informative pages? Does that buy throughput at fixed quality?

### 16.4 First-principles questions

- What is the **cheapest possible flow** that still clears the
  quality floor? Strip ops one at a time from the current best;
  the first thing that breaks is the load-bearing piece.
- What is the **flow with the smallest cache footprint** that
  clears the floor? Each cache field costs decode-time bandwidth.
- What is the **flow with the fewest indexer ops** (count
  `__init__` declarations) that clears the floor? Selector
  simplicity is itself a throughput knob (LServe §8).
- What scoring function would make `approxTopK(tolerate_ratio=0.5)`
  match `topK` quality? If you can bound the score in `[0, M]`
  with a known shape, the radix algorithm prunes faster.
- What signal in `cache["v"]` does no paper here use? Most papers
  score on `k` and treat `v` as the carried payload. Is there a
  v-derived score that's actually predictive?

### 16.5 Anti-prompts — things that look novel but aren't

To avoid wasted batches, these are *not* novel directions, even if
they feel like they are:

- "Use a smaller `vortex_topk_val`." Pure throughput knob, sweep it
  in the catalog batch (§14.7 / §14.9).
- "Try `approxTopK` instead of `topK`." Same — sweep it (§14.7).
- "Skip more layers." Sweep it (§14.8).
- "Switch to fp8 KV cache." Engine knob, not a flow change.
- **"Add a monotonic transform between the score and `topK`."**
  `topK` selects the largest `k` entries by *order*, so any
  function that preserves order (subtracting a per-sequence mean,
  scaling by a single scalar, applying any monotonically
  increasing element-wise op like `Exp` / `Add(α=1, β=-c)` /
  `Add_Mul(α=0, β=c)` for `c > 0`) returns the **same picked
  set**. To change selection you need a transform whose effect
  *varies per page* — multiply by a per-page magnitude, add a
  per-page bias from a different signal, threshold against a
  per-page floor, etc. See §14.6 for an example that does this
  correctly.

These are *parameter sweeps* or no-ops, valuable as orthogonal-knob
slots in a batch but never as the off-catalog slot.

---

The literature is a starting point. Read it, internalize the patterns
in §11, and then — every batch — earn at least one variant by
asking what nobody has tried.
