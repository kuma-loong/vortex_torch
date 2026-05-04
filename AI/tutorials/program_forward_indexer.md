# Writing `forward_indexer` — the single-sequence mental model

This is the first tutorial you should read before writing a `vFlow`.
The framework takes care of the real batching, paging, and scheduling
details under the hood — **you don't need any of that to write a correct
`forward_indexer`**. This file stays entirely in the user's view.

---

## 1. The mental model: one sequence, one head group

> Pretend the system contains exactly **one** sequence and exactly **one
> kv-head group**. Write your code as if it only has to compute a score
> for that single sequence.

Under this fiction:

| tensor | logical shape | what it is |
|---|---|---|
| `q` | `[1, H_q, D]` | the current query (one position). `H_q` is the number of query heads grouped to one kv head; `D` is the feature dim |
| every `cache["<name>"]` | `[S, D_0, D_1]` | a per-page field with `S` pages laid out along the leading axis. `(D_0, D_1)` is whatever inner shape you declared in `create_cache` |
| `o` | opaque | a buffer the framework hands you. Your only job is to call `topK(score, o, ctx)` (exact) or `approxTopK(tolerate_ratio=…)(score, o, ctx)` (faster, approximate) to fill it |

That is the whole surface. You describe the single-sequence compute and
the framework takes care of running it everywhere it needs to run.

---

## 2. The goal (the only constraint you must satisfy)

Produce a **`[S, 1, 1]` score tensor** — one scalar per page — and hand
it to `topK` (or its faster approximate variant `approxTopK`):

```python
self.output_func(score, o, ctx=ctx)   # score.shape == [S, 1, 1]
```

`topK` selects the top-`k` pages per sequence using the score and writes
the surviving page indices to `o`. Downstream, flashinfer attends only
to those pages.

Two terminal-op choices:

- **`topK()`** — exact top-k (sorted output). Use unless `topK` cost
  shows up as a bottleneck.
- **`approxTopK(tolerate_ratio=0.0..1.0)`** — adaptive 8-bit radix
  variant with a quality / cost knob. `0.0` = exact, `1.0` =
  cheapest-possible single-round, in between = adaptive. Same
  `(score, o, ctx=ctx)` call shape; same BOS/EOS reservation;
  output indices unsorted within each segment. See
  `indexer_op.md §9` for the full math + tuning guidance.

Every op between `forward_indexer(self, q, o, cache, ctx)` and
`self.output_func(score, o, ctx)` just transforms `[S, D_0, D_1]` tensors
via the standard NumPy-style broadcasting rules until you reach
`[S, 1, 1]`. That's the whole programming model.

---

## 3. How the ops behave on these logical shapes

Exactly like you'd expect from plain PyTorch / NumPy:

| op | signature | example |
|---|---|---|
| `Multiply()(x, y)` | elementwise product with broadcasting | `[1, H_q, D] * [S, 1, D] → [S, H_q, D]` |
| `Add(α, β)(x, y)` | `α*x + β*y`, broadcasting | `[S, 1, 1] + [1, 1, 1] → [S, 1, 1]` |
| `Maximum()(x, y)` | elementwise max | `[S, H_q, D], [S, H_q, D] → [S, H_q, D]` |
| `Mean(dim=k)(x)` | reduce over `dim=k`, keepdim | `Mean(dim=1)([1, H_q, D]) → [1, 1, D]` |
| `Sum(dim=2)(x)` | reduce over `D` | `[S, H_q, D] → [S, H_q, 1]` |
| `Max(dim=1)(x)` | reduce over `H_q` | `[S, H_q, 1] → [S, 1, 1]` |
| `GeMM()(x, y)` | `y @ x.T` per S-slice: `[*, N_x, K] · [S, N_y, K] → [S, N_y, N_x]` | `[1, 1, D] · [S, 1, D] → [S, 1, 1]` |
| `MaskSlice(start, end, dim, α, β)(x)` | position-based mask: write `α` where `start ≤ i < end`, `β` elsewhere along `dim` | shape-preserving |

Two broadcasting rules you only have to remember:

1. **The leading `1` on `q` broadcasts against the leading `S` of the
   cache.** That's the "one query scoring every page" motion — `q *
   cache['centroids']` lifts `q` to `[S, H_q, D]`.
2. **Any tensor with a dim equal to 1 broadcasts against the same dim
   elsewhere.** Exactly NumPy. You'll mostly use this on the reduction
   axes (`H_q → 1` or `D → 1`) to collapse intermediates into `[S, 1, 1]`.

---

## 4. Worked example: `block_sparse_attention`

The simplest non-trivial flow. Score a page by the cosine of its
centroid against the mean query.

```python
def forward_indexer(self, q, o, cache, ctx):
    q_mean = self.mean(q, ctx=ctx)                           # [1, 1, D]
    score  = self.gemm(q_mean, cache["centroids"], ctx=ctx)  # [S, 1, 1]
    self.output_func(score, o, ctx=ctx)                      # write top-k into o
```

Reading line-by-line in the mental model:

- `q` enters as `[1, H_q, D]`.
- `self.mean(q, ctx=ctx)` uses `Mean(dim=1)` — reduces the query-head
  axis, giving `[1, 1, D]`: the mean query vector.
- `cache["centroids"]` is `[S, 1, D]`.
- `self.gemm([1, 1, D], [S, 1, D])` computes `y @ x.T` per S-slice:
  `N_x = 1, N_y = 1`, output `[S, 1, 1]`. One dot-product per page.
- Hand the `[S, 1, 1]` score to `topK`. Done.

No loop, no branch, no bookkeeping. You wrote attention routing in
three lines.

---

## 5. Worked example: `gqa_block_sparse_attention`

Like §4, but keeps the full query-head axis: every query head scores every
page against the centroid, and we aggregate across query heads with `Max`.
Adds an in-place `Softmax` over the page axis so the scores are
comparable across pages of the same sequence.

```python
def forward_indexer(self, q, o, cache, ctx):
    score       = self.gemm(q, cache["centroids"], ctx=ctx)    # [S, 1, H_q]
    normalized  = self.softmax(score, ctx=ctx)                 # [S, 1, H_q]  (softmax over pages)
    aggr        = self.max_op(normalized, ctx=ctx)             # [S, 1, 1]    (max over query heads)
    self.output_func(aggr, o, ctx=ctx)
```

Reading each line:

- `q` is `[1, H_q, D]`, `cache["centroids"]` is `[S, 1, D]`.
- `GeMM([1, H_q, D], [S, 1, D])` uses `N_x = H_q, N_y = 1`, so the result
  is `[S, 1, H_q]` — one dot-product per `(page, query_head)`.
- `Softmax(dim=0)` normalises across pages for each query head.
- `Max(dim=2)` aggregates across query heads, giving `[S, 1, 1]`.
- Hand to `topK`.

Two things to notice vs §4:

1. The GeMM here keeps the query-head axis in the output (`N_x = H_q`)
   instead of collapsing it first with a `Mean`. The choice is purely
   about what the scoring function should be — both shapes are legal.
2. `Softmax(dim=0)` is a normalisation step. The framework supports
   `Softmax` along the page axis as a single op; you don't write the
   `exp` / `sum` / `divide` expansion yourself.

---

## 6. Worked example: `gqa_quest_sparse_attention`

QUEST bounds the attention score per page using element-wise `max`/`min`
envelopes of the keys.

```python
def forward_indexer(self, q, o, cache, ctx):
    s_max  = self.mul_max(q, cache["max"], ctx=ctx)     # [1, H_q, D] * [S, 1, D] → [S, H_q, D]
    s_min  = self.mul_min(q, cache["min"], ctx=ctx)     # same
    s      = self.maximum_op(s_max, s_min, ctx=ctx)     # [S, H_q, D]
    score  = self.sum(s, ctx=ctx)                       # Sum(dim=2) → [S, H_q, 1]
    aggr   = self.max_op(score, ctx=ctx)                # Max(dim=1) → [S, 1, 1]
    self.output_func(aggr, o, ctx=ctx)
```

Again, every intermediate has a shape you can predict by staring at the
line. `q` broadcasts against each cached envelope; `maximum_op` combines
them page-wise; `sum` collapses `D`; `max` collapses `H_q`. Arrive at
`[S, 1, 1]` and call `topK`.

**Note on reused semantics:** `self.mul_max` and `self.mul_min` are *two
separate* `Multiply()` instances even though they do the same thing.
Don't share a single `Multiply` between both call sites — see the
pitfalls section.

---

## 7. Worked example: `masked_quest_sparse_attention`

QUEST + a position-dependent feature mask. Suppresses the first few
feature planes of the envelope score — a cheap way to down-weight
low-signal dims without changing the cache side.

```python
# MASK_END = 8 in __init__:
#   self.feature_mask = MaskSlice(start=0, end=MASK_END, dim=2, α=0.0, β=1.0)

def forward_indexer(self, q, o, cache, ctx):
    s_max    = self.mul_max(q, cache["max"], ctx=ctx)       # [S, H_q, D]
    s_min    = self.mul_min(q, cache["min"], ctx=ctx)       # [S, H_q, D]
    s        = self.maximum_op(s_max, s_min, ctx=ctx)       # [S, H_q, D]
    mask     = self.feature_mask(s, ctx=ctx)                # [S, H_q, D]  — 0 in [0, 8), 1 elsewhere
    masked_s = self.mul_mask(s, mask, ctx=ctx)              # [S, H_q, D]  — first 8 features zeroed
    score    = self.sum(masked_s, ctx=ctx)                  # [S, H_q, 1]
    aggr     = self.max_op(score, ctx=ctx)                  # [S, 1, 1]
    self.output_func(aggr, o, ctx=ctx)
```

New op: **`MaskSlice(start, end, dim, α, β)`** writes `α` to positions
`[start, end)` along `dim` and `β` everywhere else, preserving the input's
shape. With `α=0, β=1` it's a "zero-out range, keep the rest" feature
mask; with `α=1, β=0` it's "keep only this range". Because `MaskSlice`'s
output has the same shape as its input, a follow-up `Multiply` combines
it with any tensor you've already built up — here, the QUEST envelope
`s`.

---

## 8. Worked example: `centered_block_sparse_attention` — per-sequence summary

Sometimes a page's score should depend on *all* the pages of the
sequence — e.g. "drop pages whose score is below the sequence mean".
That's the one case where the single-sequence fiction needs one extra
op: **`Reduce(dim=0)`** (`Mean(dim=0)` / `Max(dim=0)` / etc.).

`Reduce(dim=0)` collapses the `S` axis, giving you a `[1, D_0, D_1]`
logical tensor that broadcasts back against any per-page tensor in
normal binary ops (`Add`, `Multiply`, `WhereGreater`, …).

> **Important — `topK` is invariant to monotonic transforms.** Doing
> `score - mean_seq` (an `Add(α=1, β=-1)` against a `[1,1,1]` shift)
> and feeding the result straight to `topK` picks the **same pages**
> as feeding `score` directly. Subtracting one scalar from every page
> is order-preserving. The reference flow
> `centered_block_sparse_attention` was written that way and is
> equivalent to plain `block_sparse_attention` — useful as a pedagogical
> illustration of `Reduce(dim=0)`'s machinery, *not* a recommended
> recipe.

To make a per-sequence statistic actually change the picked set, thread
it through a **non-monotonic** op (a threshold gate, a per-page
multiplier from a different signal, etc.):

```python
def forward_indexer(self, q, o, cache, ctx):
    s        = self.mul(q, cache["centroids"], ctx=ctx)   # [S, H_q, D]
    score_d  = self.sum_d(s, ctx=ctx)                     # Sum(dim=2) → [S, H_q, 1]
    score    = self.mean_h(score_d, ctx=ctx)              # Mean(dim=1) → [S, 1, 1]
    mean_seq = self.mean_seq(score, ctx=ctx)              # Mean(dim=0) → [1, 1, 1]
    above    = self.cmp(score, mean_seq, ctx=ctx)         # WhereGreater → 0 / -1e30  [S, 1, 1]
    gated    = self.add_gate(score, above, ctx=ctx)       # Add(α=1, β=1) → masks below-mean to ~-inf
    self.output_func(gated, o, ctx=ctx)
```

Reading line-by-line:

- Build a per-(page, query-head) raw score: `q * centroids`,
  `Sum` over `D`.
- Collapse query heads with `Mean(dim=1)` → one score per page `[S, 1, 1]`.
- `Mean(dim=0)` on that → one scalar for the whole sequence `[1, 1, 1]`.
- `WhereGreater(score, mean_seq)`: `0` for pages above the sequence
  mean, `-1e30` for the rest. This is a **per-page** decision, not a
  constant shift.
- `Add(α=1, β=1)` adds the mask to the score, sending below-mean
  pages to ~`-inf` while leaving above-mean pages unchanged.
- Hand to `topK` — now the picked set genuinely depends on the
  per-sequence statistic.

`Reduce(dim=0)` is a real and valuable primitive; the only caveat is
that you have to consume it through something that changes order,
not just shifts it.

---

## 9. Persistent state across decode steps: `Save` / `Load`

`Save` and `Load` give `forward_indexer` **persistent memory across
decode steps**: values you compute this step can be stashed somewhere
the next step will see them. This is how you build running statistics,
learned routing state, EMAs over past queries, etc.

**The only rule:** every `Save` target and every `Load` source must be a
tensor **declared in `create_cache`**. You don't get free-floating
persistent buffers — if you want something to survive across steps,
give it a name in `create_cache` alongside `centroids`, `max`, etc.,
then read it with `Load` and write it with `Save`.

### Worked example: running-average page score

Pages routed with a momentum term. Each decode step, we compute the
**current** per-page score (say, via `q · centroid`), fuse it with the
stored **running** score using

```
running_score ← α * last_running_score + current_score
```

then run `topK` on the updated running score and persist it for the next
step. A page that keeps scoring highly accumulates; a page whose
relevance decayed gets overtaken.

```python
from vortex_torch.flow import vFlow, register
from vortex_torch.indexer import Mean, GeMM, Load, Add, Save, topK
from vortex_torch.cache import Mean as CMean, Fill as CFill

@register("running_avg_block_sparse")
class RunningAvgBlockSparse(vFlow):
    ALPHA = 0.9  # momentum — higher means history matters more

    def __init__(self):
        super().__init__()
        # Indexer-side ops
        self.mean        = Mean(dim=1)
        self.gemm        = GeMM()
        self.load_score  = Load()
        # fuse(last_running, current) = ALPHA * last_running + 1.0 * current
        self.fuse        = Add(alpha=self.ALPHA, beta=1.0)
        self.save_score  = Save()
        self.output_func = topK()
        # Cache-side ops
        self.reduction          = CMean(dim=1)    # centroids = mean of the block's keys
        # Zero-initialise the persistent per-block scalar when the block
        # completes. Without this, the first ``Load`` on a newly
        # allocated block reads uninitialised memory (stale bytes from
        # whatever lived there before).
        self.init_running_score = CFill(alpha=0.0)

    def create_cache(self, block_size, head_dim):
        return {
            "centroids":     (1, head_dim),  # maintained by forward_cache
            "running_score": (1, 1),         # zeroed by forward_cache, accumulated by forward_indexer
        }

    def forward_indexer(self, q, o, cache, ctx):
        q_mean       = self.mean(q, ctx=ctx)                          # [1, 1, D]
        current      = self.gemm(q_mean, cache["centroids"], ctx=ctx) # [S, 1, 1]
        last_running = self.load_score(cache["running_score"], ctx=ctx)  # [S, 1, 1]
        running      = self.fuse(last_running, current, ctx=ctx)      # α*last + current, [S, 1, 1]
        self.save_score(running, cache["running_score"], ctx=ctx)     # persist for next step
        self.output_func(running, o, ctx=ctx)

    def forward_cache(self, cache, loc, ctx):
        self.reduction(cache["k"], cache["centroids"], loc=loc, ctx=ctx)
        self.init_running_score(cache["running_score"], loc=loc, ctx=ctx)
```

Reading the `forward_indexer` body:

- `current` is a fresh per-page score from the usual `q_mean · centroid`
  pipeline — nothing novel, shape `[S, 1, 1]`.
- `Load(cache["running_score"])` reads the value saved at the **end of
  the previous decode step**. First-step value is whatever the field was
  initialised to (typically zeros).
- `Add(α=ALPHA, β=1.0)(last_running, current)` computes the formula
  `α*last + 1*current`. Operand order matters — `Add`'s semantics are
  `α*x + β*y`, so `x = last_running, y = current`.
- `Save(running, cache["running_score"])` writes the new running score
  back. The next decode step's `Load` will see this value.
- Route on the updated running score.

### Notes

- **Why the `CFill(0.0)` on the cache side?** `forward_cache` fires
  once each time a block is newly *completed*. At that moment the
  block's `running_score` slot is still holding whatever bytes happened
  to live there before (the cache pool is reused across sequences).
  Zeroing the slot at block-completion time makes the very first
  `Load` on that block see `0`, which together with `Add(α=0.9, β=1)`
  gives `running = 1·current` on the first-use step — momentum kicks
  in from the next step onward.
- **Operand order in `Add`.** `Add(alpha=α, beta=β)(x, y)` is
  `α*x + β*y`, not `α*(x+β*y)`. For "add current to a momentum-weighted
  history" you want `α = history weight, β = 1.0` and the order
  `(history, current)`. Swap the order and the weights follow.
- **Two writers, different fields.** `cache["centroids"]` is written
  by `forward_cache` (from the just-landed keys); `cache["running_score"]`
  is zero-initialised by `forward_cache` and then maintained by
  `forward_indexer` via `Save`. Each field has exactly one "primary
  writer" on each side — `create_cache` just declares the slots;
  where the writes happen is your design choice.

### Other typical uses of `Save` / `Load`

- **EMA of centroids** computed from past queries instead of past keys.
- **Usage counters** that bias routing away from over-used pages
  (`Add` the one-hot top-k picks into a counter; decay the counter
  with a running-average step like above).
- **Running max / running argmax** over past queries.
- **Learned router state** that evolves with the generation.

If you don't need state to persist across steps, don't use `Save`/`Load`
— just let ops compose; the framework handles intermediates for you.

---

## 10. Choosing the terminal op: `topK` vs `approxTopK`

Every `forward_indexer` ends with one terminal call that consumes a
`[S, 1, 1]` score and writes the chosen page-id set into `o`. You
have **two** choices:

### `topK()` — exact, the default

```python
self.output_func = topK()
...
self.output_func(score, o, ctx=ctx)
```

Picks the strictly top-`k` pages by descending score. `k` is the
runtime constant `ctx.topk_val` (configured via `vortex_topk_val` /
`vortex_topk_ratio`); the call doesn't take a `k` argument. BOS / EOS
reservations are layered on automatically. **Use this unless `topK`
shows up as a measurable cost in the indexer path.**

### `approxTopK(tolerate_ratio)` — adaptive radix, faster

Drop-in replacement with a quality / cost knob:

```python
self.output_func = approxTopK(tolerate_ratio=0.20)
...
self.output_func(score, o, ctx=ctx)   # SAME call shape
```

Runs an adaptive 8-bit radix top-k (up to four 8-bit refinement
rounds = 32 bits). After each round, if the slots still owed by the
threshold bin are within `tolerate_ratio · target_k`, the kernel
**stops early** and fills those slots in arrival order from the
current candidate set. The trade-off is a single dial:

| `tolerate_ratio` | behavior | when to pick |
|---|---|---|
| `0.0` | all 4 rounds always run; bit-exact top-k | parity test against `topK()` |
| `0.05 - 0.15` | adaptive: cheap on well-separated scores, refines on tight ones | **typical sweep range** for throughput hunting |
| `0.50 - 1.0` | aggressive early-exit; selection becomes coarse | scores have a clear gap between selected and dropped pages, and you'll trade quality for speed |

Same per-segment contract as `topK`: BOS / EOS preserved, exactly
`topk_val` chosen. Only the *selection quality* moves; the slot
**count** is identical.

**Output ordering caveat.** `approxTopK` emits indices in
**unsorted** order within each segment (matches the underlying
`topk_output_v2` C kernel). The framework's downstream consumers
treat the kv-indices as a set, so you don't need to care — but if
you ever inspect the `kv_indices` tensor by hand, don't assume
sorted.

### Worked example: GQA-Quest with approximate selection

Taking [`gqa_quest_sparse_attention`](../../vortex_torch/flow/algorithms.py)
and swapping the terminal op:

```python
from vortex_torch.indexer import (
    approxTopK, Multiply, Maximum, Sum, Max,
)
from vortex_torch.cache import Max as CMax, Min as CMin

@register("gqa_quest_approx_cls")
class GQAQuestApprox(vFlow):
    def __init__(self):
        super().__init__()
        self.mul_max  = Multiply()
        self.mul_min  = Multiply()
        self.maximum  = Maximum()
        self.sum      = Sum(dim=2)
        self.max_op   = Max(dim=1)
        self.output_func = approxTopK(tolerate_ratio=0.1)   # ← only difference
        self.cmax = CMax(dim=1)
        self.cmin = CMin(dim=1)

    def create_cache(self, block_size, head_dim):
        return {"max": (1, head_dim), "min": (1, head_dim)}

    def forward_cache(self, cache, loc, ctx):
        self.cmax(cache["k"], cache["max"], loc=loc, ctx=ctx)
        self.cmin(cache["k"], cache["min"], loc=loc, ctx=ctx)

    def forward_indexer(self, q, o, cache, ctx):
        s_max = self.mul_max(q, cache["max"], ctx=ctx)
        s_min = self.mul_min(q, cache["min"], ctx=ctx)
        s     = self.maximum(s_max, s_min, ctx=ctx)
        score = self.sum(s, ctx=ctx)
        aggr  = self.max_op(score, ctx=ctx)
        self.output_func(aggr, o, ctx=ctx)
```

The fused score-chain kernel is byte-identical to the `topK`
version; only the terminal call site changes from `topk_output(...)`
to `approx_topk_output(..., 0.25)`. This is the cleanest "free
throughput" lever to sweep across a batch — try `tolerate_ratio ∈
{0.0, 0.05, 0.10, 0.20, 0.30}` against the same score chain.

### When NOT to use `approxTopK`

- For new flows where the score chain itself is uncharacterized —
  use exact `topK()` first to establish an accuracy floor, then
  swap.
- When downstream code or a debug print actually depends on sorted
  output ordering. (Production decode does not — the framework
  treats kv-indices as a set.)

See [`indexer_op.md §9`](indexer_op.md) for the full math reference.

---

## 11. What you DON'T write

Explicitly, the framework handles all of these — your `forward_indexer`
body should never mention them:

- No explicit loops over sequences or heads.
- No index-math or table lookups.
- No Triton / CUDA code.
- No shape branching on runtime sizes.
- No cache allocation, no buffer recycling.
- No ad-hoc per-step scratch buffers. (`Save` / `Load` *are* allowed —
  see §9 — but each `Save` must target a cache tensor declared in
  `create_cache`, and each `Load` must read from one. You can't create
  transient persistent state on the fly.)

If you find yourself wanting one of those, you're off the happy path —
re-read the op you wanted and see whether its logical semantics on
`[S, D_0, D_1]` already do what you want.

---

## 12. Common recipes

| you want | how to do it |
|---|---|
| "dot product with a centroid" | `Multiply` → `Sum(dim=2)` (or a single `GeMM`) |
| "envelope bound" (QUEST) | `Multiply` twice + `Maximum` + `Sum(dim=2)` + `Max(dim=1)` |
| "cosine-like on multiple centroids" (top-k among C centroids) | `GeMM` → `Max(dim=1)` over the centroid axis |
| "gate by per-sequence threshold" (mask below-mean pages to `-inf`) | score → `Reduce(dim=0)` → `WhereGreater` → `Add(α=1, β=1)`. NB: a plain `Add(α=1, β=-1)` *into* `topK` is a **no-op** — `topK` is invariant to constant shifts |
| "softmax over pages" | `Softmax(dim=0)` |
| "mask out a positional range of the query heads" | `MaskSlice(start, end, dim=1, α, β)` then `Multiply` with the score |
| "bias the first/last few pages" (BOS/EOS) | the framework already handles this via `block_reserved_bos` / `block_reserved_eos`; don't recode it |
| "persistent state across decode steps" | declare a field in `create_cache`, then `Load` it at the start of `forward_indexer` and `Save` the update back |

---

## 13. Pitfalls that trip people up

1. **Don't reuse an op instance.** `self.mul = Multiply()` used twice is
   a bug — each call site needs its own instance. Declare
   `self.mul_a = Multiply(); self.mul_b = Multiply()`. (The op instance
   holds buffer metadata from profile time; sharing clobbers it.)
2. **No native torch ops — not even on `q`.** Every tensor you touch
   inside `forward_indexer` must go through `vortex_torch.indexer.*` ops;
   every tensor in `forward_cache` must go through `vortex_torch.cache.*`
   ops. `q`, `o`, and every `cache[...]` are `vTensor`s at profile time —
   calling `.view` / `.contiguous` / elementwise-torch on them won't
   compile. If you need to reshape `q`, use `Mean` / `Sum` / `Max` / a
   `GeMM` with the right `(N_x, N_y)` shape — one of the indexer ops
   already does what you want.
3. **Your chain must end in `[S, 1, 1]`.** If you've got a stray `H_q`
   or `D` axis left over, fold it with `Mean` / `Max` / `Sum` before
   calling `topK` (or `approxTopK`). The dispatch in both ops will
   reject anything else.
4. **`create_cache` declares inner shapes only.** Returning
   `{"centroids": (1, head_dim)}` tells the framework "each block owns
   one vector of size `head_dim`". The leading `S` axis is synthesised
   at runtime.
5. **The cache pipeline and the indexer pipeline share `cache["..."]`
   but run on different grids.** If your indexer reads a field, your
   `forward_cache` must write it. If it doesn't, you'll score pages
   against stale / zero data.

---

## 14. Debugging checklist when a flow compiles but doesn't seem to work

1. Write the shapes as comments after every op:
   `score = self.sum(s, ctx=ctx)  # [S, H_q, 1]`. If you can't predict a
   shape, the op isn't doing what you think it is.
2. Run `verify_flow_compilable(flow)` from `vortex_torch.flow.verify`.
   It sweeps `(G, D, block_size, page_size)` and surfaces dispatch /
   shape errors with a focused traceback. Do this **before** booting
   sglang.
3. Open the compiled file in `~/.vortex_compilation_cache/` (or the
   verify cache dir). Search for `tl.sum` / `tl.max` / `tl.maximum` and
   read the Triton kernel — the shape constants next to each
   `tensor_X_block` tell you what leading dim the compiler inferred.
4. If the result looks "roughly right but scores are off", check
   `forward_cache`: the fields you're scoring on are only as fresh as
   the last block-completion event. A centroid that was never written
   (e.g. because you forgot a `CMean` call) is zero.

---

## 15. Mental-model summary

- `q: [1, H_q, D]`, each `cache["name"]: [S, D_0, D_1]`.
- Compose ops with NumPy-style broadcasting to produce `score: [S, 1, 1]`.
- `topK(score, o, ctx)` (exact) **or**
  `approxTopK(tolerate_ratio=…)(score, o, ctx)` (faster, see §10) → done.
- `Reduce(dim=0)` is your escape hatch for per-sequence summaries.
- Everything else is the framework's problem. Trust it.
