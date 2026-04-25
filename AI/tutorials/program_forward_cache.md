# Writing `forward_cache` — the single-block mental model

This is the cache-side companion to
[`program_forward_indexer.md`](program_forward_indexer.md). Same
framework, same `vFlow` class — but `forward_cache` has a much smaller
surface. Read it once; you'll rarely touch it again.

---

## 1. The mental model: one block, uniform `[1, D_0, D_1]` views

> Pretend the system is executing exactly **one** block at a time.
> Every cache field hands you a single `[1, D_0, D_1]` tile — the data
> associated with that one block.

Under this fiction:

| tensor | logical shape | what it is |
|---|---|---|
| `cache["k"]` | `[1, block_size, D]` | the fresh block of keys that just landed (always available — **do not declare**) |
| `cache["v"]` | `[1, block_size, D]` | the fresh block of values that just landed (always available — **do not declare**) |
| every user field `cache["<name>"]` | `[1, D_0, D_1]` | the slot you want to read/write for this block; `(D_0, D_1)` is what you declared in `create_cache` |
| `loc`, `ctx` | opaque | pass-through tokens — don't interpret |

> **`cache["k"]` and `cache["v"]` are always there.** The framework
> allocates them for you and populates them with the incoming key/value
> tokens before your `forward_cache` runs. You **must not** include them
> in your `create_cache` return value — the framework asserts their
> absence and injects them automatically with inner shape
> `(block_size, head_dim)`. Use them freely as inputs; never try to
> declare or allocate them.

That's the whole surface. No query `q`. No leading `S` axis. No batch.
Each op you call runs exactly once for this block, and the framework
replays your single-block program across the real cache pool.

The cache pipeline fires at **block completion**: the moment a freshly
written token is the last one in its block, this code runs to update
whatever derived summaries you maintain. On any other token write, the
framework short-circuits — your `forward_cache` body isn't executed at
all.

---

## 2. The goal

Read `cache["k"]` / `cache["v"]` (and any previously-written summary
fields), compute new per-block summaries, and **persist them back into
cache fields**. That's it — no return value, no `topK`, no sparse
routing. Cache ops are pure side effects: they update fields so the
indexer side can consume them next decode step.

Typical summaries:

- per-block centroid (mean over the `block_size` token axis)
- per-block max / min envelope of the keys
- per-block L2-norm of the keys
- a masked / sliced version of the keys (e.g. ignoring the first few
  token slots)
- learned per-block projections (`GeMM` against a fixed weight)

Everything you maintain here is what `forward_indexer` gets to *read*.
The two sides meet at the named fields declared in `create_cache`.

---

## 3. The calling convention

Every cache op is called like this:

```python
self.op(x, output, loc=loc, ctx=ctx)
```

- `x` — the input field (or an intermediate from a previous op).
- `output` — the cache field to write the result into. Must be a field
  declared in `create_cache`.
- `loc` and `ctx` — pass-through tokens from the `forward_cache`
  signature. Always thread them verbatim; don't interpret.

Example — compute the per-block mean of keys and store it under
`cache["centroids"]`:

```python
def forward_cache(self, cache, loc, ctx):
    self.reduction(cache["k"], cache["centroids"], loc=loc, ctx=ctx)
```

One op, one call, one side effect. Read `cache["k"]`, write
`cache["centroids"]`. Done.

---

## 4. How the ops behave on these logical shapes

Exactly like you'd expect from plain PyTorch / NumPy on rank-3 tensors
with a leading 1:

| op | signature | example |
|---|---|---|
| `Mean(dim=k)(x, output, loc, ctx)` | reduce over `dim=k`, keepdim | `Mean(dim=1)([1, 16, D])` → `[1, 1, D]` |
| `Max(dim=k)(x, output, loc, ctx)`  | reduce over `dim=k`, keepdim | `Max(dim=1)([1, 16, D])`  → `[1, 1, D]` |
| `Min(dim=k)(x, output, loc, ctx)`  | reduce over `dim=k`, keepdim | `Min(dim=1)([1, 16, D])`  → `[1, 1, D]` |
| `L2Norm(dim=k)(x, output, loc, ctx)` | `sqrt(sum(x·x, dim=k))`    | `L2Norm(dim=1)([1, 16, D])` → `[1, 1, D]` |
| `GeMM()(x, y, output, loc, ctx)` | per-block `Y @ Xᵀ`      | `[1, N_x, K], [1, N_y, K]` → `[1, N_y, N_x]` |
| `Multiply()(x, y, output, loc, ctx)` | elementwise product   | broadcasted |
| `Add(α, β)(x, y, output, loc, ctx)`  | `α·x + β·y`           | broadcasted |
| `Maximum()(x, y, output, loc, ctx)`  | elementwise max        | broadcasted |
| `Minimum()(x, y, output, loc, ctx)`  | elementwise min        | broadcasted |
| `Relu/Sigmoid/Silu/... (α, β)(x, output, loc, ctx)` | unary       | shape-preserving |
| `MaskSlice(start, end, dim, α, β)(x, output, loc, ctx)` | position-only mask | shape-preserving |
| `Fill(value)(output, loc, ctx)` | fill with a constant       | writes the destination |

Two rules that cover every situation:

1. **Every cache op has a leading `1` on every input and every output.**
   That's the single-block view. Standard NumPy broadcasting applies:
   a dim of size 1 broadcasts against any size on the same axis.
2. **Reductions only collapse the inner axes (`dim ∈ {1, 2}`).**
   There is no `dim=0` on the cache side — each program owns exactly one
   block, so there's nothing to reduce across. If you need cross-block
   aggregation, do it on the indexer side via `Reduce(dim=0)`.

---

## 5. Worked example: `block_sparse_attention`

Maintain one centroid (the mean of the block's keys) per block.

```python
def __init__(self):
    super().__init__()
    self.reduction = CMean(dim=1)   # reduce the block_size axis of cache["k"]

def create_cache(self, block_size, head_dim):
    return {
        "centroids": (1, head_dim),
    }

def forward_cache(self, cache, loc, ctx):
    self.reduction(cache["k"], cache["centroids"], loc=loc, ctx=ctx)
```

Reading line-by-line in the mental model:

- `cache["k"]` is `[1, block_size, head_dim]` — the freshly-completed
  block.
- `CMean(dim=1)` reduces over `block_size` → `[1, 1, head_dim]`.
- The result lands in `cache["centroids"]` at the same block slot.

One line of compute. That's the entire cache side.

---

## 6. Worked example: `gqa_quest_sparse_attention`

Maintain two per-block envelopes — an elementwise max and an elementwise
min over the block's keys.

```python
def __init__(self):
    super().__init__()
    self.reduction_max = CMax(dim=1)
    self.reduction_min = CMin(dim=1)

def create_cache(self, block_size, head_dim):
    return {
        "max": (1, head_dim),
        "min": (1, head_dim),
    }

def forward_cache(self, cache, loc, ctx):
    self.reduction_max(cache["k"], cache["max"], loc=loc, ctx=ctx)
    self.reduction_min(cache["k"], cache["min"], loc=loc, ctx=ctx)
```

Both reductions read the same `cache["k"]` tile `[1, block_size, D]`
and collapse the token axis:

- `CMax(dim=1)` → `[1, 1, D]` per-feature max, written to `cache["max"]`.
- `CMin(dim=1)` → `[1, 1, D]` per-feature min, written to `cache["min"]`.

Two ops. The framework fuses them into a single per-block kernel that
loads `cache["k"]` once and emits two stores.

---

## 6b. Worked example: `running_avg_block_sparse` — zero-initialising a block

This is the cache counterpart of the `Save` / `Load` example in
[`program_forward_indexer.md` §9](program_forward_indexer.md). Recap of
the full flow: the indexer maintains `cache["running_score"]` across
decode steps with `running ← α·last + current` (via `Load` / `Add` /
`Save`), and the cache maintains `cache["centroids"]` the usual way.

There's a subtle bug if `forward_cache` *only* updates `centroids`:
when a brand-new block becomes active, its `running_score` slot still
contains whatever bytes lived there previously (the cache pool is
reused across sequences — nothing zeroes it automatically). The first
`Load` on that block reads garbage.

The fix is to explicitly **zero-initialise `running_score` at block
completion**, using `CFill`:

```python
def __init__(self):
    super().__init__()
    self.reduction          = CMean(dim=1)
    self.init_running_score = CFill(alpha=0.0)

def create_cache(self, block_size, head_dim):
    return {
        "centroids":     (1, head_dim),  # maintained here
        "running_score": (1, 1),         # zeroed here, accumulated in forward_indexer
    }

def forward_cache(self, cache, loc, ctx):
    self.reduction(cache["k"], cache["centroids"], loc=loc, ctx=ctx)
    self.init_running_score(cache["running_score"], loc=loc, ctx=ctx)
```

Reading line-by-line:

- `CMean(k → centroids)` — the familiar per-block centroid update.
- `CFill(0.0)(running_score)` — overwrite this block's `running_score`
  tile with zeros. This runs exactly once per block (at the block's
  completion event), so it doesn't clobber `Save`'s writes that happen
  later: the ordering within a single decode step is (1) `forward_cache`
  fires for newly-completed blocks (zeroing + centroid), (2)
  `forward_indexer` runs for *all* currently-live blocks (loading the
  zero, fusing with `current`, saving the new value).

The general pattern: **any cache field that the indexer side accumulates
into (read-then-write with `Load`/`Save`) needs a one-shot
initialisation on the cache side**. `CFill` is the tool.

---

## 7. Chaining ops via intermediates

When a summary needs more than one step, chain cache ops by omitting
`output` (an intermediate tile is allocated) and then final-write to a
cache field at the end:

```python
def __init__(self):
    super().__init__()
    self.abs   = CAbs()               # |·|
    self.mean  = CMean(dim=1)         # reduce block_size axis

def forward_cache(self, cache, loc, ctx):
    absv       = self.abs(cache["k"], loc=loc, ctx=ctx)     # [1, block_size, D]
    self.mean(absv, cache["centroids"], loc=loc, ctx=ctx)   # [1, 1, D] → cache field
```

The intermediate `absv` lives entirely inside the fused per-block
kernel — no materialisation, no extra memory.

Rule of thumb: **the final op in the chain must target a cache field.**
An intermediate is only meaningful if something downstream in the same
`forward_cache` consumes it; the cache pipeline has no notion of
"persist this intermediate for the next step" (for that, see
`Save`/`Load` in the indexer tutorial).

---

## 8. What you DON'T write

The framework handles all of these:

- No batch loop, no head loop.
- No block-position logic, no `loc`-indexing — just thread `loc`
  through every call.
- No dense/sparse routing — cache runs uniformly on every block.
- No gating on "is this a decode step?" — the framework decides when
  `forward_cache` runs and doesn't call it when it shouldn't.
- No Triton / CUDA code, no kernel launches.
- No read-my-own-write patterns. Each cache op reads its inputs and
  writes its output; no "read the field I just wrote earlier in
  `forward_cache`" — model that in `forward_indexer` with `Load`/`Save`
  if you need it.

---

## 9. Common recipes

| you want | how to do it |
|---|---|
| "centroid per block" | `CMean(dim=1)(cache["k"], cache["centroids"])` |
| "QUEST envelope" (max and min per feature) | `CMax(dim=1)(cache["k"], cache["max"])` + `CMin(dim=1)(cache["k"], cache["min"])` |
| "L2 norm of the keys per block" | `CL2Norm(dim=1)(cache["k"], cache["norm"])` |
| "signed-magnitude centroid" (mean of `|k|`) | chain `CAbs` → `CMean(dim=1)` |
| "projected centroid" (learned matrix on the mean) | chain `CMean(dim=1)` → `CGeMM` with a fixed weight |
| "ignore the first few tokens of a block" | `MaskSlice(start, end, dim=1, α=0, β=1)` → `Multiply` → `CMean(dim=1)` |
| "fill a field with a constant" (e.g. initialise a bias) | `CFill(value=c)(cache["bias"])` |
| "use both k and v" | two op chains, one per source, writing to two different fields |

---

## 10. Pitfalls that trip people up

1. **Don't reuse an op instance.** Each call site needs its own
   instance. `self.mean_k = CMean(dim=1); self.mean_v = CMean(dim=1)`,
   not one `self.mean` used twice.
2. **No native torch ops.** Every tensor in `forward_cache` must go
   through `vortex_torch.cache.*`. `cache[...]` is a `vTensor` at
   profile time — `.view` / `.contiguous` / elementwise-torch won't
   compile.
3. **No `Reduce(dim=0)` on the cache side.** The cache runs per-block;
   "across all blocks" isn't a thing here. If you need it, do it in
   `forward_indexer`.
4. **The final op must target a declared field.** If all your cache
   ops have `output=None`, nothing gets persisted — the whole chain is
   a no-op from the indexer's perspective.
5. **Declare the fields you write to in `create_cache` — *except*
   `"k"` and `"v"`.** Every user-defined field (`"centroids"`, `"max"`,
   `"running_score"`, …) must appear in `create_cache`, otherwise the
   framework has nowhere to allocate the backing buffer. But `"k"` and
   `"v"` are the one pair you **must not** declare: the framework
   auto-injects them with inner shape `(block_size, head_dim)` and will
   hard-assert if `create_cache` returns them. They're always present
   in `cache` at runtime — just read them, don't try to create them.
6. **The indexer side reads only what the cache side writes.** If
   `forward_indexer` reads `cache["centroids"]` and `forward_cache`
   never writes it, you'll score against zeros.

---

## 11. Mental-model summary

- **One block at a time.** Every tensor is `[1, D_0, D_1]`.
- **Read `cache["k"]` / `cache["v"]`** (the fresh block) and any
  summary fields from previous steps. `k` and `v` are always there —
  do **not** include them in `create_cache`.
- **Compose ops with standard broadcasting** to get
  `[1, new_D_0, new_D_1]` tiles.
- **Write the final result into a cache field** declared in
  `create_cache`. That's the persist side.
- **Reductions go over `dim ∈ {1, 2}` only.** No cross-block work.
- **Thread `loc` and `ctx` through every call.** Don't interpret them.

Everything else is the framework's problem.
