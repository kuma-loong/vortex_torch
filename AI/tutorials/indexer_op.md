# Indexer op reference — what each op computes

This file documents the mathematical semantics of every op in
`vortex_torch.indexer`, written from the user's view. If you've read
[`program_forward_indexer.md`](program_forward_indexer.md), you already
have the mental model:

- **`batch_size = 1` always.** Pretend the system has exactly one
  sequence in flight. The query leading axis is always 1, and the cache
  leading axis `S` is the number of pages of that single sequence.
- `q` has shape `[1, H_q, D]`.
- Every `cache["<name>"]` has shape `[S, D_0, D_1]`.
- You compose ops to produce a score `[S, 1, 1]` and call `topK`.

That batch-size-1 view is the *entire* surface. The framework replays
your one-sequence program across the real batch and kv-head axes; you
never write the multiplexing.

Every op below is a **construction-time value** you declare once in
`__init__`, then call as `self.op(args..., ctx=ctx)` inside
`forward_indexer`. Unless stated otherwise, each call site needs its own
instance — don't share.

Shapes in the tables are logical, not physical. Broadcasting follows the
NumPy rule: a dim of size 1 broadcasts against any dim of the same
position, including the leading `S` axis (so a `[1, …]` q broadcasts
across all `S` pages of the cache).

---

## 1. Matrix multiply — `GeMM()`

Matrix multiply per page with a transpose on the right operand:

$$
\text{GeMM}(X, Y)[s, n_y, n_x] = \sum_{k=0}^{K-1} X[s, n_x, k] \cdot Y[s, n_y, k]
$$

`X`'s leading axis can be either `1` (broadcast — e.g. `X` comes from
`q` or a reduction of `q`) or `S` (per-page — e.g. `X` comes from a
previous `GeMM` / `Multiply` output). When it's `1`, treat the formula
as standard NumPy broadcasting: `X[0]` is used for every page `s`.
When it's `S`, `X[s]` is used for page `s`. Either way the output
picks up the full `S` axis from `Y`.

| input | shape |
|---|---|
| `x` | `[1, N_x, K]` (broadcast across pages) *or* `[S, N_x, K]` (per-page) |
| `y` | `[S, N_y, K]` |
| output | `[S, N_y, N_x]` |

Common uses:

- **Dot product**: `GeMM(q_mean, cache["centroids"])` with
  `q_mean = [1, 1, D]` and `centroids = [S, 1, D]` → `[S, 1, 1]`.
- **Multi-head cosine**: `GeMM(q, cache["centroids"])` with
  `q = [1, H_q, D]` and `centroids = [S, 1, D]` → `[S, 1, H_q]`.
- **Top-k of multiple centroids per page**:
  `GeMM(q_mean, centroids_C)` with `q_mean = [1, 1, D]` and
  `centroids_C = [S, C, D]` → `[S, C, 1]` (one score per page per
  centroid).

---

## 2. Reductions

All reductions have a `dim` parameter set at construction time. They
preserve the rank of the tensor (the reduced axis becomes size 1).

### `Sum(dim=k)` / `Mean(dim=k)` / `Max(dim=k)` / `Min(dim=k)` / `L2Norm(dim=k)`

For `dim=1` (reduce over the `D_0` axis):

$$
\text{Sum}(X)[s, 0, d_1]     = \sum_{d_0} X[s, d_0, d_1], \quad
\text{Mean}(X)[s, 0, d_1]    = \tfrac{1}{D_0} \sum_{d_0} X[s, d_0, d_1]
$$
$$
\text{Max}(X)[s, 0, d_1]     = \max_{d_0} X[s, d_0, d_1], \quad
\text{Min}(X)[s, 0, d_1]     = \min_{d_0} X[s, d_0, d_1]
$$
$$
\text{L2Norm}(X)[s, 0, d_1]  = \sqrt{\sum_{d_0} X[s, d_0, d_1]^2}
$$

For `dim=2` (reduce over `D_1`): symmetric, the collapsed axis becomes
index 0 on dim 2.

| input | shape | output shape |
|---|---|---|
| `dim=1` | `[S, D_0, D_1]` | `[S, 1,   D_1]` |
| `dim=2` | `[S, D_0, D_1]` | `[S, D_0, 1]` |

`Sum` accepts `dim in {0, 1, 2}` too — `dim=0` collapses the leading
`S` axis (see next).

### `Reduce(dim=0)` — cross-page / per-sequence summary

Special case of the above with `dim=0`: collapses the entire page axis
into a single row.

$$
\text{Reduce}(X)[0, d_0, d_1] = \mathcal{R}_s\, X[s, d_0, d_1]
$$

| input | shape | output shape |
|---|---|---|
| `dim=0` | `[S, D_0, D_1]` | `[1, D_0, D_1]` |

The output broadcasts back against the `S` axis in any subsequent
elementwise binary op. Typical use: **per-page gating against a
sequence-level statistic**:

```python
score    = ...                                       # [S, 1, 1]
mean_all = self.mean_all(score, ctx=ctx)             # [1, 1, 1]  (Mean(dim=0))
gate     = self.cmp(score, mean_all, ctx=ctx)        # WhereGreater → 0 / -inf [S, 1, 1]
gated    = self.add(score, gate, ctx=ctx)            # Add(α=1, β=1) → masks below-mean to -inf
self.topk(gated, o, ctx=ctx)
```

> **Caveat — `topK` is invariant to monotonic transforms.** Piping
> `Reduce(dim=0)(score)` directly into `Add(α=1, β=-1)` and then
> `topK` is a *no-op*: subtracting one scalar from every page is
> order-preserving, and `topK` selects by order. To make
> per-sequence statistics actually change the picked set, combine
> them with a *non-monotonic* op (the `WhereGreater`-then-`Add`
> threshold above), or use them as a per-page **multiplier**
> against a different signal (e.g. multiply the score by
> `score / mean_all` to amplify above-mean pages and damp below-
> mean ones).

---

## 3. Binary elementwise

All binary ops broadcast NumPy-style. In the batch-size-1 view you'll
almost always combine a `[1, …]` tensor (from `q` or a cross-sequence
summary) with a `[S, …]` tensor (from cache or a per-page intermediate)
— the leading axis broadcasts against `S`, giving a `[S, …]` result.

### `Multiply()`

$$
\text{Multiply}(X, Y) = X \odot Y
$$

Pointwise product. The workhorse for "score × something".

### `Add(alpha=1.0, beta=1.0)`

$$
\text{Add}_{\alpha, \beta}(X, Y) = \alpha \cdot X + \beta \cdot Y
$$

Weighted sum. Not just addition — any affine mix.

- `Add(α=1, β=1)`: plain `X + Y`.
- `Add(α=1, β=-1)`: `X - Y` (subtract).
- `Add(α=0.9, β=1)`: `0.9·X + Y` (momentum / EMA-style).

### `Maximum()` / `Minimum()`

$$
\text{Maximum}(X, Y) = \max(X, Y), \qquad
\text{Minimum}(X, Y) = \min(X, Y)
$$

Elementwise. Used heavily in QUEST-style envelope bounds.

### `Kron(dim=(1, 2))`

Per-page Kronecker product over a configurable subset of the inner
axes. Axes listed in `dim` get the Kron expansion (output size = product
of input sizes); the remaining inner axis follows ordinary broadcast
(equal sizes or one of them is `1`). The leading `S` axis is always
elementwise (with `BATCHED` partners broadcasting).

For `dim=(1, 2)` (full Kron):

$$
\text{Kron}(X, Y)[s,\, i \cdot y_1 + j,\, k \cdot y_2 + l]
  = X[s, i, k] \cdot Y[s, j, l]
$$

For `dim=1` (row-Kron; the `D_1` axis is elementwise, sizes must agree
or be 1):

$$
\text{Kron}_{\text{dim}=1}(X, Y)[s,\, i \cdot y_1 + j,\, d]
  = X[s, i, d] \cdot Y[s, j, d]
$$

For `dim=2` (col-Kron; the `D_0` axis is elementwise, sizes must agree
or be 1):

$$
\text{Kron}_{\text{dim}=2}(X, Y)[s,\, c,\, k \cdot y_2 + l]
  = X[s, c, k] \cdot Y[s, c, l]
$$

| `dim` | `x` shape | `y` shape | output shape |
|---|---|---|---|
| `(1, 2)` | `[S, x_1, x_2]` | `[S, y_1, y_2]` | `[S, x_1·y_1, x_2·y_2]` |
| `1`     | `[S, x_1, D]`   | `[S, y_1, D]`   | `[S, x_1·y_1, D]`       |
| `2`     | `[S, C, x_2]`   | `[S, C, y_2]`   | `[S, C, x_2·y_2]`       |

`dim` accepts an int or any iterable of ints in `{1, 2}`. Each
contributing axis size should be a power of 2 (Triton tile constraint).
Format dispatch: any `RAGGED` or `PAGED` participant yields a `RAGGED`
output; `BATCHED ⊗ BATCHED` yields a `BATCHED` output (one tile per
`(batch, head)`); `BATCHED` inputs broadcast along the workload axis so
they pair cleanly with `RAGGED`/`PAGED` partners.

Typical uses:

- **Multi-head × multi-centroid score grid**: Kron a per-head q-summary
  `[S, H, 1]` with a per-page multi-centroid field `[S, 1, C]` to get
  `[S, H, C]` scores, then fold the `H` axis with `Max(dim=1)` /
  `Mean(dim=1)` before `topK`.
- **Cartesian feature combinations**: combine two per-page feature
  vectors `[S, 1, A]` and `[S, 1, B]` into `[S, A, B]` cross-features,
  then reduce to a scalar score.

### `WhereEqual()` / `WhereNotEqual()` / `WhereGreater()` / `WhereGreaterEqual()` / `WhereLess()` / `WhereLessEqual()`

Comparison → additive mask. Emit `0` where the predicate holds,
`-∞` (≈ `-1e30`) otherwise:

$$
\text{WhereGreater}(X, Y) =
\begin{cases}
0      & \text{if } X > Y \\
-\infty & \text{otherwise}
\end{cases}
$$

Typical use: **mask a score before softmax**. `score + WhereGreater(x, y)`
zeros out untouched positions and drives the rest to `-∞`, which
`Softmax` then collapses to zero probability.

---

## 4. Unary elementwise

All unary ops accept `alpha` and `beta` as construction-time scalars.
They apply to every element of the input and preserve shape.

### `Relu(alpha=0.0, beta=0.0)`

$$
\text{Relu}_{\alpha, \beta}(x) =
\begin{cases}
x     & \text{if } x \ge \alpha \\
\beta & \text{otherwise}
\end{cases}
$$

Standard `ReLU` is `Relu(α=0, β=0)`. A thresholded clamp with fallback
value is `Relu(α=τ, β=τ)`.

### `Sigmoid(alpha=0.0, beta=0.0)`

$$
\sigma_{\alpha, \beta}(x) = \frac{1}{1 + \exp(\beta x + \alpha)}
$$

Standard logistic is `Sigmoid(α=0, β=-1)`.

### `Silu(alpha=0.0, beta=0.0)`

$$
\text{SiLU}_{\alpha, \beta}(x) = \frac{x}{1 + \exp(\beta x + \alpha)}
$$

Standard SiLU / Swish is `Silu(α=0, β=-1)`.

### `Add_Mul(alpha=0.0, beta=1.0)`

$$
\text{Add\_Mul}_{\alpha, \beta}(x) = \beta \cdot x + \alpha
$$

Pure affine transform. Equivalent to a scalar multiply + scalar add.

### `Abs(alpha=0.0, beta=1.0)`

$$
\text{Abs}_{\alpha, \beta}(x) = |\beta x + \alpha|
$$

Absolute value of an affine argument. Standard `|x|` is
`Abs(α=0, β=1)`.

### `Log(alpha=0.0, beta=1.0)`

$$
\text{Log}_{\alpha, \beta}(x) = \log(\beta x + \alpha)
$$

Defined for `β·x + α > 0`. Standard `log(x)` is `Log(α=0, β=1)`.

### `Exp(alpha=0.0, beta=1.0)`

$$
\text{Exp}_{\alpha, \beta}(x) = \exp(\beta x + \alpha)
$$

Standard `exp(x)` is `Exp(α=0, β=1)`.

---

## 5. Scan / normalization

### `Softmax(dim=0, scale=1.0)`

Numerically stable softmax over the chosen axis, with an optional
pre-scale:

$$
\text{Softmax}(X)[s]
  = \frac{\exp(\text{scale} \cdot X[s])}{\sum_{s'} \exp(\text{scale} \cdot X[s'])}
$$

For `dim=0`: one probability distribution per `(D_0, D_1)` slot,
normalised across the `S` axis. The canonical use is **softmax over
pages**.

| input | shape | output shape |
|---|---|---|
| any `[S, D_0, D_1]` | — | `[S, D_0, D_1]` (same) |

### `Normalize(dim=0)`

Sum-to-one normalisation over the chosen axis:

$$
\text{Normalize}(X)[s] = \frac{X[s]}{\sum_{s'} X[s']}
$$

Like `Softmax` but without the `exp`; useful when the input is already
a non-negative weight distribution.

### `Conv1d(weight, dim=0, dtype=bfloat16)`

1-D convolution along the chosen axis with a user-provided kernel:

$$
\text{Conv1d}(X)[s, d_0, d_1]
  = \sum_{k=0}^{K-1} W[k, d_0, d_1] \cdot X[s + k - K/2, d_0, d_1]
$$

`weight` is a Python nested list of shape `[K, D_0, D_1]` that gets
materialised into a `torch.Tensor` at construction time. Use for
**smoothing scores along the page axis** or **learned stencil
operators**.

---

## 6. Layout — `Transpose()`

Swap the last two axes:

$$
\text{Transpose}(X)[s, d_1, d_0] = X[s, d_0, d_1]
$$

| input | shape | output shape |
|---|---|---|
| `X` | `[S, D_0, D_1]` | `[S, D_1, D_0]` |

Useful when you want to apply an op along the other inner axis without
rewriting the op. Not a pure-metadata "view" — it emits a real
reshuffle.

---

## 7. Position-based mask — `MaskSlice(start, end, dim, alpha, beta)`

Writes `alpha` inside the slice `[start, end)` along `dim` and `beta`
outside, ignoring the input values entirely. Shape is preserved.

$$
\text{MaskSlice}(X)[\ldots, i, \ldots] =
\begin{cases}
\alpha & \text{if } \text{start} \le i < \text{end} \\
\beta  & \text{otherwise}
\end{cases}
$$

Constructor parameters:

- `start`, `end` — half-open range along `dim`.
- `dim` — `1` or `2` (the `D_0` or `D_1` axis).
- `alpha` — value inside the range.
- `beta` — value outside the range.

The output depends only on *position*, not on the input's values, so
`MaskSlice` is really a tensor builder disguised as a unary op. Pair it
with `Multiply` or `Add` to selectively down-weight or bias a region of
an existing tensor.

Examples:

- **Zero out the first 8 feature planes**:
  `MaskSlice(start=0, end=8, dim=2, alpha=0.0, beta=1.0)` followed by
  `Multiply` with the envelope score.
- **Additive `-∞` mask** for positions after some cutoff:
  `MaskSlice(start=cutoff, end=BIG, dim=1, alpha=-1e30, beta=0.0)`
  followed by `Add(α=1, β=1)` with the score.

---

## 8. Persistent memory — `Save()` / `Load()`

`Save` and `Load` give `forward_indexer` state that survives across
decode steps. See §9 of `program_forward_indexer.md` for the end-to-end
pattern and the running-average example. Mathematically:

### `Load()`

$$
\text{Load}(F)[s, d_0, d_1] = F[s, d_0, d_1]
$$

Returns a fresh tensor with the same values as the cache field `F`.
The value `F[s]` is whatever was stored by the *previous* decode step's
`Save`, or the field's initialisation (zero by default) on step 0.

### `Save()`

$$
F[s, d_0, d_1] \leftarrow X[s, d_0, d_1]
$$

Writes `X` into the cache field `F` in place. The next decode step's
`Load(F)` will see this value.

| call | input | cache field |
|---|---|---|
| `Load()(F, ctx)` | `F: [S, D_0, D_1]` declared in `create_cache` | read |
| `Save()(X, F, ctx)` | `X: [S, D_0, D_1]`, same shape as `F` | written |

One rule: both `F`s must be cache fields declared in `create_cache`.
You can't create free-floating persistent buffers.

---

## 9. Final output — `topK()`

$$
(\text{topK}(S, k))_r = \operatorname*{argsort}_{s \in \text{pages}} S[s, 0, 0] \big|_{r\text{-th largest}}
$$

Select the top-`k` page indices by descending score and write them into
the output buffer `o`. `k` is the runtime constant
`ctx.topk_val` (configured via the engine JSON as
`vortex_topk_val` / `vortex_topk_ratio`); the op doesn't take a `k`
argument.

| input | shape |
|---|---|
| `score` | `[S, 1, 1]` — **strict requirement** |
| `o` | framework-provided buffer (the sparse `kv_indices` slot) |

Constraints:

- Must be called **exactly once** per `forward_indexer`.
- The score argument must be RAGGED `[S, 1, 1]` (the framework enforces
  this via the dispatch table). If your pipeline produces `[S, H_q, D]`
  or similar, fold the extra axes with `Max` / `Mean` / `Sum` first.

The framework also honours `vortex_block_reserved_bos` and
`vortex_block_reserved_eos` — the first `bos` and last `eos` pages of
every sequence are *always* selected regardless of score, so you don't
need to add a positional bias for them.

### `approxTopK(tolerate_ratio=0.0)` — approximate variant

Drop-in replacement for `topK` that swaps the exact-selection kernel
for an **adaptive 8-bit radix top-k** with a quality / cost knob.

```python
self.output_func = approxTopK(tolerate_ratio=0.25)
...
self.output_func(score, o, ctx=ctx)   # same call shape as topK
```

The kernel runs up to four 8-bit refinement rounds (32 bits total)
on fp32-promoted scores. After each round, the threshold bin is
found and `topk_remaining` is the number of slots still owed by
that bin. The kernel **stops early** as soon as

$$
\text{topk\_remaining} \;\le\; \text{tolerate\_ratio} \cdot \text{target\_k}
$$

filling the remaining slots from the current candidate set in
arrival order. The trade-off:

- `tolerate_ratio = 0.0` → all four rounds run; result is the exact
  top-k (only the output ordering is unsorted vs. classic `topK`).
- `tolerate_ratio = 1.0` → kernel stops after round 0, cheapest
  setting; selection is coarse.
- `0 < tolerate_ratio < 1` → adaptive: cheap when scores are
  well-separated, refines when they're tightly bunched.

**Same contract as `topK`** — RAGGED `[S, 1, 1]` input, BOS/EOS
preserved, exactly one call per `forward_indexer`. Only the
*selection quality* is traded for speed; the output **slot count**
is identical.

**Practical guidance:**

- When score distributions are heavy-tailed and `topK` cost
  dominates the indexer path, this is a clean throughput win.
- For tight score distributions where many candidates cluster near
  the threshold, keep `tolerate_ratio ≤ 0.05`.
- For distributions with a clear gap between selected and dropped
  pages, push `tolerate_ratio` up to `0.2 - 0.5`.
- Output indices are **unsorted within each segment** (this matches
  the underlying `topk_output_v2` C kernel). Downstream consumers
  must not assume sorted order.

---

## 10. Quick-reference cheat sheet

| op | shape transformation | math |
|---|---|---|
| `GeMM()(x, y)` | `[1 or S, N_x, K], [S, N_y, K] → [S, N_y, N_x]` | per-page `Y[s] @ Xᵀ` (X broadcast if leading dim 1) |
| `Sum/Mean/Max/Min/L2Norm(dim=k)(x)` | collapses axis `k`, keepdim | reduction |
| `Sum/Mean/... (dim=0)(x)` | `[S, D_0, D_1] → [1, D_0, D_1]` | cross-sequence summary |
| `Multiply()(x, y)` | broadcast | `x * y` |
| `Add(α, β)(x, y)` | broadcast | `α·x + β·y` |
| `Maximum/Minimum()(x, y)` | broadcast | elementwise |
| `Kron(dim)(x, y)` | Kron-expand `dim` axes, broadcast the rest | `x_1·y_1` × `x_2·y_2` (full) or one axis only |
| `Where*()(x, y)` | broadcast | 0 if predicate, else `-∞` |
| `Relu/Sigmoid/Silu/... (α, β)(x)` | same shape | (see §4) |
| `Softmax(dim, scale)(x)` | same shape | `exp(scale·x) / Σ exp(scale·x)` |
| `Normalize(dim)(x)` | same shape | `x / Σ x` |
| `Conv1d(weight, dim)(x)` | same shape | 1-D conv with user kernel |
| `Transpose()(x)` | `[S, D_0, D_1] → [S, D_1, D_0]` | swap inner axes |
| `MaskSlice(start, end, dim, α, β)(x)` | same shape | α inside range, β outside |
| `Load()(F)` | `[S, D_0, D_1]` | read cache field |
| `Save()(x, F)` | writes `F` | persist `x` into cache field |
| `topK()(score, o)` | `[S, 1, 1] →` writes `o` | pick top-k pages (exact, sorted) |
| `approxTopK(t)(score, o)` | `[S, 1, 1] →` writes `o` | pick top-k pages (adaptive radix; `t ∈ [0,1]` cheaper-but-looser at higher t; output unsorted) |

---

## 11. Common idioms

| scoring pattern | ops |
|---|---|
| dot product with centroid | `Multiply` → `Sum(dim=2)` (or one `GeMM`) |
| multi-head dot product | `GeMM(q, centroid)` → `Max(dim=2)` or `Mean(dim=2)` over `H_q` |
| QUEST envelope | `Multiply` ×2, `Maximum`, `Sum(dim=2)`, `Max(dim=1)` |
| masked features | `MaskSlice(dim=2, α=0, β=1)` → `Multiply` |
| gate by per-sequence threshold | score → `Mean(dim=0)` → `WhereGreater` → `Add(α=1, β=1)` (plain `Add(α=1, β=-1)` into `topK` is a **no-op**: `topK` is invariant to constant shifts) |
| softmax over pages | `Softmax(dim=0, scale)` |
| running average across steps | `Load` → `Add(α=momentum, β=1)` → `Save` |
| learned-kernel smoothing | `Conv1d(weight=[...], dim=0)` |
| top-k of multiple centroids | `GeMM` with `N_x=C` centroids → `Max(dim=2)` |
| multi-head × multi-centroid score grid | `Kron(dim=(1,2))` on `[S, H, 1]` × `[S, 1, C]` → `[S, H, C]` → `Max(dim=1)` |
| Cartesian cross-features per page | `Kron(dim=(1,2))` on `[S, 1, A]` × `[S, 1, B]` → `[S, A, B]` → reduce |
| row-Kron of two centroid sets sharing a feature dim | `Kron(dim=1)` on `[S, x_1, D]` × `[S, y_1, D]` → `[S, x_1·y_1, D]` |

When in doubt, walk the shapes in your head op by op; every op in this
doc is a pure, side-effect-free transform of the shapes in the table
above (with `Save` the lone exception — it writes a cache field).
