# Cache op reference — what each op computes

This file documents the mathematical semantics of every op in
`vortex_torch.cache`, written from the user's view. The companion doc
for the indexer side is [`indexer_op.md`](indexer_op.md); if the
overall model is new to you, start with
[`program_forward_cache.md`](program_forward_cache.md) first.

Mental-model recap:

- **One block at a time.** Every tensor in `forward_cache` is
  `[1, D_0, D_1]` — the tile for the block that just completed.
- **`cache["k"]` and `cache["v"]`** are `[1, block_size, D]`, always
  available. **Do not declare them** in `create_cache`.
- **Every other field** is `[1, D_0, D_1]` where `(D_0, D_1)` is what
  you returned from `create_cache`.
- **Reductions only collapse the inner axes (`dim ∈ {1, 2}`).** There
  is no `dim=0` on the cache side — each program owns exactly one
  block, so there's nothing to reduce across.

All ops below are **construction-time values** you declare once in
`__init__` and call as `self.op(..., loc=loc, ctx=ctx)` inside
`forward_cache`. Unless stated otherwise, each call site needs its own
instance — don't share.

Every op takes `loc` and `ctx` kwargs; just thread them through from
the `forward_cache(self, cache, loc, ctx)` signature.

## Call signatures

| family | signature |
|---|---|
| unary elementwise, reductions, `MaskSlice` | `op(x, output, loc=loc, ctx=ctx)` |
| binary elementwise, `GeMM` | `op(x, y, output, loc=loc, ctx=ctx)` |
| `Fill` | `op(dest, loc=loc, ctx=ctx)` (writes `dest` directly) |

The `output` argument is the cache field to write into. Pass `None` to
get a fresh intermediate tile you can feed into the next op; the final
op in a chain must target a cache field declared in `create_cache`.

---

## 1. Reductions

All cache reductions collapse one inner axis and keep its size as `1`
(keepdim). `dim ∈ {1, 2}` only.

### `Mean(dim=k)` / `Max(dim=k)` / `Min(dim=k)` / `L2Norm(dim=k)`

For `dim=1` (reduce over the `D_0` axis):

$$
\text{Mean}(X)[0, 0, d_1]    = \tfrac{1}{D_0} \sum_{d_0} X[0, d_0, d_1], \quad
\text{Max}(X)[0, 0, d_1]     = \max_{d_0} X[0, d_0, d_1]
$$
$$
\text{Min}(X)[0, 0, d_1]     = \min_{d_0} X[0, d_0, d_1], \quad
\text{L2Norm}(X)[0, 0, d_1]  = \sqrt{\sum_{d_0} X[0, d_0, d_1]^2}
$$

For `dim=2`: symmetric.

| input shape | `dim` | output shape |
|---|---|---|
| `[1, D_0, D_1]` | 1 | `[1, 1,   D_1]` |
| `[1, D_0, D_1]` | 2 | `[1, D_0, 1]` |

Most common use: **collapse `cache["k"]`'s block-size axis** —
`CMean(dim=1)(cache["k"], cache["centroids"])` takes `[1, block_size, D]`
and writes `[1, 1, D]` per-block summaries.

---

## 2. Matrix multiply — `GeMM()`

Per-block matrix multiply with a transpose on the right operand:

$$
\text{GeMM}(X, Y)[0, n_y, n_x] = \sum_{k=0}^{K-1} X[0, n_x, k] \cdot Y[0, n_y, k]
$$

Both operands live on the same single block.

| input | shape |
|---|---|
| `x` | `[1, N_x, K]` |
| `y` | `[1, N_y, K]` |
| output | `[1, N_y, N_x]` |

Useful for **learned linear projections** applied per block — e.g.
project the block's centroid through a fixed weight matrix, or compute
per-block feature statistics like `k @ k.T` for similarity-based
summaries.

---

## 3. Binary elementwise

Broadcast NumPy-style. All accept two `[1, D_0, D_1]`-shaped inputs
(any dim of size 1 broadcasts) and produce the broadcasted shape.

### `Multiply()`

$$
\text{Multiply}(X, Y) = X \odot Y
$$

Pointwise product.

### `Add(alpha=1.0, beta=1.0)`

$$
\text{Add}_{\alpha, \beta}(X, Y) = \alpha \cdot X + \beta \cdot Y
$$

Affine combination. `Add(α=1, β=1)` is plain `X + Y`;
`Add(α=1, β=-1)` is `X - Y`; `Add(α=0.9, β=1)` is an EMA-style blend.

### `Maximum()` / `Minimum()`

$$
\text{Maximum}(X, Y) = \max(X, Y), \quad \text{Minimum}(X, Y) = \min(X, Y)
$$

Elementwise.

### `WhereEqual()` / `WhereNotEqual()` / `WhereGreater()` / `WhereGreaterEqual()` / `WhereLess()` / `WhereLessEqual()`

Comparison → additive mask. Emit `0` where the predicate holds and
`-∞` (≈ `-1e30`) otherwise:

$$
\text{WhereGreater}(X, Y) =
\begin{cases}
0 & \text{if } X > Y \\
-\infty & \text{otherwise}
\end{cases}
$$

Used for building additive masks you later fold into a score with
`Add`.

---

## 4. Unary elementwise

All unary ops accept `alpha` and `beta` as construction-time scalars.
Each applies to every element of the input and preserves shape.

### `Relu(alpha=0.0, beta=0.0)`

$$
\text{Relu}_{\alpha, \beta}(x) =
\begin{cases}
x & \text{if } x \ge \alpha \\
\beta & \text{otherwise}
\end{cases}
$$

### `Sigmoid(alpha=0.0, beta=0.0)`

$$
\sigma_{\alpha, \beta}(x) = \frac{1}{1 + \exp(\beta x + \alpha)}
$$

### `Silu(alpha=0.0, beta=0.0)`

$$
\text{SiLU}_{\alpha, \beta}(x) = \frac{x}{1 + \exp(\beta x + \alpha)}
$$

### `Add_Mul(alpha=0.0, beta=1.0)`

$$
\text{Add\_Mul}_{\alpha, \beta}(x) = \beta \cdot x + \alpha
$$

Pure affine transform.

### `Abs(alpha=0.0, beta=1.0)`

$$
\text{Abs}_{\alpha, \beta}(x) = |\beta x + \alpha|
$$

### `Log(alpha=0.0, beta=1.0)`

$$
\text{Log}_{\alpha, \beta}(x) = \log(\beta x + \alpha)
$$

### `Exp(alpha=0.0, beta=1.0)`

$$
\text{Exp}_{\alpha, \beta}(x) = \exp(\beta x + \alpha)
$$

Standard forms (per the indexer-side doc):

| identity | parameters |
|---|---|
| `ReLU(x) = max(x, 0)` | `Relu(α=0, β=0)` |
| `σ(x) = 1/(1+e^{-x})` | `Sigmoid(α=0, β=-1)` |
| `SiLU(x) = x·σ(x)` | `Silu(α=0, β=-1)` |
| `y = x`  | `Add_Mul(α=0, β=1)` |
| `y = |x|` | `Abs(α=0, β=1)` |
| `y = log x` | `Log(α=0, β=1)` |
| `y = exp x` | `Exp(α=0, β=1)` |

---

## 5. Constant fill — `Fill(alpha=0.0)`

Overwrite a cache field with a scalar:

$$
\text{Fill}_{\alpha}(F)[0, d_0, d_1] = \alpha
$$

| input | shape |
|---|---|
| `dest` | a cache field `[1, D_0, D_1]` |

Only one positional argument at call time — the destination field.
Typical uses:

- **Zero-initialise a block**. Essential when `forward_indexer` will
  later accumulate into the field via `Load`/`Save`: if you don't
  `Fill(0.0)` the slot at block completion, the first `Load` on that
  block reads stale memory. See the running-average example in
  [`program_forward_cache.md §6b`](program_forward_cache.md).
- **Install a bias or offset** that stays constant for a block's
  lifetime.

---

## 6. Position-based mask — `MaskSlice(start, end, dim, alpha, beta)`

Writes `alpha` inside the slice `[start, end)` along `dim` and `beta`
outside, ignoring the input values entirely. Shape-preserving.

$$
\text{MaskSlice}(X)[0, \ldots, i, \ldots] =
\begin{cases}
\alpha & \text{if } \text{start} \le i < \text{end} \\
\beta  & \text{otherwise}
\end{cases}
$$

Constructor parameters:

- `start`, `end` — half-open range along `dim`.
- `dim` — `1` or `2`.
- `alpha` — value inside the range.
- `beta` — value outside the range.

The output depends only on *position*, not on the input's values — so
`MaskSlice` is really a constant-tile builder with a positional shape.
Pair it with `Multiply` or `Add` to down-weight or bias a region of an
existing tensor.

Examples:

- **Ignore the first few tokens of a block** when building a centroid:
  ```python
  mask    = self.mask(cache["k"], ..., loc=loc, ctx=ctx)   # [1, block_size, D], 0 on [0, 4), 1 elsewhere
  masked  = self.mul(cache["k"], mask, ..., loc=loc, ctx=ctx)
  self.mean(masked, cache["centroids"], loc=loc, ctx=ctx)
  ```
- **Additive `-∞` mask** on a feature range followed by `Add`.

---

## 6b. Using `cache["v"]` — value-based summaries

Every op that works on `cache["k"]` works identically on `cache["v"]` —
both are `[1, block_size, D]` tiles of the just-completed block. Most
sparse-attention flows route on **keys** because that's what the
attention dot product cares about, but **value** statistics are
sometimes the better signal: pages whose values are small or degenerate
won't contribute much to the attention-weighted sum regardless of
their key similarity.

### Per-feature value norm (the most common v-summary)

Maintain, per block, the L2 norm of each feature column across the
`block_size` tokens. Intuition: "how much signal does this block carry
in each feature?"

```python
def create_cache(self, block_size, head_dim):
    return {
        "centroids": (1, head_dim),   # as usual, from cache["k"]
        "v_norm":    (1, head_dim),   # new: per-feature block norm of cache["v"]
    }

def __init__(self):
    super().__init__()
    self.k_mean = CMean(dim=1)       # [1, block_size, D] → [1, 1, D]
    self.v_norm = CL2Norm(dim=1)     # [1, block_size, D] → [1, 1, D]

def forward_cache(self, cache, loc, ctx):
    self.k_mean(cache["k"], cache["centroids"], loc=loc, ctx=ctx)
    self.v_norm(cache["v"], cache["v_norm"],    loc=loc, ctx=ctx)
```

Each cache block now carries **two** summaries: the key centroid and
the value norm. Both ops read their respective `[1, block_size, D]`
input and write a `[1, 1, D]` output. The framework fuses them into
one per-block kernel.

Over on the indexer side you'd combine them, e.g.:

```python
# forward_indexer
raw    = self.mul(q_mean, cache["centroids"], ctx=ctx)          # [S, 1, D] cosine-style
scaled = self.mul_vn(raw,     cache["v_norm"],    ctx=ctx)      # [S, 1, D] bias by value norm
score  = self.sum(scaled, ctx=ctx)                              # [S, 1, 1]
```

Pages with both a well-aligned centroid *and* a strong value norm
score highest.

### Per-row value norm (norm per token within the block)

If you instead want "how much signal does each token in the block
carry", reduce on the feature axis:

```python
self.v_token_norm = CL2Norm(dim=2)                              # [1, block_size, D] → [1, block_size, 1]
```

And declare:

```python
return {
    "v_token_norm": (block_size, 1),   # one scalar per token slot
}
```

This gives the indexer a per-token importance vector per block.

### General rule

- `cache["v"]`, like `cache["k"]`, is always present and you don't
  declare it in `create_cache`.
- Any op in this reference that reads `cache["k"]` accepts `cache["v"]`
  the same way.
- Declare one user field per summary you want to maintain; `forward_cache`
  can fire as many cache ops in one pass as you need — the framework
  fuses them into a single per-block kernel.

---

## 7. Quick-reference cheat sheet

| op | shape transformation | math |
|---|---|---|
| `Mean/Max/Min/L2Norm(dim=k)(x, output)` | collapses axis `k`, keepdim | reduction |
| `GeMM()(x, y, output)` | `[1, N_x, K], [1, N_y, K] → [1, N_y, N_x]` | `Y @ Xᵀ` |
| `Multiply()(x, y, output)` | broadcast | `x * y` |
| `Add(α, β)(x, y, output)` | broadcast | `α·x + β·y` |
| `Maximum/Minimum()(x, y, output)` | broadcast | elementwise |
| `Where*()(x, y, output)` | broadcast | 0 if predicate, else `-∞` |
| `Relu/Sigmoid/Silu/... (α, β)(x, output)` | same shape | (see §4) |
| `Fill(α)(dest)` | writes `dest` | all entries ← `α` |
| `MaskSlice(start, end, dim, α, β)(x, output)` | same shape | α inside range, β outside |

(`loc=loc, ctx=ctx` omitted everywhere — thread them through
verbatim.)

---

## 8. Common idioms

| you want | how to do it |
|---|---|
| "centroid per block" | `CMean(dim=1)(cache["k"], cache["centroids"])` |
| "QUEST envelope" | `CMax(dim=1)(cache["k"], cache["max"])` + `CMin(dim=1)(cache["k"], cache["min"])` |
| "per-block L2 norm of keys" | `CL2Norm(dim=1)(cache["k"], cache["norm"])` |
| "signed-magnitude centroid" (mean of \|k\|) | `CAbs` → `CMean(dim=1)` |
| "projected centroid" | `CMean(dim=1)` → `CGeMM` with a fixed learned weight |
| "ignore first few tokens of a block" | `MaskSlice(start, end, dim=1, α=0, β=1)` → `Multiply` → `CMean` |
| "fill a field with a constant" | `CFill(α=value)(cache["bias"])` |
| "zero-init a persistent scalar before indexer accumulates into it" | `CFill(α=0.0)(cache["running_*"])` |
| "additive mask over a feature range" | `MaskSlice(dim=2, α=-1e30, β=0)` → `Add(α=1, β=1)` |
| "use both k and v" | two independent op chains writing to two different fields (see §6b) |
| "per-feature block norm of values" | `CL2Norm(dim=1)(cache["v"], cache["v_norm"])` — `[1, block_size, D] → [1, 1, D]` |
| "per-token norm within a block" | `CL2Norm(dim=2)(cache["v"], cache["v_token_norm"])` — `[1, block_size, D] → [1, block_size, 1]` |

---

## 9. What's *not* on the cache side

Easy reference for what you won't find here (and why):

| missing op | reason |
|---|---|
| `Sum` | Not wired on the cache side; use `Mean` (divide by the known axis size if you need a true sum). |
| `Reduce(dim=0)` | Each cache kernel owns one block; there is no "across blocks" in a single pass. Do cross-block aggregation on the indexer side. |
| `Softmax` / `Normalize` / `Conv1d` | These are scan-style ops; the cache side only runs fused per-block kernels. Use the indexer side. |
| `Transpose` | Not exposed on the cache side. If you need the swapped view, compute the op that would consume it differently (e.g. a different reduction axis). |
| `Save` / `Load` | These are indexer-side mechanisms for persisting state across decode steps. On the cache side, "write to a cache field" *is* the persistence — just pass the field as `output`. |
| `topK` | The cache side is write-only for summaries; routing decisions happen in `forward_indexer`. |

---

## 10. Final sanity checks

Before you commit a `forward_cache`, walk through this list:

1. Every op I call has an `output=` pointing at a cache field I
   declared in `create_cache` — *or* it's an intermediate step
   (`output=None`) feeding directly into the next op.
2. I didn't declare `"k"` or `"v"` in `create_cache`.
3. I'm only reducing on `dim ∈ {1, 2}`.
4. I pass `loc` and `ctx` to every op.
5. Each op instance is used at exactly one call site (or I made as
   many copies as I have call sites).
6. If the indexer accumulates into a field via `Load`/`Save`, I zero
   that field here with `CFill(0.0)`.
