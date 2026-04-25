# Writing `create_cache` — declaring your auxiliary cache fields

`create_cache` is the smallest method in a `vFlow` — usually 3-5 lines.
It's also the one that **shapes the whole contract** between
`forward_cache` and `forward_indexer`: every summary the two sides
exchange has to be declared here first.

Read [`program_forward_cache.md`](program_forward_cache.md) and
[`program_forward_indexer.md`](program_forward_indexer.md) first — this
file only covers the declaration.

---

## 1. What `create_cache` does

It tells the framework **what per-block summary slots to allocate, and
how big each is**. One slot per block, for every layer that uses sparse
attention. At runtime:

- In `forward_cache`, each declared name resolves to a `[1, D_0, D_1]`
  tile you can read/write for the block that just completed.
- In `forward_indexer`, the same declared name resolves to a
  `[S, D_0, D_1]` page-packed tensor covering all blocks of the
  current sequence (in the batch-size-1 mental model).

The framework handles **all** the allocation, alignment, device
placement, dtype, and paging — you only declare shapes.

---

## 2. Signature

```python
def create_cache(self, block_size: int, head_dim: int) -> Dict[str, Tuple[int, int]]:
    ...
```

| arg | what it is |
|---|---|
| `block_size` | number of tokens per block (e.g. 16). Set by the engine config `vortex_block_size`. |
| `head_dim` | feature dim per attention head (model config). |
| return value | dict mapping `name -> (D_0, D_1)` — the inner shape of each field you want |

Return a plain Python `dict`. The keys become `cache["<name>"]` in both
your forward methods. The values are pairs of positive integers.

---

## 3. The rules

1. **Do not include `"k"` or `"v"`.** The framework auto-injects
   `cache["k"]` and `cache["v"]` with inner shape `(block_size,
   head_dim)` — and actively asserts that `create_cache` does not
   return them. They're always available in `forward_cache` /
   `forward_indexer`; treat them as part of the framework, not
   something you own.
2. **Every entry is `(D_0, D_1)` — two positive ints.** These are the
   inner dims. The leading axis (1 per block in `forward_cache`, `S`
   page-packed in `forward_indexer`) is synthesised at runtime.
3. **Someone has to write each field.** Either `forward_cache` writes
   it via a cache op's `output=` argument, or `forward_indexer` writes
   it via `Save`. If nothing writes a declared field, reading it will
   only ever see the initialisation bytes.
4. **Use `block_size` and `head_dim` directly.** They're the natural
   building blocks for most summaries; everything else can be a
   constant or a flow attribute you stored in `__init__`.

---

## 4. Worked walkthroughs

Every registered reference flow has a short `create_cache`. Line up
their declarations against what each flow maintains:

### `block_sparse_attention`

```python
def create_cache(self, block_size, head_dim):
    return {
        "centroids": (1, head_dim),   # one centroid vector per block
    }
```

One user field: a `(1, head_dim)` centroid per block. In
`forward_cache`, `CMean(dim=1)(cache["k"], cache["centroids"])` collapses
the `block_size` axis of keys into this slot. In `forward_indexer`,
`GeMM(q_mean, cache["centroids"])` scores each page against its
centroid.

### `gqa_quest_sparse_attention`

```python
def create_cache(self, block_size, head_dim):
    return {
        "max": (1, head_dim),
        "min": (1, head_dim),
    }
```

Two user fields, same shape as above but holding the elementwise max
and min of the block's keys. `forward_cache` does `CMax(dim=1)` and
`CMin(dim=1)` into them; `forward_indexer` uses them for the
QUEST-style envelope bound.

### `running_avg_block_sparse`

```python
def create_cache(self, block_size, head_dim):
    return {
        "centroids":     (1, head_dim),  # centroid from keys (as in block-sparse)
        "running_score": (1, 1),         # per-block scalar accumulated across decode steps
    }
```

One `(1, head_dim)` centroid and one `(1, 1)` scalar. The scalar is
zero-initialised in `forward_cache` (via `CFill(0.0)`) and then
accumulated across decode steps in `forward_indexer` via `Load` /
`Add` / `Save`.

### `centered_block_sparse_attention`

```python
def create_cache(self, block_size, head_dim):
    return {
        "centroids": (1, head_dim),
    }
```

Same single-field declaration as the plain block-sparse flow — the
per-sequence centering happens entirely on the indexer side via
`Mean(dim=0)` and doesn't need a new cache field.

---

## 5. Common inner-shape patterns

| what you want per block | shape | typical use |
|---|---|---|
| a single vector | `(1, head_dim)` | centroid, max/min envelope, running feature mean |
| a single scalar | `(1, 1)` | running score, per-block importance, a counter |
| multiple vectors | `(C, head_dim)` | `C` centroids per block (multi-cluster routing), `C` learned projections |
| per-token scalar | `(block_size, 1)` | per-token importance weights within a block |
| per-token vector | `(block_size, head_dim)` | a transformed copy of `k` or `v` — rarely needed; usually you reduce instead |

When in doubt: the smaller the shape, the cheaper the cache is to
maintain and the less indexer bandwidth you spend reading it. `(1, 1)`
and `(1, head_dim)` cover the overwhelming majority of flows.

---

## 6. The producer/consumer contract

Every field you declare has to have a producer and a consumer. Walk
this table before you commit:

| field | who **writes** it | who **reads** it |
|---|---|---|
| `"k"` (auto) | the framework (K/V copy path) | `forward_cache` (deriving summaries from it) |
| `"v"` (auto) | the framework (K/V copy path) | `forward_cache`, optionally |
| your `"summary"` | `forward_cache` via an op's `output=` arg | `forward_indexer` to score pages |
| your `"persistent"` (e.g. running average) | `forward_cache` via `CFill` (init) + `forward_indexer` via `Save` (accumulate) | `forward_indexer` via `Load` |

Two common shapes of a full contract:

- **Derived summary**: `forward_cache` computes it from `cache["k"]` /
  `cache["v"]` at block completion; `forward_indexer` reads it per
  decode step. Examples: centroids, envelopes, per-block norms.
- **Persistent state**: `forward_cache` zeros it at block completion
  (`CFill(0.0)`); `forward_indexer` does `Load → compute → Save` each
  decode step. Example: running scores, EMAs of past queries.

If a declared field doesn't fit either pattern, it probably shouldn't
be declared.

---

## 7. What you DON'T write in `create_cache`

- No dtype. The engine's `vortex_dtype` config sets the dtype of every
  user field (default bfloat16).
- No device. The framework places each field on the same device as the
  attention kernels.
- No initial-value argument. If you want a non-zero starting value,
  write it in `forward_cache` with a `CFill` or a computed op. For
  fields that only need to be zero before the first `Load`, the
  `CFill(0.0)` in `forward_cache` handles it.
- No per-layer variation. Every layer that uses sparse attention gets
  the same set of declared fields. If you want different fields for
  different layers, split into different `vFlow`s (not usually
  needed).
- No runtime-dependent shapes. `D_0` and `D_1` are construction-time
  integers — they cannot depend on `ctx` or any runtime state.

---

## 8. Pitfalls

1. **Declaring `"k"` or `"v"`.** Framework hard-asserts; you'll get
   `AssertionError: create_cache must not declare 'k' key`.
2. **Declaring a field nothing writes.** The field gets allocated but
   stays uninitialised. When `forward_indexer` reads it, it sees zero
   bytes (freshly allocated blocks) or whatever previous sequence left
   there (reused blocks) — silently wrong rather than crashing.
3. **Reading a field that only `Save` writes, without zeroing it
   first.** The first `Load` on a new block reads stale memory. Add
   `CFill(0.0)(cache["field"], loc=loc, ctx=ctx)` in `forward_cache`.
4. **Oversized inner shapes.** Every decode step has to read every
   declared field for every candidate page. A `(block_size, head_dim)`
   slot is `block_size ×` more data than `(1, head_dim)`; it's rarely
   worth it.
5. **Using names that collide with framework names.** Stick to plain
   lowercase identifiers (`centroids`, `max`, `min`, `v_norm`,
   `running_score`). Avoid `"k"`, `"v"` (reserved).

---

## 9. Mental-model summary

- `create_cache` returns a dict of inner shapes: `name -> (D_0, D_1)`.
- You get `[1, D_0, D_1]` tiles in `forward_cache`, `[S, D_0, D_1]`
  in `forward_indexer`.
- `"k"` and `"v"` are always there — don't declare them.
- Every declared field needs a writer and a reader. Walk the contract
  once before you finish.
- `(1, head_dim)` for a per-block vector and `(1, 1)` for a per-block
  scalar handle the vast majority of flows.
