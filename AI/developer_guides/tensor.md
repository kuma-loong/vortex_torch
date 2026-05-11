# Tensor layouts in `vortex_torch`

This document explains the tensor formats and addressing conventions
used by `vortex_torch`. The key thing to internalise upfront:

> **The indexer and the cache pipelines have separate tensor type
> systems.** Cache has *no* BATCHED format. The two pipelines share
> physical paged storage but interpret its leading axis differently.
> Treating "RAGGED" as one concept across both pipelines will trip you up
> — they're addressed differently because the surrounding kernels run
> on different program grids.

The rest of this file is structured in three parts:

1. **Indexer tensor model** — `BATCHED`, `PAGED`, `RAGGED` and the
   `dense_kv_indptr/indices` plumbing.
2. **Cache tensor model** — `PAGED`, `RAGGED` (only; no BATCHED) and
   the `loc`-driven, per-token gated kernel.
3. **The shared paged buffer** — how the two sides agree on `cache['k']`
   and friends despite using different addressing math.

Read [`overview.md`](overview.md) first for the framework-level picture;
this file fills in the addressing math and walks two reference flows
(`block_sparse_attention` and `gqa_quest_sparse_attention`) annotated
with shapes and formats at every step.

---

# Part A — Indexer tensor model

The indexer kernel runs on a **per-workload** grid. A "workload" is
`workload_chunk_size` consecutive blocks of one `(batch, kv_head)`'s
ragged span. A single Triton program walks several workloads in a
`for i in range(start, end):` loop.

## A.1. The three formats

| format | leading axis indexed by | physical leading-dim size | role |
|---|---|---|---|
| **BATCHED** | `(batch_id, kv_head_id)` linearised → `b * num_kv_heads + h` | `max_bs * num_kv_heads` | the query `q`; per-(batch, head) summaries (e.g. `Reduce(dim=0)` outputs) |
| **PAGED** | absolute `block_id` in the cache pool | `max_num_blocks` | every cache field (`k`, `v`, `centroids`, `max`, …) |
| **RAGGED** | position in the *flat per-(batch, head) listing* of valid blocks | `max_num_blocks` | per-page intermediates and the final score |

The interesting distinction is **PAGED vs RAGGED** on the indexer side:

- A **PAGED** row is at the **block's permanent address** in the paged
  pool. Two `(batch, head)` entries that happen to share a `block_id`
  (rare, but possible) read the same PAGED row.
- A **RAGGED** row is at the **position-within-the-(batch, head)-listing**.
  Rows for `(batch=b, head=h)` occupy the contiguous span
  `[dense_kv_indptr[b*H_kv + h], dense_kv_indptr[b*H_kv + h + 1])`.

The distinction matters because **the score `topK` consumes is RAGGED**,
not PAGED. `topK` walks each `(batch, head)`'s ragged span using
`dense_kv_indptr`, picks the top-`k` row positions, looks up the
corresponding `block_id`s through `dense_kv_indices`, and writes them to
`sparse_kv_indices`.

## A.2. The indptr / indices vocabulary

Four arrays, all populated by the C++ planner
(`vortex_torch.indexer.utils_sglang.get_decode_planner`) before each
batch:

| name | shape | dtype | meaning |
|---|---|---|---|
| `dense_kv_indptr` | `(max_bs * num_kv_heads + 1,)` | `int32` | CSR-style. `dense_kv_indptr[b*H_kv + h]` = first row in the flat listing belonging to `(batch=b, kv_head=h)`. |
| `dense_kv_indices` | `(max_num_blocks,)` | `int32` | The flat listing itself. `dense_kv_indices[ragged_idx]` is the absolute `block_id` for that ragged position. |
| `sparse_kv_indptr` | same as dense | `int32` | Post-`topK` sparse CSR — written by `topK`. |
| `sparse_kv_indices` | same as dense | `int32` | The selected `block_id`s — written by `topK`, then handed to flashinfer. |

A small worked example. Suppose `B=2`, `num_kv_heads=2`, and the cached
sequences hold:

| `(batch, head)` | num blocks | absolute `block_id`s |
|---|---|---|
| (0, 0) | 5 | 3, 7, 11, 12, 18 |
| (0, 1) | 5 | 4, 9, 15, 22, 27 |
| (1, 0) | 3 | 5, 8, 16 |
| (1, 1) | 3 | 6, 14, 19 |

The planner produces:

```
dense_kv_indptr  = [0, 5, 10, 13, 16]                                 # length = 4 + 1
dense_kv_indices = [3, 7, 11, 12, 18,
                    4, 9, 15, 22, 27,
                    5, 8, 16,
                    6, 14, 19]                                        # length = 16
```

For `(batch=1, head=0)` (linear id `1*2 + 0 = 2`):

- ragged span: `[dense_kv_indptr[2], dense_kv_indptr[3]) = [10, 13)`.
- absolute block ids: `dense_kv_indices[10:13] = [5, 8, 16]`.

So a PAGED tensor of size 32 has its rows 5 / 8 / 16 inhabited for this
entry; a RAGGED tensor of size 16 has its rows 10 / 11 / 12 inhabited.

`topK` with `k=2` for this `(batch=1, head=0)` would read scores at
RAGGED rows 10..12, pick e.g. the two highest, and write
`sparse_kv_indices[...] = [5, 16]` plus
`sparse_kv_indptr[3] - sparse_kv_indptr[2] = 2` so flashinfer knows the
sparse span length.

## A.3. Workload chunking — how the kernel walks RAGGED

The planner additionally splits each `(batch, head)`'s ragged span into
**workloads** of `workload_chunk_size` consecutive blocks (the last one
may be shorter). Per workload `j`:

```
winfo_q_indices[j]                   = (b * H_kv + h)        ← BATCHED row to load for q
winfo_kv_offsets[j]                  = ragged_idx of the first block in this workload
winfo_kv_lens[j]                     = number of valid blocks (≤ workload_chunk_size)
winfo_is_first_workload_per_batch[j] = 1 iff this is the first workload of (b, h)
winfo_num_workloads                  = total workloads scheduled
```

Inside the W kernel each Triton program picks a slice of workloads via
`pid` and runs `for i in range(start, end):`. Within one iteration:

```python
new_batch_idx_i32 = tl.load(winfo_x_indices + i)             # (b * H_kv + h)
ragged_idx_i32    = tl.load(winfo_y_offsets + i)             # first ragged row of this workload
page_idx_i32      = tl.load(indices + ragged_idx_i32)        # single-page case
# or, when num_pages_per_workload > 1:
page_indices_i32  = tl.load(indices + ragged_idx_i32
                            + page_idx_i32_ptr * num_blocks_per_page)
```

`indices` is the kernel-level alias for `dense_kv_indices`. So the load
`tl.load(indices + ragged_idx_i32)` *is* the ragged → paged translation:
the workload's first block lives at absolute
`block_id = dense_kv_indices[ragged_idx_i32]`.

This is the kernel-level mechanism that lets a single fused W kernel
load a BATCHED `q` (one row per `(batch, head)`), several PAGED cache
fields (rows at the absolute `page_idx_i32`s), and write a RAGGED score
(rows at `ragged_idx_i32 + workload_ptr`).

## A.4. Indexer-side address expressions

The compiler emits these literal Triton pointer expressions:

### BATCHED

```
addr(BATCHED, batch=b, head=h) = ptr + (b * H_kv + h) * (D_0 * D_1) + ...
```

Loads broadcast to a `(1, D_0, D_1)` block (the leading 1 is artificial:
each workload processes one `(batch, head)` row at a time).

Stores are gated on `winfo_is_first_workload_per_batch[i]` to avoid
redundant writes when multiple workloads share the same `(batch, head)`.

### PAGED

```
addr(PAGED, block_id) = ptr + block_id * (D_0 * D_1) + ...
```

Loads a `(D_0, D_1)` block per active block in the workload. When a
workload covers multiple blocks, the leading axis broadcasts to
`(workload_chunk_size, D_0, D_1)` via the `page_indices_i32[:, None]`
indirection through `dense_kv_indices`. Indexer-side ops do not store
into PAGED — that's the cache pipeline's job (or an explicit `Save` op).

### RAGGED

```
addr(RAGGED, ragged_idx) = ptr + ragged_idx     * (D_0 * D_1)
                              + workload_ptr[:, None, None] * (D_0 * D_1)
                              + dim1_ptr[None, :, None] * D_1
                              + dim2_ptr[None, None, :]
```

`workload_ptr = tl.arange(0, workload_chunk_size)` so a single workload
writes its entire chunk to consecutive ragged rows starting at
`ragged_idx_i32`. Stores are masked by `valid = workload_ptr < _len`
where `_len = winfo_kv_lens[i]`.

## A.5. Indexer end-to-end: `block_sparse_attention`

Setup: `B = 2`, `num_kv_heads = 2`, `G = 4` (so `H_q = 4` per kv-head),
`D = 128`, `block_size = 16`, `page_size = 16`, `num_blocks_per_page = 1`.
Worst-case `max_num_blocks = 256`.

```python
def forward_indexer(self, q, o, cache, ctx):
    q_mean = self.mean(q, ctx=ctx)
    score  = self.gemm(q_mean, cache["centroids"], ctx=ctx)
    self.output_func(score, o, ctx=ctx)
```

| line | tensor | format | logical shape | physical leading dim |
|---|---|---|---|---|
| input | `q`                  | BATCHED | `[B, H_q=4, D=128]`   | `max_bs * H_kv = 4` |
| input | `cache["centroids"]` | PAGED   | `[S, 1, D=128]`       | `max_num_blocks=256` |
| 1     | `q_mean = mean(q, dim=1)` | BATCHED | `[B, 1, D=128]`   | 4 |
| 2     | `score = gemm(q_mean, centroids)` | RAGGED | `[S, 1, 1]` | 256 |
| 3     | `topK(score, o)`     | -       | -                     | -   |

Per-`(batch, head)` view of `score` (using the example from A.2):

```
score[ 0]  ← (b=0,h=0) block #0  ← cache['centroids'][block_id=3]
score[ 1]  ← (b=0,h=0) block #1  ← cache['centroids'][block_id=7]
...
score[ 4]  ← (b=0,h=0) block #4  ← cache['centroids'][block_id=18]
score[ 5]  ← (b=0,h=1) block #0  ← cache['centroids'][block_id=4]
...
score[10]  ← (b=1,h=0) block #0  ← cache['centroids'][block_id=5]
score[11]  ← (b=1,h=0) block #1  ← cache['centroids'][block_id=8]
score[12]  ← (b=1,h=0) block #2  ← cache['centroids'][block_id=16]
score[13]  ← (b=1,h=1) block #0  ← cache['centroids'][block_id=6]
...
```

`topK` then walks each `(batch, head)`'s ragged span, picks the top-`k`
row indices, looks up `block_id`s through `dense_kv_indices`, and writes
them to `sparse_kv_indices`.

## A.6. Indexer end-to-end: `gqa_quest_sparse_attention`

Same setup. QUEST keeps per-block `max` and `min` envelopes of the keys,
then bounds the per-page attention score from above using
`q · max_envelope` and `q · min_envelope`.

```python
def forward_indexer(self, q, o, cache, ctx):
    s_max  = self.mul_max(q, cache["max"], ctx=ctx)
    s_min  = self.mul_min(q, cache["min"], ctx=ctx)
    s      = self.maximum_op(s_max, s_min, ctx=ctx)
    score  = self.sum(s, ctx=ctx)
    aggr   = self.max_op(score, ctx=ctx)
    self.output_func(aggr, o, ctx=ctx)
```

| line | tensor | format | logical shape | physical leading dim |
|---|---|---|---|---|
| input | `q`                  | BATCHED | `[B, H_q=4, D=128]` | 4 |
| input | `cache["max"]`       | PAGED   | `[S, 1, D=128]`     | 256 |
| input | `cache["min"]`       | PAGED   | `[S, 1, D=128]`     | 256 |
| 1     | `s_max = q * max`    | RAGGED  | `[S, H_q=4, D=128]` | 256 |
| 2     | `s_min = q * min`    | RAGGED  | `[S, H_q=4, D=128]` | 256 |
| 3     | `s = max(s_max, s_min)` | RAGGED | `[S, H_q=4, D=128]` | 256 |
| 4     | `score = sum(s, dim=2)` | RAGGED | `[S, H_q=4, 1]`   | 256 |
| 5     | `aggr = max(score, dim=1)` | RAGGED | `[S, 1, 1]`    | 256 |
| 6     | `topK(aggr, o)`      | -       | -                   | -   |

All five W ops in steps 1–5 fuse into **one** Triton kernel — every
intermediate stays in registers because they're all `Schedule.W` and the
format chain stays RAGGED after step 1. Step 6 is `Schedule.S` so it's
a separate kernel launch.

---

# Part B — Cache tensor model

The cache kernel runs on a fundamentally different grid: one Triton
program per `(token_id, head_id)`, gated to fire only when the token
completes a block. There is **no workload chunking, no `winfo_*`, no
ragged → paged indirection through `dense_kv_indices`**, and **no
BATCHED format** at all.

## B.1. Two formats only — PAGED and RAGGED

| format | leading axis indexed by | physical leading-dim size | role |
|---|---|---|---|
| **PAGED** | absolute `block_id` derived from `(token_position, head_id)` | `total_num_blocks` (= `total_num_pages * num_blocks_per_page`) | every cache field (`k`, `v`, `centroids`, `max`, …) — both reads and writes |
| **RAGGED** | linearised `(token_id, head_id)` → `token_id * num_kv_heads + head_id` | `max_new_tokens_per_batch * num_kv_heads` | per-trigger-token intermediates that live entirely inside one fused per-block kernel |

There is no BATCHED here because there is no notion of "the whole
sequence" inside a cache kernel: each program already owns exactly one
`(batch, head, token)` and writes one block. Cross-batch or cross-head
reductions don't exist in the cache pipeline. (If you need them, do them
on the indexer side via `Reduce(dim=0)`.)

## B.2. The driver: `loc` and the block-completion gate

The cache kernel takes a single per-token positions tensor:

| name | shape | dtype | meaning |
|---|---|---|---|
| `loc` | `(num_new_tokens,)` | `int64` | `loc[token_id]` is the **absolute position** of the token within its `(batch, head)` sequence (i.e. the slot in `cache['k']` it just landed in) |

Grid: one Triton program per `(token_id, head_id)` for `token_id in
range(num_new_tokens)` and `head_id in range(num_kv_heads)`. Inside the
program:

```python
token_id      = tl.program_id(0)
head_id       = tl.program_id(1)
token_position = tl.load(loc + token_id)

# Trigger only when this token finishes a block:
if (token_position + 1) % BLOCK_SIZE != 0:
    return

# Compute the block_id this token completed:
page_id  = (token_position // PAGE_SIZE) * NUM_KV_HEAD + head_id
block_id = page_id * NUM_BLOCKS_PER_PAGE + (token_position % PAGE_SIZE) // BLOCK_SIZE
```

The gate is what makes the cache pipeline efficient: most token writes
short-circuit, and the heavy work (loading the block, computing
summaries, storing back) only fires once every `BLOCK_SIZE` tokens.

This block-completion semantic is also why cache ops can't accumulate
state across blocks within a single pass — by the time a block fires,
the next block's tokens haven't been written yet, so `cache['k']`
*beyond* the current block is stale or zero.

## B.3. Cache-side address expressions

### PAGED (cache fields — both load and store)

```
addr(PAGED, block_id) = ptr + block_id * (D_0 * D_1)
                            + dim1_ptr[:, None] * D_1
                            + dim2_ptr[None, :]
```

The `block_id` comes from the per-program `(token_position, head_id)`
math above. Reads are unmasked (the block is guaranteed to be in
range — that's why we gated on `(token_position + 1) % BLOCK_SIZE`).
Writes go straight back to the same slot, often a different field of the
same `block_id` (e.g. read `cache['k'][block_id]`, write
`cache['centroids'][block_id]`).

### RAGGED (cache-side intermediates)

```
addr(RAGGED, token, head) = ptr + (token_id * NUM_KV_HEAD + head_id) * (D_0 * D_1)
                                + dim1_ptr[:, None] * D_1
                                + dim2_ptr[None, :]
```

This is a token-major addressing scheme that's only used when a
fused per-block kernel needs to pass an intermediate from one cache op
to another within the same program. In practice the compiler keeps these
in registers; the RAGGED layout matters mostly for sizing the buffer.

## B.4. Cache end-to-end: `block_sparse_attention`

```python
def forward_cache(self, cache, loc, ctx):
    self.reduction(cache["k"], cache["centroids"], loc=loc, ctx=ctx)
```

| line | tensor | format | logical shape | comment |
|---|---|---|---|---|
| input | `cache["k"]`         | PAGED | `[B, page_size=16, D=128]` | the just-completed block of keys |
| input | `cache["centroids"]` | PAGED | `[B, 1, D=128]`            | output destination (one centroid per block) |
| 1     | `CMean(k → centroids, dim=1)` | PAGED | writes `[B, 1, D]` | mean over the 16 token slots, written back to `cache['centroids'][block_id]` |

The `[B, ...]` notation in the table is a *per-block* view: `B` here is
"the block this program owns", logically size 1 — not the request
batch size from the indexer. The same physical buffer that the indexer
sees as `cache['centroids']: [S, 1, D]` (page-packed) is seen here as
"the row at `block_id`", a `(1, D)` tile per program.

## B.5. Cache end-to-end: `gqa_quest_sparse_attention`

```python
def forward_cache(self, cache, loc, ctx):
    self.reduction_max(cache["k"], cache["max"], loc=loc, ctx=ctx)
    self.reduction_min(cache["k"], cache["min"], loc=loc, ctx=ctx)
```

| line | tensor | format | logical shape | comment |
|---|---|---|---|---|
| input | `cache["k"]`   | PAGED | `[B, page_size=16, D=128]` | the just-completed block of keys |
| input | `cache["max"]` | PAGED | `[B, 1, D=128]`            | output destination |
| input | `cache["min"]` | PAGED | `[B, 1, D=128]`            | output destination |
| 1     | `CMax(k → max, dim=1)` | PAGED | writes `[B, 1, D]` | per-feature max over the 16 token slots |
| 2     | `CMin(k → min, dim=1)` | PAGED | writes `[B, 1, D]` | per-feature min |

Both reductions land in the **same fused per-block kernel** — they share
the `cache['k']` block load and emit two stores. Both reads and stores
go through the same `block_id` derived from `token_position`.

---

# Part C — The shared paged buffer

The two pipelines never directly share an address-space view; they share
the underlying `torch.Tensor`. A single field declared by `create_cache`
has two interpretations:

| pipeline | view | leading axis | element addressed by |
|---|---|---|---|
| **cache** | per-block, fired by token completion | `B` = "this program's block" (logical size 1) | `block_id = f(token_position, head_id)` |
| **indexer** | page-packed, walked by workload | `S` = `max_num_blocks` | `dense_kv_indices[ragged_idx]` |

The cache writes `cache['centroids'][block_id]` from `loc`-driven
`(token_position, head_id)`; the indexer reads `cache['centroids']` at
PAGED rows whose `block_id`s come from `dense_kv_indices[ragged_idx]`.
Both expressions resolve to the same physical row in the same paged
`torch.Tensor` — the framework guarantees the math agrees because both
sides use the same `(num_blocks_per_page, page_size, num_kv_heads)`
constants pulled from the runtime context.

This is the discipline that lets you write `forward_cache` and
`forward_indexer` independently: the cache side describes *what* to
maintain per block, the indexer side describes *how* to consume per
block — they meet at the absolute `block_id`.

---

# Part D — Checklist for new flows

When you sketch a new `vFlow`, walk through these in order:

1. **Indexer inputs**: `q` is BATCHED `[B, H_q, D]`; every cache field
   is PAGED `[S, r, c]` with `(r, c)` declared by `create_cache`.
2. **Indexer output format**: compiler-generated ops (every op except
   `Save` / `Load`, `topK` / `approxTopK`, `Softmax` / `Normalize` /
   `Conv1d`, and `Reduce(dim=0)`) derive their output format inline:
   `BATCHED` iff every input is `BATCHED`, else `RAGGED`. The custom
   kernels listed above still use an explicit `_impl_map` /
   `_supported_formats` table — check the table when you mix
   formats into one of them.
3. **Score contract**: the indexer chain *must* terminate in a RAGGED
   `[S, 1, 1]` score handed to `topK(score, o)`. Fold extra `H_q` / `D`
   axes with `Mean` / `Max` / `Sum`.
4. **Cross-row aggregation**: if you need per-(batch, head) summaries
   (mean / max / etc. across all of one entry's blocks), use
   `Reduce(dim=0)` on a RAGGED tensor — output is BATCHED. Mix back
   into a RAGGED score via `Add(alpha=1, beta=-1)` or `Multiply` (both
   support `(RAGGED, BATCHED) → RAGGED`).
5. **Cache inputs**: every cache field is PAGED `[B, r, c]` (the per-block
   view; `B` here is the program's single block). No BATCHED. No `q`.
6. **Cache output format**: the output is `PAGED` iff the caller
   passed an `output=cache["..."]` argument with `PAGED` format
   (writing back into a cache field), otherwise `RAGGED` (an
   auto-allocated intermediate within the fused per-block kernel).
   No `_impl_map` dispatch — just the caller's choice.
7. **Cache restriction**: `Reduce(dim=0)` is forbidden — there's no
   cross-block aggregation in one cache pass.
8. **Schedule barriers**: every `Schedule.S` op (`topK`, `Softmax`,
   `Normalize`, `Reduce(dim=0)`) materialises its inputs to a real
   buffer. Two W chains separated by an S op is the cost model for
   "force materialisation".

When in doubt: write the flow, run `verify_flow_compilable(flow)`, then
open the generated `.py` files in `~/.vortex_compilation_cache` and read
the kernel. The shape constants in the kernel signature are the ground
truth for what the compiler thinks each tensor's leading dim is.
