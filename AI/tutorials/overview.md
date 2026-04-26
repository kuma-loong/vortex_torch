# vortex_torch — Overview

`vortex_torch` is a **JIT-compiled sparse-attention framework**. It
plugs into sglang's decode loop and lets you define — in a few lines
of Python — which KV-cache pages the attention kernel should actually
attend to at each decode step.

You describe *what* summaries to maintain per block of the KV cache
and *how* to route queries to pages. The framework turns that
description into two fused Triton kernels (one on the cache side, one
on the decode side) and runs them inside sglang.

This document is the 5-minute map of what you need to write and where
the detail lives. Read this once, then dive into the detail files
linked at the bottom.

---

## 1. What you actually write

A single Python class — a subclass of `vFlow` — with **three methods**:

```python
from vortex_torch.flow import vFlow, register
from vortex_torch.indexer import topK, GeMM, Mean
from vortex_torch.cache   import Mean as CMean

@register("my_flow")
class MyFlow(vFlow):

    def __init__(self):
        super().__init__()
        # op instances live here; each call site needs its own instance
        self.q_mean       = Mean(dim=1)
        self.gemm         = GeMM()
        self.output_func  = topK()
        self.k_mean       = CMean(dim=1)

    # (1) Declare what per-block summary slots you want.
    def create_cache(self, block_size, head_dim):
        return {
            "centroids": (1, head_dim),   # one centroid vector per block
        }

    # (2) Compute & persist those summaries when a new block completes.
    def forward_cache(self, cache, loc, ctx):
        self.k_mean(cache["k"], cache["centroids"], loc=loc, ctx=ctx)

    # (3) Route queries to pages each decode step: build a per-page
    #     score of shape [S, 1, 1] and hand it to topK (or its faster
    #     approximate variant approxTopK).
    def forward_indexer(self, q, o, cache, ctx):
        qm     = self.q_mean(q, ctx=ctx)                         # [1, 1, D]
        score  = self.gemm(qm, cache["centroids"], ctx=ctx)      # [S, 1, 1]
        self.output_func(score, o, ctx=ctx)                      # writes top-k page ids
```

> **Choice of terminal op:** `self.output_func` above is `topK()`, the
> exact selector. For a faster approximate alternative, swap in
> `approxTopK(tolerate_ratio=t)` with `t ∈ [0.0, 1.0]` (`0.0` = exact,
> higher = cheaper-but-looser; output indices unsorted). Same call
> shape, same BOS/EOS guarantees. See
> [`indexer_op.md §9`](indexer_op.md) for the full math + tuning hints.

That's the entire user surface. You don't write kernels, loops,
allocations, or any plumbing — the framework handles all of it.

---

## 2. The mental model

Pretend the system has exactly **one sequence** in flight:

- `q` (in `forward_indexer`) has shape `[1, H_q, D]`.
- Every `cache["<name>"]` is `[S, D_0, D_1]` on the indexer side
  (`S` = number of blocks for that one sequence) and `[1, D_0, D_1]`
  on the cache side (the one block that just completed).
- `cache["k"]` and `cache["v"]` are always there — you don't declare
  them.
- Reductions, broadcasting, matmul, etc. work like NumPy on these
  shapes.

The framework replays your one-sequence program across the real
batch and kv-head axes. You never write batch loops, page-table
lookups, or Triton code.

---

## 3. The goal of each method

| method | what it does | the contract |
|---|---|---|
| `create_cache(block_size, head_dim)` | declares auxiliary cache fields (centroids, envelopes, running scores, …) | returns a `{name: (D_0, D_1)}` dict. **Don't** include `"k"` / `"v"`. |
| `forward_cache(cache, loc, ctx)` | computes per-block summaries from `cache["k"]` / `cache["v"]` and writes them into your declared fields | runs once per block completion; purely side-effect writes |
| `forward_indexer(q, o, cache, ctx)` | builds a per-page score and picks top-k pages | must terminate in `topK(score, o, ctx=ctx)` *or* `approxTopK(tolerate_ratio=…)(score, o, ctx=ctx)` with `score.shape == [S, 1, 1]` |

**Rules of thumb**:

1. **Op instances are not reusable.** Two call sites → declare two
   instances.
2. **No native torch ops** anywhere in the flow. Every tensor goes
   through `vortex_torch.indexer.*` or `vortex_torch.cache.*`.
3. **Every declared field needs a writer and a reader.** Walk the
   contract once before you finish.

---

## 4. How it slots into sglang

At runtime:

1. **Token write (cache side).** Every time new K/V tokens land, the
   framework copies them into the paged KV pool. If the just-written
   token is the last of its block, `forward_cache` fires and updates
   your summary fields.
2. **Decode step (indexer side).** Once per layer per decode step,
   `forward_indexer` reads the current query + summary fields,
   computes a score, and picks top-k page ids.
3. **Sparse attention.** flashinfer's attention kernel attends only
   to the pages your indexer picked.

Layers listed in `vortex_layers_skip` bypass both of these — they run
dense flashinfer with no summary update and no routing. (Bottom
layers usually go here.)

---

## 5. Engine configuration

Every flow ships with a tiny JSON that the engine reads at boot.
Example: `submissions/example_block_sparse_attention.json`:

```json
{
  "vortex_module_path":       "submissions/example_block_sparse_attention.py",
  "vortex_module_name":       "example_block_sparse_attention_cls",
  "vortex_block_size":        16,
  "vortex_workload_chunk_size": 32,
  "vortex_topk_val":          29,
  "vortex_topk_ratio":        0.0625,
  "vortex_block_reserved_bos": 1,
  "vortex_block_reserved_eos": 2,
  "vortex_layers_skip":       [0],
  "vortex_dtype":             "bfloat16",
  "kv_cache_dtype":           "auto"
}
```

| key | what it controls |
|---|---|
| `vortex_module_path` / `vortex_module_name` | where to find your `@register(...)`'d vFlow |
| `vortex_block_size` | tokens per block (power of 2) |
| `vortex_workload_chunk_size` | indexer workload size (power of 2) |
| `vortex_topk_val` / `vortex_topk_ratio` | how many pages to select per sequence |
| `vortex_block_reserved_bos` / `_eos` | always-selected first / last segments |
| `vortex_layers_skip` | layer ids that stay dense |
| `vortex_dtype` | dtype for intermediate tensors (`bf16` default, `fp16`, `fp32`, `fp8_e5m2`, `fp8_e4m3`) |
| `kv_cache_dtype` | dtype for the K/V cache itself |

Validate the JSON before booting the engine:

```python
from vortex_torch.engine.sgl import check_engine_config
check_engine_config("submissions/my_flow.json")
```

It parses the JSON, checks every setting, loads your module, and runs a
small compile sweep. If it returns without raising, sglang can boot
this flow.

---

## 6. Where to go next (the five detail tutorials)

Read these in order the first time through. Each stays entirely in the
user's mental model — no kernel internals, no format dispatch tables,
no Triton.

1. **[`program_create_cache.md`](program_create_cache.md)** — how to
   declare your auxiliary cache fields. Shapes, common patterns, and
   the producer/consumer contract for each declared field.
2. **[`program_forward_cache.md`](program_forward_cache.md)** — how
   to write `forward_cache`. Runs once per block; build summaries from
   `cache["k"]` / `cache["v"]` and persist them.
3. **[`program_forward_indexer.md`](program_forward_indexer.md)** —
   how to write `forward_indexer`. Build a `[S, 1, 1]` score per
   decode step and hand it to `topK` (exact) or `approxTopK`
   (faster, adaptive radix). Covers `Save` / `Load` for cross-step
   persistent state.
4. **[`cache_op.md`](cache_op.md)** — math reference for every op
   available in `vortex_torch.cache` (`CMean`, `CMax`, `CMin`,
   `CL2Norm`, `CGeMM`, `CAdd`, `CFill`, `MaskSlice`, …). One formula
   and one shape table per op.
5. **[`indexer_op.md`](indexer_op.md)** — math reference for every op
   in `vortex_torch.indexer` (`GeMM`, the reductions, `Softmax`,
   `Add`, `Multiply`, `Maximum`, `MaskSlice`, `Save`, `Load`, `topK`,
   `approxTopK`, …). Same format as `cache_op.md`.

Plus two references you'll want once you're iterating:

- **`vortex_torch/flow/algorithms.py`** — five fully-worked reference
  flows (`block_sparse_attention`, `gqa_block_sparse_attention`,
  `gqa_quest_sparse_attention`, `masked_quest_sparse_attention`,
  `centered_block_sparse_attention`, `running_avg_block_sparse`). Every
  detail doc above walks at least one of these line-by-line.
- **`vortex_torch.flow.verify.verify_flow_compilable(...)`** — a
  CPU-only compile sweep over `(G, D, block_size, page_size)`. Run it
  on every new flow before booting sglang; it surfaces shape or
  dispatch mistakes in seconds.

### If you want to develop `vortex_torch` itself

The five files above are written for **users of the framework** —
people writing new sparse-attention flows. They deliberately hide the
tensor-format machinery, workload scheduling, and indptr/indices
plumbing.

If you're instead **working on the framework itself** (adding a new
op, touching the compiler, changing how cache / indexer kernels are
scheduled), read **[`tensor.md`](tensor.md)** next. It documents the
internal `BATCHED` / `PAGED` / `RAGGED` formats, the
`dense_kv_indptr` / `dense_kv_indices` addressing, workload chunking,
the `winfo_*` arrays, and the per-format Triton pointer expressions
the compiler emits. Everything the user-facing tutorials hide is
spelled out there.

---

## 7. Typical authoring loop

1. Sketch `create_cache` — what summaries do you want?
2. Write `forward_cache` — how do you compute each summary from
   `cache["k"]` / `cache["v"]`?
3. Write `forward_indexer` — how do you score pages using `q` + those
   summaries? Make sure it ends in `topK(score, o)` (exact) or
   `approxTopK(tolerate_ratio=…)(score, o)` (faster) with `score` of
   shape `[S, 1, 1]`.
4. Run `verify_flow_compilable(my_flow)` (or `check_engine_config` on
   your JSON) — fix whatever it complains about.
5. Point the engine JSON at your file and launch sglang.

If something looks right on paper but wrong at runtime, open the
compiled `.py` dumped to `~/.vortex_compilation_cache/` — it's real
Python with Triton kernels inline, and reading it tells you exactly
what the compiler thinks each op computes.
