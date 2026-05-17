# Supporting a block-table-based indexer (trtllm backend)

This guide is for framework developers (not submission authors). It
describes how to extend the vortex_torch indexer + compiler so every op
that currently consults the **CSR** `(dense_kv_indices, dense_kv_indptr)`
form can also consume the **2D block-table** form used by the trtllm
attention backend, and how to finish the migration the existing
TODOs in the code anticipate.

The trtllm planner already produces both layouts side-by-side; the
remaining work is on the **consumers** (the compiler-emitted Triton
kernels and the workload scheduler that feeds them). When that's done
the CSR write can be dropped from the trtllm planner entirely.

> **Correctness invariant.** This migration is a *layout
> reinterpretation*, not an algorithm change. The same elements are
> selected and the same reductions are computed on the same KV
> cache pages — only the address arithmetic differs (constant row
> stride vs prefix-sum offset). Numerically the outputs will not be
> bit-identical to the flashinfer/CSR path: the trtllm decode
> kernel and the per-op Triton kernels reorder fp32 accumulations
> differently, and tensor-core MMAs can change last-bit results
> when input strides change. But the *math* is unchanged, so
> downstream task accuracy must hold. The expected validation bar
> is **RULER ≥ 97%** on the BT path for any submission that
> already passes RULER on the flashinfer/CSR path. Anything lower
> is a layout bug, not numerical drift — see §4.

---

## 0. Why two layouts coexist today

The indexer pipeline has three phases that all reach for "which page
holds the next chunk of KV for this (request, kv-head)":

1. **Planner** — a CUDA C++ kernel that walks `req_to_token` and emits
   per-(req, kv-head) page selections.
2. **Workload scheduler** — partitions the dense selection into
   workload chunks (`winfo_*` arrays) that one Triton program will
   process at a time.
3. **Op codegen** — the Triton kernel(s) the compiler emits per op.
   These read the page id with `tl.load(indices + ragged_idx_i32)` and
   then load/store the actual KV cache slab.

For the flashinfer backend the canonical layout is a flat **CSR**
buffer:

  - `dense_kv_indptr  : [eff_bs + 1]`  (where `eff_bs = batch * num_kv_heads`)
  - `dense_kv_indices : [Σ pages]`     flat page ids, row `i` is
    `[dense_kv_indptr[i] : dense_kv_indptr[i+1])`
  - mirror `sparse_kv_indptr / sparse_kv_indices` for the sparse path.

The trtllm decode kernel
(`flashinfer.decode.trtllm_batch_decode_with_kv_cache`) wants a 2D
**block-table** instead:

  - `dense_block_tables  : [eff_bs, max_blocks_per_seq]` int32
  - `sparse_block_tables : [eff_bs, max_blocks_per_seq]` int32
  - `dense_seqlens / sparse_seqlens : [eff_bs]` int32 token counts
  - (plus the indptr arrays — see below — which the workload
    scheduler still needs.)

Both layouts encode the same information; only the row stride
differs (CSR uses `dense_kv_indptr[i+1] - dense_kv_indptr[i]`,
block-table uses a fixed `max_blocks_per_seq`). The current code keeps
both so callers can opt in incrementally:

  - The **trtllm planner** (`sglang_plan_decode_v2_trtllm` in
    [vortex_torch/indexer/planner_sglang.py](vortex_torch/indexer/planner_sglang.py))
    writes `dense_block_tables`, `sparse_block_tables`,
    `dense_seqlens`, `sparse_seqlens` **and** the legacy
    `dense_kv_indices` (CSR). The CSR write is a TODO marked
    `opt-b` — every consumer-side migration below is what unblocks
    dropping it.
  - The **topk codegen** (Schedule.S) at
    [vortex_torch/indexer/compiler/triton_impl/topk.py](vortex_torch/indexer/compiler/triton_impl/topk.py)
    already branches on `ctx.vortex_attention_backend` and reads
    `ctx.dense_block_tables` directly in trtllm mode.
  - **Every other op** (Schedule.W) still goes through the CSR path
    via the workload-kernel preamble in
    [vortex_torch/indexer/compiler/triton_impl/kernel_gen.py](vortex_torch/indexer/compiler/triton_impl/kernel_gen.py)
    (`page_idx_i32 = tl.load(indices + ragged_idx_i32)`).

The migration has **two coupled parts**:

  1. **Page-id resolution** — Schedule.W's per-workload preamble and
     the workload scheduler must resolve `page_idx_i32` from
     `dense_block_tables[row, col]` instead of
     `dense_kv_indices[ragged_idx_i32]`.
  2. **RAGGED tensor reinterpretation** — every consumer of a
     `FORMAT.RAGGED` tensor (Schedule.W RAGGED store/load preamble +
     **every** Schedule.S op: softmax, normalize, conv1d,
     reduce(dim=0), topk, approxtopk) must switch from
     "flat-packed buffer keyed by `dense_kv_indptr`" to
     "row-padded 2D buffer keyed by `(eff_bs, max_blocks_per_seq)`
     with valid counts in `dense_seqlens` / `sparse_seqlens`". See
     §2.4 — this is the larger change.

Both parts share the same `vortex_attention_backend` switch and land
together for any given submission.

---

## 1. The three surfaces to touch

When you add a new sparse-attention algorithm, or extend the framework
to a new attention backend that uses block-tables (e.g. a future H200
trtllm variant), three things have to agree on the page-id layout:

| Surface | File | What it does |
|---|---|---|
| **Planner** (incl. workload scheduler) | `vortex_torch/indexer/planner_sglang.py`, `vortex_torch/indexer/utils_sglang.py` (`get_decode_planner_trtllm`) | Picks pages, writes `*_block_tables` / `*_seqlens` / `*_kv_indptr`. The workload scheduler is part of this kernel — it produces `winfo_*`. In BT mode `winfo_kv_offsets[i]` must mean **block-in-row column**, not flat-CSR offset. |
| **Schedule.W ops** | `vortex_torch/indexer/compiler/triton_impl/{kernel_gen.py, save_load.py, gemm.py, reduce.py, ...}` | Inlined into the per-workload kernel. Two things change here: (a) `page_idx_i32` resolution moves from `dense_kv_indices` to `dense_block_tables[row, col]`; (b) the RAGGED store/load preamble switches from `ragged_idx_i32 * shape[1]` to `(row*max_blocks_per_seq + col) * shape[1]`. |
| **Schedule.S ops** | `vortex_torch/indexer/compiler/triton_impl/{topk.py, softmax.py, normalize.py, conv1d.py, reduce.py::generate_reduce_dim0_impl}` | Standalone kernels. **All** of them walk a RAGGED tensor per row via `dense_kv_indptr` today and must move to `(dense_seqlens, max_blocks_per_seq)`. `topK` / `approxTopK` additionally mutate per-row counts (input is `dense_seqlens`, output is `sparse_seqlens`). |

The user's "W-schedule op vs non-W-schedule op" distinction maps onto
the framework's `Schedule.W` vs `Schedule.S` enum, defined at
[vortex_torch/utils.py:13-15](vortex_torch/utils.py#L13-L15) and
keyed in
[vortex_torch/indexer/compiler/triton_impl/register.py:40-62](vortex_torch/indexer/compiler/triton_impl/register.py#L40-L62).

  - `Schedule.W` ops are fused **into** the per-workload Triton
    kernel emitted by `generate_triton_kernel` (kernel_gen.py).
    They never see the page id directly — they consume
    `tensor_<tid>_block` which the surrounding workload-kernel
    preamble already gathered from the right page.
  - `Schedule.S` ops launch their own kernel and are responsible for
    their own page-id resolution. `topk` already does this in a
    backend-aware way.

**Both Schedule.W and Schedule.S have substantive work.** Schedule.W
loses its `dense_kv_indices`-based page-id resolution and rewrites
both the page-id preamble and the RAGGED store/load address
arithmetic. Schedule.S loses its `dense_kv_indptr`-based per-row
walker and rewrites it as a `seqlens`-bounded loop over a fixed row
stride. `topK` / `approxTopK` have their page-id-write side already
migrated ([topk.py:52-58](vortex_torch/indexer/compiler/triton_impl/topk.py#L52-L58))
but their score-input read still uses the CSR row walker and must
move with the rest of Schedule.S.

---

## 2. Migration plan

### 2.1. Step 1 — Make `ctx.vortex_attention_backend` the single source of truth

Already done; preserved here for reference.

  - The string `"flashinfer"` or `"trtllm"` lives at
    [vortex_torch/indexer/context.py:222-224](vortex_torch/indexer/context.py#L222-L224)
    and is read at codegen time. **Set it before `ctx.create()`** —
    `VortexTRTLLMBackend.__init__` does exactly that at
    [trtllm.py:285-287](vortex_torch/engine/sgl/attention_backend/trtllm.py#L285-L287).
  - Any new backend that uses block-tables must follow the same
    pattern: set
    `model_runner.server_args.vortex_attention_backend = "<name>"`
    before constructing the indexer Context, and decide whether
    `"<name>"` reuses the same layout as `"trtllm"` (preferred — same
    branch in the codegen) or introduces a third layout.

### 2.2. Step 2 — Planner writes both layouts (already done, optimisation pending)

`sglang_plan_decode_v2_trtllm` in
[planner_sglang.py](vortex_torch/indexer/planner_sglang.py)
already emits `dense_block_tables`, `sparse_block_tables`,
`dense_seqlens`, `sparse_seqlens`, `dense_kv_indptr`,
`sparse_kv_indptr`, **and** the legacy `dense_kv_indices`. The
`dense_kv_indices` write is the optimisation target — it's a
strict redundancy once every Schedule.W op resolves page ids from the
block-table.

If you add a new policy (custom `static_kv_budget` / `dynamic_kv_budget`
expression), no schema change is needed — `get_sglang_plan_decode_v2_module`
takes the policy body string and JIT-compiles a fresh module. **Do
not add fields silently**; update
`get_decode_planner_trtllm` in
[utils_sglang.py:78-113](vortex_torch/indexer/utils_sglang.py#L78-L113)
and the `Context.__slots__` in
[context.py:13-51](vortex_torch/indexer/context.py#L13-L51) at
the same time so the JIT call site, the Python wrapper, and the
context object stay in lockstep.

#### Workload scheduler

The workload scheduler is *part of* `sglang_plan_decode_v2[_trtllm]` —
it writes the `winfo_*` arrays the per-workload kernels consume:

  - `winfo_q_indices`           — which (batch, head) this workload belongs to
  - `winfo_is_first_workload_per_batch`  — gate for BATCHED stores
  - `winfo_kv_offsets`          — block-of-page offset of this workload
                                  in `dense_kv_indices` (CSR base) **or**
                                  in `dense_block_tables[row]` (trtllm base)
  - `winfo_kv_lens`             — length of this workload's KV chunk
  - `winfo_num_workloads`       — total workload count
  - `winfo_chunk_size`          — runtime knob

`winfo_kv_offsets` is the field that ties the scheduler to the page-id
layout. In CSR mode it's interpreted as `dense_kv_indices +
winfo_kv_offsets[i]`. In trtllm mode the same offset must mean
"linear position into row `winfo_q_indices[i]` of
`dense_block_tables`". The planner kernel already produces the right
linear offsets — what changes is **how the consumer dereferences
them** (Step 3 below).

If you introduce a new policy that changes how many pages a workload
spans (i.e. a new `workload_chunk_size` knob), update the assertion in
[context.py:181](vortex_torch/indexer/context.py#L181)
(`workload_chunk_size % num_blocks_per_page == 0`) and verify the
schedule still produces `num_pages_per_workload = workload_chunk_size //
num_blocks_per_page` pages per program.

### 2.3. Step 3 — Schedule.W codegen reads `dense_block_tables` in trtllm mode

This is the load-bearing change. The relevant block is in
[kernel_gen.py:253-267](vortex_torch/indexer/compiler/triton_impl/kernel_gen.py#L253-L267):

```python
if ctx.num_pages_per_workload == 1:
    page_idx_line = "page_idx_i32 = tl.load(indices + ragged_idx_i32).to(tl.int32)"
else:
    page_idx_line = (
        f"page_indices_i32 = tl.load(indices + ragged_idx_i32 + "
        f"page_idx_i32_ptr * {ctx.num_blocks_per_page}, mask=page_valid, "
        f"other=0).to(tl.int32)"
    )
```

`indices` here is bound to `ctx.dense_kv_indices` by the launcher
([kernel_gen.py:524-528](vortex_torch/indexer/compiler/triton_impl/kernel_gen.py#L524-L528)).
The migration is:

1. **In the launcher**, branch on
   `ctx.vortex_attention_backend`:
   - `"flashinfer"` → bind `indices = ctx.dense_kv_indices` (current
     behaviour).
   - `"trtllm"`     → bind two tensors instead: `block_tables =
     ctx.dense_block_tables` and `block_tables_row_stride =
     ctx.dense_block_tables.shape[1]`. Drop `indices` entirely (the
     planner can stop writing `dense_kv_indices` once this branch is
     the only consumer).

2. **In the kernel signature** (`_build_kernel_signature` at
   [kernel_gen.py:432-452](vortex_torch/indexer/compiler/triton_impl/kernel_gen.py#L432-L452)),
   add a `tl.constexpr` for the row stride and rename the parameter
   so both backends share a uniform name (`page_indices_ptr`):
   - flashinfer: `page_indices_ptr = ctx.dense_kv_indices`,
     `page_indices_row_stride : tl.constexpr = 1` (sentinel, unused
     in CSR mode).
   - trtllm:    `page_indices_ptr = ctx.dense_block_tables.view(-1)`,
     `page_indices_row_stride : tl.constexpr = max_blocks_per_seq`.

3. **In the preamble** (`generate_load_tensor_str`,
   [kernel_gen.py:253-267](vortex_torch/indexer/compiler/triton_impl/kernel_gen.py#L253-L267)),
   parameterise the address arithmetic on the layout:

   - **CSR** (today): the per-workload kernel already knows which
     workload it's on (`i`). Look up
     `ragged_idx_i32 = tl.load(winfo_kv_offsets + i)` and
     `page_idx_i32 = tl.load(indices + ragged_idx_i32)`. `ragged_idx_i32`
     is the **block-of-page offset into the flat `dense_kv_indices`
     array** for this workload's (req, kv-head) slot.
   - **trtllm** (target): the kernel additionally needs the **row
     index** in `dense_block_tables`, which is
     `winfo_q_indices[i]` (already loaded for BATCHED stores at
     [kernel_gen.py:247](vortex_torch/indexer/compiler/triton_impl/kernel_gen.py#L247)).
     With that, the page-id resolution becomes:

     ```python
     # Single-page workload
     row_i32 = tl.load(winfo_q_indices + i).to(tl.int32)
     col_i32 = tl.load(winfo_kv_offsets + i).to(tl.int32)  # block-of-page index
     # Block-table is keyed at block (not page) granularity:
     page_idx_i32 = tl.load(
         page_indices_ptr
         + row_i32 * page_indices_row_stride
         + col_i32
     ).to(tl.int32)
     ```

     ```python
     # Multi-page workload (num_pages_per_workload > 1)
     row_i32 = tl.load(winfo_q_indices + i).to(tl.int32)
     col_i32 = tl.load(winfo_kv_offsets + i).to(tl.int32)
     page_indices_i32 = tl.load(
         page_indices_ptr
         + row_i32 * page_indices_row_stride
         + col_i32
         + page_idx_i32_ptr * ctx.num_blocks_per_page,
         mask=page_valid, other=0,
     ).to(tl.int32)
     ```

   Pick **one** of these two by branching on
   `ctx.vortex_attention_backend` inside the codegen (mirror the
   pattern in `topk.py:52-58`); do not emit both into the kernel.

4. **No change** to the downstream PAGED load/store math
   ([kernel_gen.py:171-220, 282-320](vortex_torch/indexer/compiler/triton_impl/kernel_gen.py#L171-L220)) —
   once `page_idx_i32` / `page_indices_i32` resolves correctly, the
   `tensor_<tid>_ptr_row_start = page_idx_i32 * shape[1]` arithmetic
   is layout-agnostic.

### 2.4. Step 4 — Re-interpret RAGGED tensors (affects **every** Schedule.S op, and Schedule.W RAGGED load/store)

The earlier draft of this guide claimed only `topK` / `approxTopK`
needed changes on Schedule.S. **That is wrong.** The block-table
migration also redefines how a `FORMAT.RAGGED` tensor is *laid out
in memory*, and every consumer that walks a RAGGED tensor — i.e.
**every** Schedule.S codegen, plus the Schedule.W RAGGED store/load
preamble — has to be updated.

#### 2.4.1 Layout change

**CSR layout (current).** A RAGGED tensor with shape
`(N, feat0, feat1)` is stored as a flat row-packed buffer:

```
RAGGED[CSR] : [ Σ_i  rows_of_(req_i, head_i) , feat0, feat1 ]
              ─────────────  N  ─────────────
```

A row `i ∈ [0, eff_bs)` occupies the slice
`dense_kv_indptr[i] : dense_kv_indptr[i+1]` along the leading axis.
Codegen addressing is `ragged_idx_i32 = winfo_kv_offsets[i]`, and
both the Schedule.W kernel preamble
([kernel_gen.py:217,326](vortex_torch/indexer/compiler/triton_impl/kernel_gen.py#L217))
and the Schedule.S kernels (softmax/normalize/conv1d/reduce-dim0/topk/approxtopk —
they each take `ctx.dense_kv_indptr` as an argument, e.g.
[softmax.py:105](vortex_torch/indexer/compiler/triton_impl/softmax.py#L105))
read it the same way.

**Block-table layout (target).** The same tensor is interpreted as a
**dense row-padded 2D** buffer keyed by `(eff_bs, max_blocks_per_seq)`:

```
RAGGED[BT]  : [ eff_bs, max_blocks_per_seq, feat0, feat1 ]
              row=(req, head)   col=block-in-row
```

Row `i` has only `seqlens[i]` valid block-rows; the rest are padding
that consumers must mask. The allocation budget is unchanged —
`max_num_blocks == max_bs * num_kv_heads * max_blocks_per_seq` already
— so the migration is **layout reinterpretation only**, not a
reallocation.

The advantage: indexing is `row * max_blocks_per_seq + col` (constant
row stride) instead of `dense_kv_indptr[i] + col` (data-dependent
prefix sum). This matches `dense_block_tables` exactly and lets the
indexer feed `trtllm_batch_decode_with_kv_cache` without a final
gather.

#### 2.4.2 New / repurposed Context fields

Each RAGGED tensor that participates in the new layout is addressed by
**two** context fields: a row stride (constexpr at codegen time) and a
per-row seqlens vector.

| Old (CSR) | New (BT) |
|---|---|
| `ctx.dense_kv_indptr [eff_bs+1]` (prefix sum, walked once per row) | `ctx.dense_seqlens [eff_bs]` (per-row valid count) + `max_blocks_per_seq : tl.constexpr` (row stride) |
| `ctx.sparse_kv_indptr [eff_bs+1]` | `ctx.sparse_seqlens [eff_bs]` (already allocated; written by planner; mutated by topk/approxTopK) |

`dense_kv_indptr` / `sparse_kv_indptr` are **still needed** by the
workload scheduler (planner emits them) and by topk's CSR-write
optimisation hatch — do not delete them in this step.

#### 2.4.3 Per-tensor `seqlens` and `seqlens`-mutating ops

In CSR mode, `dense_kv_indptr` was a single global indptr that every
op consulted. In BT mode the "valid-count vector" is **per-RAGGED-tensor
flavour**: most ops consume `ctx.dense_seqlens`, but the *output* of
`topK` / `approxTopK` follows `ctx.sparse_seqlens` because those ops
shrink each row from `dense_seqlens[i]` to
`block_reserved_bos + topk + block_reserved_eos` (or the policy's
`static_kv_budget`).

This adds two responsibilities the CSR design didn't have:

  1. **Codegen must track which seqlens applies to each RAGGED
     tensor.** Two options:
     - **Implicit (recommended for the first cut).** Codegen assumes
       every RAGGED tensor produced *before* the topk in the dataflow
       uses `dense_seqlens`, and every RAGGED tensor produced *after*
       uses `sparse_seqlens`. Inferred from the graph by a single
       pass that flags ops downstream of a `topK` / `approxTopK`
       producer. Requires a one-line annotation on those op classes
       (e.g. `Op.mutates_seqlens = True`) so the graph walker can
       find the split.
     - **Explicit.** Add a `seqlens_ctx_name` slot to each RAGGED
       tensor (`"dense"` or `"sparse"`) at tensor-creation time, and
       have codegen render `ctx.{name}_seqlens` accordingly. More
       invasive but generalises to future ops that produce *new*
       seqlens vectors.

  2. **`topK` / `approxTopK` now have a side-effect output.** Today
     the planner writes both `dense_seqlens` and `sparse_seqlens`
     before any layer runs — the topk kernel only writes into
     `sparse_block_tables[row, bos:bos+topk]`. That stays true in
     the simple case, but if you add an op that *recomputes* the
     valid count (e.g. a "filter rows below threshold" Schedule.S
     op), it must also write a fresh seqlens vector. The cleanest
     pattern is to allocate the seqlens vector as part of the
     output tensor's metadata (Option B above) so each op owns its
     output's row counts.

#### 2.4.4 Schedule.W preamble changes (RAGGED store/load)

The Schedule.W per-workload kernel reads/writes RAGGED tensors at:

  - **Load** ([kernel_gen.py:217](vortex_torch/indexer/compiler/triton_impl/kernel_gen.py#L217)):
    `tensor_{tid}_ptr_row_start = ragged_idx_i32 * t.shape[1]`
  - **Store** ([kernel_gen.py:326](vortex_torch/indexer/compiler/triton_impl/kernel_gen.py#L326)):
    `tensor_{tid}_block_ptr = tensor_{tid}_ptr + ragged_idx_i32 * shape[1] * shape[2] + workload_ptr[:,None,None] * shape[1] * shape[2] + ...`

In BT mode `ragged_idx_i32` (today: `winfo_kv_offsets[i]`, a flat
prefix-sum offset) must be replaced by a **(row, col) pair** derived
from the workload's owner:

```python
row_i32 = tl.load(winfo_q_indices  + i).to(tl.int32)
col_i32 = tl.load(winfo_kv_offsets + i).to(tl.int32)  # block-in-row
# row stride is max_blocks_per_seq, baked in as constexpr
ragged_lin_i32 = row_i32 * max_blocks_per_seq + col_i32
```

Then every site that says `ragged_idx_i32 * shape[1] [* shape[2]]`
uses `ragged_lin_i32` instead. The planner needs to be adjusted so
`winfo_kv_offsets[i]` carries the **column** (block-in-row index),
not the flat-CSR offset, when `vortex_attention_backend == "trtllm"`.
This is the matching planner-side change to Step 2.

#### 2.4.5 Schedule.S codegen changes (per op)

All six Schedule.S codegens take `ctx.dense_kv_indptr` and use it to
slice each row in their per-row kernel:

| Op | File | Current row slice | Target row slice |
|---|---|---|---|
| `Softmax`        | [softmax.py:105](vortex_torch/indexer/compiler/triton_impl/softmax.py#L105) | `[dense_kv_indptr[i] : dense_kv_indptr[i+1])` | `row=i`, len=`dense_seqlens[i]`, stride=`max_blocks_per_seq` |
| `Normalize`      | [normalize.py:99](vortex_torch/indexer/compiler/triton_impl/normalize.py#L99) | same | same |
| `Conv1d`         | [conv1d.py:104](vortex_torch/indexer/compiler/triton_impl/conv1d.py#L104) | same | same |
| `Reduce(dim=0)`  | [reduce.py:186](vortex_torch/indexer/compiler/triton_impl/reduce.py#L186) | same | same |
| `topK`           | [topk.py:60-74](vortex_torch/indexer/compiler/triton_impl/topk.py#L60-L74) | input keyed by `dense_kv_indptr`, output keyed by `sparse_kv_indptr` | input keyed by `dense_seqlens`, output keyed by `sparse_seqlens` |
| `approxTopK`     | [topk.py:114-127](vortex_torch/indexer/compiler/triton_impl/topk.py#L114-L127) | same | same |

The mechanical change for each op:

1. **Drop** the `ctx.dense_kv_indptr` (and matching `sparse_kv_indptr`)
   argument from the launcher; add `ctx.dense_seqlens` (and
   `ctx.sparse_seqlens` if relevant) plus a `max_blocks_per_seq`
   constexpr.
2. **Rewrite the per-row kernel body** to:
   - Read `row_len = tl.load(dense_seqlens + i)` (or sparse).
   - Iterate `col in range(row_len)` (or vectorise with a
     `col < row_len` mask) using stride `max_blocks_per_seq` along
     the leading axis.
3. For `topK` / `approxTopK`, additionally **write** the output
   `sparse_seqlens[i] = block_reserved_bos + topk + block_reserved_eos`.
   The planner can keep doing this pre-emptively today (it
   already does); making the kernel write it as well is forward-
   compatible with future row-count-mutating ops.
4. Branch on `ctx.vortex_attention_backend` so the flashinfer build
   keeps the CSR codepath until the planner-side CSR write is dropped
   in Step 5.

For `topK` / `approxTopK` specifically there is also the **page-id
output** to consider (already migrated): in BT mode the topk kernel
writes selected page ids into `sparse_block_tables[row, bos:bos+topk]`
at *block* granularity, matching the planner's BOS+EOS pre-fill.
That part of `generate_topk_impl` already branches correctly on the
backend ([topk.py:52-58](vortex_torch/indexer/compiler/triton_impl/topk.py#L52-L58)) —
the new work is on the **score input**, which is itself a RAGGED
tensor and inherits the layout change above.

#### 2.4.6 Adding a new Schedule.S op

  - If it reads or writes a RAGGED tensor → it must take
    `ctx.*_seqlens` and `max_blocks_per_seq` (constexpr) and address
    the leading axis as `row * max_blocks_per_seq + col`. Mirror
    `generate_softmax_impl` after migration.
  - If it mutates the per-row count → it must own its output's
    seqlens vector (Option B in §2.4.3) and write the new lengths.
  - If it consults page ids directly from the KV cache → mirror
    `generate_topk_impl`'s backend branch ([topk.py:52-58](vortex_torch/indexer/compiler/triton_impl/topk.py#L52-L58)).

### 2.5. Step 5 — Drop the redundant CSR write

Once Step 3 has landed and every W-codegen launcher binds
`page_indices_ptr` from `ctx.dense_block_tables` in trtllm mode, the
trtllm planner's `dense_kv_indices` write becomes dead code. Remove
in this order to keep `git bisect` clean:

1. In
   [utils_sglang.py:88](vortex_torch/indexer/utils_sglang.py#L88),
   stop passing `ctx.dense_kv_indices` to
   `sglang_plan_decode_v2_trtllm`.
2. In
   [planner_sglang.py:478-727](vortex_torch/indexer/planner_sglang.py#L478-L727),
   drop the `dense_kv_indices` argument from the kernel signature and
   remove the corresponding write (the `dense_csr_output[pos] =
   block_id;` line at L551 and the surrounding indptr math). Update
   the C++ wrapper at L585 and the binding at L777 to match.
3. Optional: keep `ctx.dense_kv_indices` allocated in trtllm mode
   for one release as a compatibility hatch, then drop it from
   `Context.__slots__` and the `kv_indices_decode` allocation in
   [trtllm.py:163-181](vortex_torch/engine/sgl/attention_backend/trtllm.py#L163-L181).

---

## 3. Shape & semantics cheat sheet (trtllm layout)

### 3.1 Indexer-owned buffers



All shapes assume `eff_bs = batch * num_kv_heads` (trtllm folds each
kv-head into the batch dimension and runs the kernel with
`num_kv_heads = 1`):

| Tensor | Shape | Dtype | Granularity | Filled by |
|---|---|---|---|---|
| `dense_block_tables`  | `[eff_bs, max_blocks_per_seq]` | int32 | block | planner (fully) |
| `sparse_block_tables` | `[eff_bs, max_blocks_per_seq]` | int32 | block | planner (BOS+EOS slots) + topk kernel (middle) |
| `dense_seqlens`       | `[eff_bs]` | int32 | token count | planner |
| `sparse_seqlens`      | `[eff_bs]` | int32 | token count | planner |
| `dense_kv_indptr`     | `[eff_bs + 1]` | int32 | block | planner (still needed by workload scheduler + topk) |
| `sparse_kv_indptr`    | `[eff_bs + 1]` | int32 | block | planner (still needed by topk) |
| `kv_last_page_len`    | `[eff_bs]` | int32 | tokens-in-last-page | planner |

`max_blocks_per_seq = ceil(context_len / block_size)` — note **block**,
not page. That's because vortex's logical unit is the block; trtllm's
`block_tables` happens to also be at block granularity in this build
(allocated in
[trtllm.py:244-258](vortex_torch/engine/sgl/attention_backend/trtllm.py#L244-L258)).
`page_size / block_size = num_blocks_per_page` blocks share one page
in the KV pool; the codegen at
[kernel_gen.py:259-263](vortex_torch/indexer/compiler/triton_impl/kernel_gen.py#L259-L263)
already multiplies by `num_blocks_per_page` to advance the
per-workload pointer — that line is **layout-agnostic** and survives
the migration.

The sparse path's `seq_lens` is dynamic — it's recomputed each layer
by the topk kernel via the `dense_kv_indptr` row counts and reserved
BOS/EOS slots, and the trtllm decode call reads it from
`self.forward_metadata.seq_lens[1]` ([trtllm.py:651](vortex_torch/engine/sgl/attention_backend/trtllm.py#L651)).

### 3.2 RAGGED tensor layout (codegen-owned)

A `FORMAT.RAGGED` tensor with logical shape `(N, feat0, feat1)`:

| | CSR (today) | Block-table (target) |
|---|---|---|
| Storage shape | `[N, feat0, feat1]` flat | `[eff_bs, max_blocks_per_seq, feat0, feat1]` row-padded |
| Row `i` slice | `[indptr[i] : indptr[i+1])` along axis 0 | `[i, :seqlens[i])` along axis 1 |
| Leading-axis address | `ragged_idx_i32 = winfo_kv_offsets[i]` (flat offset) | `row * max_blocks_per_seq + col` (constant stride) |
| Valid count for op | `indptr[i+1] - indptr[i]` | `seqlens[i]` |
| Producer mutates count? | rewrite both indptrs | write new `seqlens` (per output) |
| Allocation budget | `max_num_blocks * feat0 * feat1` | identical (`max_num_blocks == max_bs * num_kv_heads * max_blocks_per_seq`) |

Schedule.W RAGGED store/load address arithmetic
([kernel_gen.py:217,326](vortex_torch/indexer/compiler/triton_impl/kernel_gen.py#L217)):

  - **CSR:**  `tensor_<tid>_ptr_row_start = ragged_idx_i32 * shape[1]`
  - **BT:**   `tensor_<tid>_ptr_row_start = (row * max_blocks_per_seq + col) * shape[1]`
              where `row = winfo_q_indices[i]` and
              `col = winfo_kv_offsets[i]` (column-only in BT mode).

Schedule.S RAGGED row walk:

  - **CSR:**  `for k in range(indptr[i], indptr[i+1]): ...`
  - **BT:**   `row_base = i * max_blocks_per_seq;
              for k in range(seqlens[i]): ... row_base + k ...`

---

## 4. Validation checklist

When you finish a step, validate in this order; later checks assume
earlier ones pass.

1. **Pre-flight (CPU-only)** — `check_engine_config(...)` on any
   submission JSON that pins
   `"vortex_attention_backend": "trtllm"`. Catches `Save` without
   `disable_radix_cache: true`, missing `CFill`, etc.
2. **RULER ≥ 97% on the BT path — verified by RULER, not AIME24.**
   Because this migration is a layout reinterpretation (not an
   algorithm change — see the invariant at the top of this doc),
   downstream task accuracy must hold even though raw outputs will
   not be bit-identical (fp32 accumulation reorder + MMA stride
   sensitivity). Use `algorithm_scientist/run_ruler.py` to
   verify: it (a) is deterministic at the protocol level
   (substring-match scoring over a fixed
   `examples/validation.jsonl`), (b) finishes in minutes per
   variant, and (c) the ≥ 0.85 gate doubles as a
   structurally-broken-attention detector. AIME24 is for final
   `mean@16`/throughput on the Pareto frontier and has enough
   sampling noise to hide a refactor regression — do not use it
   for layout-equivalence verification.

   ```bash
   for backend in flashinfer trtllm; do
     # Build a temp config with the backend flipped; reuse the
     # same submission otherwise. (Either edit the JSON in place
     # and revert, or use jq to pipe a modified copy.)
     CUDA_VISIBLE_DEVICES=0 python algorithm_scientist/run_ruler.py \
         --config submissions/<tag>/<name>__${backend}.json
   done
   ```

   Results land in `summary_ruler_submissions/<tag>/<stem>/latest.json`.
   Pass criterion:
     - BT-path RULER ≥ **0.97** (97%) for any submission whose
       flashinfer/CSR baseline already passes the standard 0.85
       gate. Equivalently, the BT/CSR ratio should be ≥ 0.97 in
       absolute terms.
     - BT-path RULER ≥ 0.85 in absolute terms regardless.
   Below 0.97 (or any large per-example flip rate) means a
   layout bug, almost always a stride bug in `page_idx_i32`
   resolution, a `(row, col)` swap in the BT-mode RAGGED address
   arithmetic, or a `seqlens` vs `indptr` mix-up — not numerical
   drift. Below 0.85 absolute means attention is structurally
   broken regardless of the migration.
3. **Single-page vs multi-page** — set `workload_chunk_size` so the
   per-workload kernel runs at `num_pages_per_workload == 1` first
   (the simpler branch), then bump it to force the multi-page
   branch. Both must produce identical accuracy.
4. **CUDA-graph capture** — `init_forward_metadata_capture_cuda_graph`
   and `..._replay_cuda_graph` call the planner from inside the
   capture; any new tensor you add to `ctx` must be a fixed
   pre-allocated buffer (capture rewrites pointers, not shapes).
5. **`CUDA_LAUNCH_BLOCKING=1`** while iterating. IMAs that come from
   the workload-kernel preamble are almost always either an
   off-by-one in `winfo_kv_offsets` (CSR-relative offset used as
   block-table column) or a row index loaded with the wrong dtype.

---

## 5. Anti-patterns

  - **Don't pass `dense_block_tables` directly into a Schedule.W
    kernel as a 2D tensor.** Triton wants pointer + constexpr
    strides; emit `ctx.dense_block_tables.view(-1)` and supply
    `max_blocks_per_seq` as a constexpr. This keeps the kernel
    signature uniform with the CSR path.
  - **Don't add a third layout enum.** Future backends that need a
    block-table-shaped buffer should reuse the trtllm branch (set
    `vortex_attention_backend = "trtllm"`). Add a new value only when
    the row stride or the per-row semantics genuinely differ.
  - **Don't read `ctx.dense_kv_indices` in trtllm mode** from any new
    code path. The whole point of the migration is that it can be
    dropped.
  - **Don't conflate page granularity and block granularity.** The
    `block_tables` row is indexed in *blocks*; the KV cache pool
    pages are `num_blocks_per_page` blocks each. Mixing the two is
    the most common cause of off-by-`num_blocks_per_page` IMAs.
  - **Don't treat RAGGED tensors as flat 1D-along-axis-0 buffers in
    BT mode.** They are `[eff_bs, max_blocks_per_seq, feat0, feat1]`
    row-padded — the row stride is `max_blocks_per_seq`, not the
    per-row valid count. Padding slots between `seqlens[i]` and
    `max_blocks_per_seq` are *undefined*; consumers must mask via
    `seqlens[i]` and producers must not assume they're zero.
  - **Don't reuse `winfo_kv_offsets` cross-mode.** In CSR mode it is
    a flat prefix-sum offset into `dense_kv_indices`. In BT mode it
    is a per-workload **column** index into the workload's row of
    `dense_block_tables`. The planner emits one or the other based on
    `vortex_attention_backend`; consumer codegen must match.
  - **Don't recompute `seqlens` for an output tensor outside the
    producer op.** If a future op shrinks/grows the per-row valid
    count, the codegen for that op writes the new `seqlens` vector
    as part of its kernel. Downstream consumers read it; they never
    re-derive it from `dense_seqlens`.

---

## 6. References

  - Planner (CSR + trtllm in one module):
    [vortex_torch/indexer/planner_sglang.py](vortex_torch/indexer/planner_sglang.py)
  - Planner Python wrappers:
    [vortex_torch/indexer/utils_sglang.py](vortex_torch/indexer/utils_sglang.py)
  - Context (slot list, defaults, `vortex_attention_backend`):
    [vortex_torch/indexer/context.py](vortex_torch/indexer/context.py)
  - Schedule enum + registry:
    [vortex_torch/utils.py:13-15](vortex_torch/utils.py#L13-L15),
    [vortex_torch/indexer/compiler/triton_impl/register.py](vortex_torch/indexer/compiler/triton_impl/register.py)
  - Schedule.W codegen (the migration target):
    [vortex_torch/indexer/compiler/triton_impl/kernel_gen.py](vortex_torch/indexer/compiler/triton_impl/kernel_gen.py)
    (`generate_load_tensor_str`, `_build_kernel_signature`,
    `_build_launcher_args`)
  - Schedule.W per-op codegens that reuse the preamble (no per-op
    change required as long as the preamble migrates correctly):
    [save_load.py](vortex_torch/indexer/compiler/triton_impl/save_load.py),
    [gemm.py](vortex_torch/indexer/compiler/triton_impl/gemm.py),
    [reduce.py](vortex_torch/indexer/compiler/triton_impl/reduce.py),
    [elementwise.py](vortex_torch/indexer/compiler/triton_impl/elementwise.py),
    [elementwise_binary.py](vortex_torch/indexer/compiler/triton_impl/elementwise_binary.py),
    [transpose.py](vortex_torch/indexer/compiler/triton_impl/transpose.py),
    [mask.py](vortex_torch/indexer/compiler/triton_impl/mask.py),
    [kron.py](vortex_torch/indexer/compiler/triton_impl/kron.py),
    [reshape.py](vortex_torch/indexer/compiler/triton_impl/reshape.py)
  - Schedule.S codegen (already block-table-aware):
    [topk.py](vortex_torch/indexer/compiler/triton_impl/topk.py)
  - trtllm backend (binds context + invokes attention kernel):
    [vortex_torch/engine/sgl/attention_backend/trtllm.py](vortex_torch/engine/sgl/attention_backend/trtllm.py)
