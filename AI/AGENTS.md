# AGENT.md — Instructions for AI agents writing sparse attention

You are tasked with writing a new **sparse-attention flow** for
`vortex_torch`. The framework JIT-compiles your Python description
into Triton kernels and plugs them into sglang's decode loop.

Your deliverable is **two files placed in `submissions/`**:

- `submissions/<your_name>.py`   — a `vFlow` subclass.
- `submissions/<your_name>.json` — an engine config that points at it.

---

## Objective

> **Maximise decoding throughput (tokens/sec) while keeping `mean@16`
> at or above a minimum acceptable floor.**

`mean@16` is a *quality gate*, not something to maximise. Once the
flow passes the gate, every extra design choice should be made in the
direction of **more throughput**: fewer / cheaper cache-side ops,
fewer intermediate cache fields, tighter `vortex_topk_val` /
`vortex_topk_ratio` budgets, smarter layer-skip patterns, aggressive
`kv_cache_dtype` (fp8). If two variants both clear the floor, pick the
faster one — not the more accurate one.

Concretely, each benchmark run gives you two headline numbers
(see §5b):

- `mean@16` — must be `≥ MIN_MEAN_AT_16` (your quality floor).
- `throughput` — maximise this subject to the floor above.

---

## 0. Read the tutorials first

Before writing any code, read the files in [`AI/tutorials/`](tutorials/)
**in this order**:

1. [`overview.md`](tutorials/overview.md) — 5-minute map of the whole
   framework and what you're being asked to write.
2. [`program_create_cache.md`](tutorials/program_create_cache.md) —
   how to declare the auxiliary cache fields your flow needs.
3. [`program_forward_cache.md`](tutorials/program_forward_cache.md) —
   how to write the per-block summary updater.
4. [`program_forward_indexer.md`](tutorials/program_forward_indexer.md) —
   how to write the per-decode-step page router.
5. [`cache_op.md`](tutorials/cache_op.md) — math reference for every
   op you can use in `forward_cache`.
6. [`indexer_op.md`](tutorials/indexer_op.md) — math reference for
   every op you can use in `forward_indexer`.

You do **not** need to read `developer_guides` — that's for framework
developers. All six files above stay entirely in the user's mental
model.

If a detail isn't covered in those six files, read the reference
implementations in [`../vortex_torch/flow/algorithms.py`](../vortex_torch/flow/algorithms.py).

---

## 1. The contract

You are writing a subclass of `vFlow` with exactly three methods:

```python
from vortex_torch.flow   import vFlow, register
from vortex_torch.indexer import Mean, GeMM, topK, ...   # whichever ops you need
from vortex_torch.cache   import Mean as CMean, ...

@register("<your_module_name>")
class YourFlow(vFlow):

    def __init__(self):
        super().__init__()
        # declare every op instance you'll use — each CALL SITE needs its own

    def create_cache(self, block_size, head_dim):
        return { "<name>": (D_0, D_1), ... }

    def forward_cache(self, cache, loc, ctx):
        # build summaries from cache["k"] / cache["v"] and write them
        # into the fields you declared above

    def forward_indexer(self, q, o, cache, ctx):
        # build a per-page score [S, 1, 1] and hand it to topK
```

The user's mental model (used throughout the tutorials):

- Pretend the system runs with `batch_size = 1`, one sequence.
- `q` has shape `[1, H_q, D]`.
- Every `cache["<name>"]` is `[S, D_0, D_1]` on the indexer side,
  `[1, D_0, D_1]` on the cache side.
- `cache["k"]` and `cache["v"]` are always there — **don't declare
  them** in `create_cache`.
- The framework replays your one-sequence program across the real
  batch and kv-head axes. You never write batching, paging, or
  Triton.

---

## 2. Hard rules — the framework will reject violations

1. **No native torch ops inside the three methods.** Every tensor
   (`q`, `o`, `cache[...]`, any intermediate) must go through
   `vortex_torch.indexer.*` or `vortex_torch.cache.*` ops. `.view`,
   `.contiguous`, elementwise torch, `.sum(dim=...)`, *etc.* will not
   compile.
2. **Each op instance is for one call site.** If `forward_indexer`
   needs two `Multiply` calls, declare `self.mul_a = Multiply()` and
   `self.mul_b = Multiply()` — don't share.
3. **Don't declare `"k"` or `"v"` in `create_cache`.** The framework
   hard-asserts against that and auto-provides both.
4. **`forward_indexer` must end in `topK(score, o, ctx=ctx)` where
   `score.shape == [S, 1, 1]`.** Fold any stray `H_q` / `D` axes with
   `Mean` / `Max` / `Sum` before calling `topK`.
5. **Every declared cache field needs a writer and a reader.** A
   field nobody writes stays silently at stale bytes; a field nobody
   reads is wasted bandwidth.
6. **Reductions on the cache side only support `dim ∈ {1, 2}`.**
   Cross-block summaries (`dim=0`) belong on the indexer side.
7. **If the indexer accumulates into a field via `Load`/`Save`, the
   cache side must `CFill(0.0)` it at block completion** — otherwise
   the first `Load` on a freshly-allocated block reads uninitialised
   memory. See the running-average example in
   `program_forward_indexer.md §9` and `program_forward_cache.md §6b`.

---

## 3. Canonical skeleton

Copy-paste this into `submissions/<your_name>.py` and fill it in:

```python
import torch
from typing import Dict

from vortex_torch.flow    import vFlow, register
from vortex_torch.abs     import ContextBase
from vortex_torch.indexer import (
    topK, Mean, Max, Min, Sum, L2Norm,
    GeMM, Multiply, Add, Maximum, Minimum,
    Softmax, Normalize,
    Relu, Sigmoid, Silu, Add_Mul, Abs, Log, Exp,
    Save, Load, MaskSlice,
    WhereEqual, WhereNotEqual, WhereGreater,
    WhereGreaterEqual, WhereLess, WhereLessEqual,
)
from vortex_torch.cache import (
    Mean as CMean, Max as CMax, Min as CMin, L2Norm as CL2Norm,
    GeMM as CGeMM,
    Multiply as CMultiply, Add as CAdd, Maximum as CMaximum, Minimum as CMinimum,
    Relu as CRelu, Sigmoid as CSigmoid, Silu as CSilu,
    Add_Mul as CAdd_Mul, Abs as CAbs, Log as CLog, Exp as CExp,
    Fill as CFill, MaskSlice as CMaskSlice,
)


@register("<your_module_name>_cls")
class YourFlowCls(vFlow):
    def __init__(self):
        super().__init__()
        # Indexer-side ops (one instance per call site)
        ...

        # Cache-side ops (one instance per call site)
        ...

    def create_cache(self, block_size: int, head_dim: int):
        return {
            # Example: "centroids": (1, head_dim),
            # Do NOT include "k" or "v".
        }

    def forward_cache(
        self,
        cache: Dict[str, torch.Tensor],
        loc: torch.Tensor,
        ctx: ContextBase,
    ):
        # For every field declared above that depends on cache["k"] or
        # cache["v"], compute and persist it here. If a field is
        # accumulated by forward_indexer via Load/Save, CFill(0.0) it
        # here at block completion.
        ...

    def forward_indexer(
        self,
        q: torch.Tensor,
        o: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        ctx: ContextBase,
    ):
        # Build a per-page score of shape [S, 1, 1] and hand it to topK.
        ...
        self.output_func(score, o, ctx=ctx)
```

And the matching config:

```json
{
  "vortex_module_path":         "submissions/<your_name>.py",
  "vortex_module_name":         "<your_module_name>_cls",
  "vortex_block_size":          16,
  "vortex_workload_chunk_size": 32,
  "vortex_topk_val":            29,
  "vortex_topk_ratio":          0.0625,
  "vortex_block_reserved_bos":  1,
  "vortex_block_reserved_eos":  2,
  "vortex_layers_skip":         [0],
  "vortex_dtype":               "bfloat16",
  "kv_cache_dtype":             "auto",
  "mem_fraction_static":        0.8
}
```

`vortex_module_name` **must exactly match** the string passed to
`@register(...)` in your Python file.

**Every field in this JSON is a tuning knob you may change.** Pick
values that suit your flow. The ones that need special care:

| field | allowed / recommended values |
|---|---|
| `vortex_block_size` | positive power of 2 (e.g. 8, 16, 32). Smaller = finer sparsity granularity, larger = less cache-summary overhead. |
| `vortex_workload_chunk_size` | positive power of 2 (e.g. 16, 32, 64). |
| `vortex_topk_val` / `vortex_topk_ratio` | per-sequence sparse-page budget (see "Budget semantics" below). |
| `vortex_block_reserved_bos` / `_eos` | ints ≥ 1. Reserved first / last blocks that are always selected (see "BOS/EOS semantics" below). |
| `vortex_layers_skip` | list of layer ids that bypass sparse attention (run dense). See "Layer-skip patterns" below. |
| `vortex_dtype` | dtype for **intermediate** tensors. **Recommended: `"bfloat16"`.** Other accepted values: `"float16"`, `"float32"`, `"fp8_e5m2"`, `"fp8_e4m3"` — use only if you have a specific reason; bf16 is the tested default. |
| `kv_cache_dtype` | dtype for the **K/V cache itself**. Choose from: `"auto"` (resolves to bfloat16), `"fp8_e4m3"`, or `"fp8_e5m2"`. Using fp8 halves cache memory at the cost of numerical precision; bf16 via `"auto"` is the safe default. |
| `mem_fraction_static` | fraction of GPU memory sglang reserves for KV cache + model weights. Float in `[0.5, 0.95]` (out-of-range values are rejected at engine-boot). **Default 0.8.** Higher values usually raise throughput by enabling larger decode batches, but raise the risk of CUDA OOM mid-run. **Sweet spot 0.8-0.9.** Try `0.85` first; if it runs cleanly, push toward `0.9` / `0.95`. If you OOM, drop back by 0.05. |

### Budget semantics (`vortex_topk_val`, `vortex_topk_ratio`)

At runtime, the number of pages the attention kernel actually attends to
for each sequence is:

```
selected = min(
    num_blocks_in_seq,
    max(vortex_topk_val + vortex_block_reserved_bos + vortex_block_reserved_eos,
        vortex_topk_ratio * num_blocks_in_seq)
)
```

In plain English:

- The **static floor** is `topk_val + bos + eos`: a fixed minimum
  regardless of sequence length. You always get at least this many
  blocks.
- The **dynamic floor** is `topk_ratio * num_blocks_in_seq`: scales
  with the sequence. Use this to keep the sparsity *fraction* stable
  as sequences grow.
- The engine picks the larger of the two — you get both "at least
  `topk_val` blocks" *and* "at least this fraction of the sequence",
  whichever is more at any given moment.
- The final `min` clamps to the number of blocks actually in the
  sequence — there's no point selecting more blocks than exist.

So `topk_val` dominates on short sequences and `topk_ratio` dominates
on long ones. Typical starting values: `topk_val = 29-253`,
`topk_ratio = 0.0625` (1/16) to `0.25`.

### BOS/EOS semantics (`vortex_block_reserved_bos`, `vortex_block_reserved_eos`)

These are "always-on" blocks that the indexer is *not* allowed to
drop:

- `vortex_block_reserved_bos = N` → the **first N blocks** of every
  sequence are always in the selected set, no matter what score your
  `forward_indexer` produces for them. These blocks hold the system
  prompt / BOS tokens and are usually globally relevant.
- `vortex_block_reserved_eos = N` → the **last N blocks** of every
  sequence are always selected. These hold the most recent tokens,
  which typical generation kernels rely on.

The framework handles these two automatically — your
`forward_indexer` does **not** need to emit large scores for the
first / last blocks; the engine layers the BOS/EOS blocks on top of
your top-k selection afterwards. That's why the static floor in the
budget formula adds `bos + eos` to `topk_val`: the count reflects
"learned top-k" plus "fixed reservations".

Sensible values: `bos = 1-2`, `eos = 1-4`.

### Layer-skip patterns (`vortex_layers_skip`)

`vortex_layers_skip` is a list of layer ids that run **dense**
attention (no cache-side update, no page routing). Common patterns:

- **`[0]`** — skip only the first layer. Maximum sparsity, maximum
  speed; works for many models since layer 0 is often most sensitive
  to full context.
- **`[0, 4, 8, 12, ...]`** or **`[0, 8, 16, ...]`** — **interleaved**
  skip. Every k-th layer runs dense, the rest run sparse. This trades
  some throughput for better quality: the dense layers act as
  "periodic refreshers" of full-context information that downstream
  sparse layers can still benefit from. Use when a pure `[0]` skip
  loses too much accuracy on your task.
- **`[0, 1]`** — skip the first two layers. Moderate-sparsity baseline.

The stride and count are entirely your choice. More skipped layers =
more throughput cost but typically better quality. Start with `[0]`
and widen only if quality is unacceptable.

---

## 4. Reference flows to learn from

Six worked examples live in
[`../vortex_torch/flow/algorithms.py`](../vortex_torch/flow/algorithms.py).
Read them in increasing complexity:

| flow | what it demonstrates |
|---|---|
| `block_sparse_attention` | simplest; centroid per block, score by `q_mean · centroid` |
| `gqa_block_sparse_attention` | adds `Softmax` over pages, multi-head aggregation |
| `gqa_quest_sparse_attention` | QUEST envelope bound (max/min of keys per block) |
| `masked_quest_sparse_attention` | QUEST + feature-axis `MaskSlice` |
| `centered_block_sparse_attention` | per-sequence mean subtraction via `Reduce(dim=0)` |
| `running_avg_block_sparse` | persistent across-step state via `Save` / `Load` + `CFill(0.0)` |

The existing submission
[`submissions/example_block_sparse_attention.py`](../submissions/example_block_sparse_attention.py)
plus its JSON is a **complete working pair** — use it as the template
for your file layout.

---

## 5. Before you submit — verify compilation

After writing your flow, **run the pre-flight check**. It validates
the JSON, loads your module, and runs a small compile sweep. It
catches most shape/dispatch mistakes before sglang ever touches a
GPU:

```python
from vortex_torch.engine.sgl import check_engine_config
check_engine_config("submissions/<your_name>.json")
```

If this returns without raising, your flow is ready for the engine.
If it raises, fix whatever error it reports and re-run. Common
failures and what they mean:

| error substring | what's wrong |
|---|---|
| `must not declare 'k' key` | you put `"k"` in `create_cache` |
| `must be a positive power of 2` | `vortex_block_size` or `vortex_workload_chunk_size` isn't a power of 2 |
| `does not contain @register` | `vortex_module_name` doesn't match the string in `@register(...)` |
| `failed to build vFlow` | typically a shape mismatch or a wrong `_impl_map` key — read the traceback |
| `vortex_module_path ... not found` | file path in the JSON is wrong |

---

## 5b. Run the benchmark and read the summary

Once `check_engine_config` passes, boot the engine and run the fixed
AIME24 protocol:

```bash
python algorithm_scientist/run_submission_aime24.py --config submissions/<your_name>.json
```

Everything else is hard-coded inside `algorithm_scientist/run_submission_aime24.py`
(16 trials, `Qwen/Qwen3-4B`, single GPU, 4096-token input cap, 16384
max new tokens, `examples/aime24.jsonl`). The only thing you change
between runs is your flow's JSON.

The script prints a summary to stdout and writes it into a
**per-submission subfolder** under `summary_submissions/`:

```
summary_submissions/
└── <config_stem>/
    ├── 2026-04-24_13-22-05__0ca8893beb0b.json   # full summary + embedded .py/.json
    ├── 2026-04-24_14-07-51__17b9a4f2d310.json
    ├── latest.json                              # symlink → newest run
    └── INDEX.jsonl                              # one row per run, grep-able
```

The per-run filename encodes two things:

- `<timestamp>` — sortable, lexicographically newest = most recent.
- `<content_hash>` — first 12 hex chars of `sha256(config.json || module.py)`.
  Two runs of the **exact same code** share a hash (visible re-run);
  any edit produces a new hash. This is the easiest way to tell
  "is this the same submission I benchmarked yesterday?" without
  opening the file.

The JSON itself also embeds the full `.py` and `.json` contents at
run time under `submission_py` / `submission_json`, so any summary
is fully self-contained — you can reproduce a result from the
summary alone.

`INDEX.jsonl` is append-only and contains one compact row per run
(headline metrics + hash + Slurm job id). Use it to compare runs at
a glance:

```bash
cat summary_submissions/<name>/INDEX.jsonl | jq -c '{finished_at, content_hash, "mean@16", throughput}'
```

Example contents of one run JSON:

```json
{
    "mean@16": 0.09375,
    "pass@16": 0.3333333333333333,
    "total_example": 480,
    "e2e_time": 505.16787934303284,
    "total_tokens": 6042801,
    "throughput": 11961.966005951565,
    "args": { ... },
    "content_hash": "0ca8893beb0b",
    "finished_at": "2026-04-24_13-22-05",
    "slurm_job_id": "57321",
    "submission_json": "{ ... full config.json text ... }",
    "submission_py":   "{ ... full module.py text ... }"
}
```

What each field means:

| field | meaning |
|---|---|
| `mean@16` | accuracy averaged across every single (question, trial) pair. The headline quality number — higher is better. |
| `pass@16` | per-question best-of-16. For each AIME24 question, take the max score across the 16 trials, then average across questions. Higher than `mean@16` when the model is good some fraction of the time but not always. |
| `total_example` | `trials × num_questions` = `16 × 30` = **480** for the fixed protocol. If this isn't 480, something's wrong with the dataset or trials count. |
| `e2e_time` | longest end-to-end latency across all 480 generations (seconds). Driven by the slowest / longest generation. |
| `total_tokens` | total completion tokens generated (summed across all 480). |
| `throughput` | `total_tokens / e2e_time` (tokens/sec). The headline *speed* number — higher is better. |
| `args` | echo of all runtime settings so a summary is self-describing. |
| `cuda_visible_devices` | value of `$CUDA_VISIBLE_DEVICES` at run time — useful when batched 8 submissions share a node. |
| `content_hash` | first 12 hex chars of `sha256(config.json \|\| module.py)`. Same code → same hash; any edit → new hash. |
| `finished_at` | `YYYY-MM-DD_HH-MM-SS` — local wall-clock time the run finished. |
| `slurm_job_id` | populated when launched via `sbatch algorithm_scientist/run_submission.slurm` — ties the summary back to the Slurm logs under `logs/submission/vortex_submission_<JOBID>.{out,err}`. |
| `submission_json` | full text of the config JSON at run time. |
| `submission_py` | full text of the module .py at run time. |

### What to iterate on

Remember: **throughput is the thing being optimised. `mean@16` is a
floor, not a ceiling.** Diagnose each run with that asymmetry in
mind:

- **`mean@16` below the floor** → the quality gate has failed. Fix
  this first, even at a throughput cost. Tighten the scoring logic in
  `forward_indexer`, *widen* `vortex_layers_skip` (e.g. `[0, 1]` or
  interleaved `[0, 4, 8, ...]`), *raise* `vortex_topk_val` /
  `vortex_topk_ratio`, back off an aggressive `kv_cache_dtype` (fp8
  → auto). Only once the floor is cleared should you think about
  speed.
- **`mean@16` clears the floor, `throughput` merely OK** → this is the
  normal case and the main optimisation target. Cut cache/indexer
  work that isn't pulling its weight: drop intermediate cache fields,
  replace `GeMM` with cheaper reductions where possible, *shrink*
  `vortex_topk_val`, try fp8 `kv_cache_dtype`, or narrow
  `vortex_layers_skip` back down.
- **`mean@16` clears the floor by a lot** → you are probably leaving
  throughput on the table. Push the sparsity knobs harder
  (lower `vortex_topk_val` / `vortex_topk_ratio`, shrink
  `vortex_layers_skip`, try `fp8_e4m3` / `fp8_e5m2` for
  `kv_cache_dtype`) until `mean@16` approaches the floor.
- **`mean@16` nontrivial but `pass@16` much higher** → the flow is
  inconsistent but not broken. Noise, not a design bug; prioritise
  throughput work over chasing this.
- **`total_example != 480`** → something is wrong with the run
  itself; investigate before trusting any other number. Usually a
  path error (`--config` pointed at the wrong file) or the engine
  crashed mid-run.

A submission is considered "good" when `mean@16` clears the agreed
quality floor **and** `throughput` is meaningfully higher than the
`example_block_sparse_attention` baseline. When in doubt between two
variants that both clear the floor, pick the one with higher
`throughput` — that is the entire point of the flow.

---

## 5c. Batched benchmarking — 8 variants per batch is mandatory

Each node of the cluster has **8 GPUs**, and the submission workflow
**requires every batch to fill all 8 slots**. Single-variant runs are
not part of the protocol — if you have only one core idea, fill the
other seven slots with orthogonal knob sweeps (different
`vortex_topk_val`, different `kv_cache_dtype`, different
`vortex_layers_skip`, different `vortex_block_size`, etc.). Submit
all 8 in one batched Slurm job; the runner pins each to its own
`CUDA_VISIBLE_DEVICES`.

> The single-variant `algorithm_scientist/run_submission.slurm` is
> retained for human debugging only. Automated agents and the
> benchmark protocol use **only** `run_submission_batch.slurm`.

Use [`algorithm_scientist/run_submission_batch.slurm`](../algorithm_scientist/run_submission_batch.slurm):

```bash
sbatch algorithm_scientist/run_submission_batch.slurm \
    submissions/v1.json submissions/v2.json submissions/v3.json \
    submissions/v4.json submissions/v5.json submissions/v6.json \
    submissions/v7.json submissions/v8.json
```

What the slurm file does:

1. Requests a **whole 8-GPU node** (`#SBATCH --gres=gpu:8`,
   `--cpus-per-task=128`).
2. Loops over the positional config args (up to 8) and for each
   launches `python algorithm_scientist/run_submission_aime24.py --config <cfg>`
   in the background inside a subshell that sets
   `export CUDA_VISIBLE_DEVICES=<i>` for `i = 0 … 7`.
3. Redirects each child's stdout/stderr into its own log file under
   `logs/submission/batch_<JOBID>/gpu<i>_<stem>.{out,err}`.
4. `wait`s for every child, collects exit codes, and returns
   non-zero if any child failed (but never short-circuits the
   waits — every child runs to completion).

The runner (`run_submission_aime24.py`) automatically records
`cuda_visible_devices` into each summary JSON, so a quick
`cat summary_submissions/v*/INDEX.jsonl | jq -c` tells you which
variant landed on which GPU in the batch.

### In-flight ceiling — at most 24 experiments simultaneously

Across all in-flight batches you are allowed **24 experiments**, i.e.
**3 concurrent batches** of 8. Before every `sbatch`, check the
queue:

```bash
squeue -u $USER -h -o '%i %j %T'
```

If three rows with `vortex_submission_batch` are already
PENDING/RUNNING, do not submit — spend the time on §5d below until a
slot frees up.

---

## 5d. The "while-you-wait" protocol

A single batch takes **8+ hours** of wall-clock time. Idle is not an
acceptable answer. While Slurm is grinding, do exactly one of the
following on each polling cycle, spreading attention across all
three over the wait:

1. **Deepen understanding.** In priority order:
   `AI/tutorials/` → `AI/developer_guides/` →
   `vortex_torch/flow/algorithms.py` →
   `vortex_torch/{indexer,cache}/*` → `csrc/`.
   After each file, append one bullet to
   [`algorithm_scientist/memory.md`](../algorithm_scientist/memory.md)
   §7 *Reading log* with the single most useful insight.
2. **Prepare the next batch.** If fewer than 3 batches are in
   flight, design 8 more orthogonal variants (different theme — not
   8 more copies of the running theme), pre-flight all 8, submit
   via `run_submission_batch.slurm`, and add a row to memory.md §1.
3. **Analyse completed batches.** When `sacct` shows a batch's
   state is terminal, read all 8 of its `latest.json` files,
   populate a §2 sub-section in memory.md (results table + 1-3
   sentence takeaway), remove the batch's row from §1, and update
   §3 (hypotheses) / §4 (anti-patterns) / §5 (winners) as the
   evidence lands.

---

## 5e. memory.md — your persistent notebook

[`algorithm_scientist/memory.md`](../algorithm_scientist/memory.md)
is the single source of truth for what's running, what's done, and
what you've learned. It survives across sessions; conversation
context does not. Sections:

| § | What goes here |
|---|---|
| §1 In-flight batches | one row per active batch (≤ 3 rows ever) |
| §2 Completed batches | results table + takeaway per batch, oldest first |
| §3 Design hypotheses | open hypotheses + accumulating evidence + verdict |
| §4 Anti-patterns | one-liners of things that didn't work, so future-you doesn't retry |
| §5 Patterns that worked | confirmed winners worth carrying forward |
| §6 Open questions | the agent's own backlog — pick from the top when a slot opens |
| §7 Reading log | timestamped notes from tutorials / dev guides / source |
| §8 Session notes | per-session freeform append-only summary |

**Read at start, update before stop.** Every batch submission and
every batch completion mutates §1 and §2.

---

## 6. Workflow checklist

- [ ] Read the six tutorials in `AI/tutorials/` in order.
- [ ] Decide what per-block summaries your flow needs (this becomes
      `create_cache`).
- [ ] Write `forward_cache` — compute each summary from
      `cache["k"]` / `cache["v"]`; `CFill(0.0)` any field that
      `forward_indexer` will accumulate into.
- [ ] Write `forward_indexer` — build a `[S, 1, 1]` score and pass it
      to `topK`.
- [ ] Save the flow to `submissions/<your_name>.py` with a unique
      `@register(...)` name.
- [ ] Save the config to `submissions/<your_name>.json`.
- [ ] Run `check_engine_config("submissions/<your_name>.json")`. Fix
      errors and re-run until it passes.
- [ ] Run the benchmark:
      `python algorithm_scientist/run_submission_aime24.py --config submissions/<your_name>.json`.
      Read the `mean@16` / `pass@16` / `throughput` fields of the saved
      summary JSON.
- [ ] Confirm `mean@16` clears the quality floor. If not, iterate on
      accuracy (see §5b) before anything else.
- [ ] Once the floor is cleared, iterate to **push throughput up** —
      tighten sparsity, drop unused cache fields, try fp8
      `kv_cache_dtype` — while keeping `mean@16` on the right side of
      the floor.
- [ ] Done — sglang can now boot your flow by pointing at the JSON.

---

## 7. Absolute-minimum rules (if you remember nothing else)

1. Two files: `submissions/<your_name>.py` and `submissions/<your_name>.json`.
2. `@register("<name>")` in the Python file, same `<name>` in the JSON's
   `vortex_module_name`.
3. `create_cache` returns `{name: (D_0, D_1)}`. Never include `"k"` or
   `"v"`.
4. Every op — cache side or indexer side — is from `vortex_torch.cache`
   or `vortex_torch.indexer`. No native torch ops.
5. `forward_indexer` ends in `topK(score, o, ctx=ctx)` with
   `score.shape == [S, 1, 1]`.
6. If any cache field is read+written across decode steps via
   `Load`/`Save`, zero-initialise it in `forward_cache` with
   `CFill(0.0)`.
7. Run `check_engine_config(...)` before declaring done.
8. Run `algorithm_scientist/run_submission_aime24.py --config <your JSON>` to get
   the `mean@16` / `pass@16` / `throughput` numbers.
9. **Objective: maximise `throughput` while keeping `mean@16` above
   the quality floor.** `mean@16` is a gate, not a score to
   maximise — once it clears the floor, every further change should
   buy throughput.
