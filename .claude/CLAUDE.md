# vortex_torch — project primer for Claude Code

`vortex_torch` is a JIT-compiled sparse-attention framework that plugs
into sglang's decode loop. Users / AI agents write a **sparse-attention
submission** — two files in `submissions/` — and the framework compiles
them into Triton kernels at runtime.

## Objective (non-negotiable)

> **Maximise decoding throughput (tokens/sec) on the AIME24 benchmark
> while keeping `mean@16` at or above a minimum acceptable floor.**

`mean@16` is a *quality gate*, not something to maximise. Once a flow
clears the floor, every subsequent change should trade accuracy-headroom
for **more throughput**: tighter `vortex_topk_val` / `vortex_topk_ratio`,
fewer cache-side ops, fewer intermediate cache fields, narrower
`vortex_layers_skip`, aggressive fp8 `kv_cache_dtype`, raise
`mem_fraction_static` toward 0.9 (range [0.5, 0.95], default 0.8 —
higher = more KV-cache headroom = more throughput, but OOM risk).
When two variants both clear the floor, pick the faster one.

## Where the instructions live

All authoritative content lives under [AI/](AI/). Read in order:

1. [AI/AGENTS.md](AI/AGENTS.md) — the full submission contract, rules,
   budget / BOS / layer-skip semantics, benchmark protocol.
2. [AI/tutorials/overview.md](AI/tutorials/overview.md) — 5-minute map.
3. [AI/tutorials/program_create_cache.md](AI/tutorials/program_create_cache.md)
4. [AI/tutorials/program_forward_cache.md](AI/tutorials/program_forward_cache.md)
5. [AI/tutorials/program_forward_indexer.md](AI/tutorials/program_forward_indexer.md)
6. [AI/tutorials/cache_op.md](AI/tutorials/cache_op.md) — indexer-side
   op math reference.
7. [AI/tutorials/indexer_op.md](AI/tutorials/indexer_op.md) — cache-side
   op math reference.

Framework-internal deep dives live in
[AI/developer_guides/](AI/developer_guides/) — needed only if you are
modifying the compiler itself, not when writing a submission.

## Hard constraints

- **No native torch ops** inside `create_cache` / `forward_cache` /
  `forward_indexer`. Every tensor goes through
  `vortex_torch.indexer.*` / `vortex_torch.cache.*` ops. `.view`,
  `.sum(dim=...)`, elementwise torch, etc. will not compile.
- **Each op instance is one call site.** `self.mul_a = Multiply()`
  and `self.mul_b = Multiply()` — do not share.
- **Do not declare `"k"` or `"v"`** in `create_cache`; they are
  auto-provided.
- **`forward_indexer` must end in `topK(score, o, ctx=ctx)`** with
  `score.shape == [S, 1, 1]`.
- **Cache-side reductions support `dim ∈ {1, 2}` only.** Cross-block
  reductions (`dim=0`) belong on the indexer side.
- **If a field is read+written across steps via `Load`/`Save`, zero
  it with `CFill(0.0)` in `forward_cache`.**

## Running the benchmark — policy

**The only allowed unit of work is a batch of 8.** Single-variant
runs (`run_submission.slurm`) are debug-only and off-limits to the
automated workflow. Each batch:

1. **8 orthogonal variants** — `submissions/<tag>_v1.{py,json}` …
   `submissions/<tag>_v8.{py,json}` — varying different knobs.
2. **Cheap local pre-flight first** for all 8 (CPU, no GPU):
   ```bash
   for i in 1 2 3 4 5 6 7 8; do
     python -c "from vortex_torch.engine.sgl import check_engine_config; check_engine_config('submissions/<tag>_v${i}.json')"
   done
   ```
   Refuse to sbatch any variant whose pre-flight fails.
3. **Submit on Slurm** — the host has no GPU, so direct invocation
   of `algorithm_scientist/run_submission_aime24.py` will fail.
   Always use the batched slurm file (whole 8-GPU node, each child
   pinned to its own `CUDA_VISIBLE_DEVICES=0..7`):
   ```bash
   sbatch algorithm_scientist/run_submission_batch.slurm \
       submissions/<tag>_v1.json submissions/<tag>_v2.json \
       submissions/<tag>_v3.json submissions/<tag>_v4.json \
       submissions/<tag>_v5.json submissions/<tag>_v6.json \
       submissions/<tag>_v7.json submissions/<tag>_v8.json
   ```
   Per-child logs land under
   `logs/submission/batch_<JOBID>/gpu<i>_<stem>.{out,err}`.

4. **In-flight ceiling = 24 experiments (3 batches).** Before every
   sbatch:
   ```bash
   squeue -u $USER -h -o '%i %j %T'
   ```
   If 3 `vortex_submission_batch` rows are already PENDING/RUNNING,
   do not submit — work on the wait-time protocol below until a
   slot frees up.

5. Poll:
   ```bash
   sacct -j <JOBID> --format=JobID,State,ExitCode -X -n -P
   ```

## While you wait (8+ hrs per batch)

Idle is not an option. Each polling cycle, do one of:

- **Read.** Priority: `AI/tutorials/` → `AI/developer_guides/` →
  `vortex_torch/flow/algorithms.py` →
  `vortex_torch/{indexer,cache}/*` → `csrc/`. After each file,
  append one insight to `algorithm_scientist/memory.md` §7.
- **Prepare the next batch.** If fewer than 3 batches in flight,
  design 8 orthogonal variants for a different theme, pre-flight
  them, submit, record in `algorithm_scientist/memory.md` §1.
- **Analyse.** For any terminal batch, read all 8 `latest.json`
  files, fill a §2 sub-section (table + takeaway), remove the §1
  row, update §3 (hypotheses) / §4 (anti-patterns) / §5 (winners).

## Persistent state — `algorithm_scientist/memory.md`

The conversation evaporates; `memory.md` does not. Read it at the
start of every session and write to it before stopping. Any batch
submission and any batch completion must mutate it.
   When the job finishes, each run is written into a
   per-submission subfolder so iterations never collide:

   ```
   summary_submissions/<name>/
       <timestamp>__<content_hash>.json   # full summary + embedded .py/.json
       latest.json                        # symlink → newest run
       INDEX.jsonl                        # one-row-per-run index
   ```

   The content hash is `sha256(config.json || module.py)` truncated
   to 12 chars — same code → same hash → you can see re-runs
   at a glance. Read `summary_submissions/<name>/latest.json`
   after a run, and on failure read
   `logs/submission/vortex_submission_<JOBID>.{out,err}`.

## Slash commands available in this session

- `/new-submission <name>` — scaffold a new submission pair.
- `/preflight <name>`      — run the cheap local pre-flight.
- `/batch-benchmark <n1> … <n8>` — sbatch exactly 8 variants on one node (the only sanctioned benchmark command).
- `/review <name>`         — audit a submission against AGENTS.md rules.
- `/iterate <name>`        — kick off a full auto-iteration loop (batches of 8, ≤ 24 in flight, updates memory.md).
- `/benchmark <name>`      — *debug only*: sbatch a single-variant AIME24 run. Do not use in normal workflow.

## Subagents available

- `vortex-submission-writer` — drafts and iterates on a submission.
- `vortex-submission-reviewer` — audits a submission pair for
  rule violations without editing anything.

Use `Task(subagent_type="vortex-submission-writer", ...)` from the main
agent or invoke via slash command.
