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
higher = more KV-cache headroom = more throughput, but OOM risk),
swap `topK()` for `approxTopK(tolerate_ratio=…)` (`0.0`=exact,
higher=cheaper-but-looser; sweet spot 0.05-0.15). When two
variants both clear the floor, pick the faster one.

## Inventing beyond the literature

The `papers/` folder and [papers/guide.md](papers/guide.md) cover
what's already published — sinks, heavy hitters, channel sparsity,
low-rank K, LSH sampling, dual-band centroids. Treat them as
**seeds, not a menu.** A winning flow does not need a citation.
Every paper in there started by noticing a gap; the framework you
have (page-level selection, fused per-block kernel, Save/Load,
`Kron`, `MeanInterleave`) opens combinations and knobs no paper
here has explored. **Every batch must reserve at least one slot
for an off-catalog variant** — see `papers/guide.md` §16 for
prompts (paper combinations, knobs nobody has tried, claims worth
inverting, first-principles questions). Replicating a paper or
sweeping a single knob does not count.

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
8. [papers/guide.md](papers/guide.md) — synthesis of the ten
   sparse-attention papers in `papers/`. §14 = catalog of
   known-good submission ideas; **§16 = prompts for inventing
   flows that no paper here explores.**

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
- **`forward_indexer` must end in `topK(score, o, ctx=ctx)` or
  `approxTopK(tolerate_ratio=...)(score, o, ctx=ctx)`** — the
  score must be RAGGED `[S, 1, 1]`. `approxTopK` is a faster
  adaptive 8-bit radix variant; `tolerate_ratio ∈ [0.0, 1.0]`
  where `0.0` = exact, higher = cheaper but looser.
- **Cache-side reductions support `dim ∈ {1, 2}` only.** Cross-block
  reductions (`dim=0`) belong on the indexer side.
- **If a field is read+written across steps via `Load`/`Save`, zero
  it with `CFill(0.0)` in `forward_cache`.**
- **If `forward_indexer` uses `Save(...)`, the engine JSON MUST set
  `"disable_radix_cache": true`** (default `false`). sglang's
  prefix-radix cache otherwise shares per-request Save'd state
  across requests with matching prompt prefixes, corrupting
  Save/Load values. `check_engine_config` rejects the violation.

## Running the benchmark — policy

**The only allowed unit of work is one batch that fills every
*free* local GPU.** The host may share GPUs with other users; the
batch size depends on what's actually available *now*, not on the
physical GPU count. Detect free GPUs at the start of every batch:

```bash
FREE_GPUS=($(algorithm_scientist/free_gpus.sh)) || {
    echo "no free GPUs — wait, do not launch" >&2; exit 1
}
N=${#FREE_GPUS[@]}
echo "free GPUs: ${FREE_GPUS[*]}  (N=$N)"
```

`free_gpus.sh` excludes GPUs that have a compute process running
on them or memory.used ≥ 1024 MiB (override via
`free_gpus.sh <mib>`). Empty result (exit 1) ⇒ hard wait.

**File layout.** All submissions you write live under
`submissions/<tag>/`, where `<tag>` is your agent identifier
(default: a sanitized lowercase form of your model name, e.g.
`claude_opus_4_7`). Within that dir, batched runs use the
convention `batch_<x>_id<y>.{py,json}` (`<x>` = batch index,
`<y>` = per-GPU variant index). Single-variant runs are
debug-only. Each batch:

1. **`N` orthogonal variants** —
   `submissions/<tag>/batch_<x>_id0.{py,json}` …
   `submissions/<tag>/batch_<x>_id<N-1>.{py,json}` — varying
   different knobs. `N = ${#FREE_GPUS[@]}`, the number of
   currently-free GPUs (NOT the physical GPU count).
2. **Cheap local pre-flight first** for all `N` (CPU, no GPU):
   ```bash
   TAG=<your_agent_tag>; BATCH=<batch_index>
   for y in $(seq 0 $((N - 1))); do
     python -c "from vortex_torch.engine.sgl import check_engine_config; check_engine_config('submissions/${TAG}/batch_${BATCH}_id${y}.json')"
   done
   ```
   Refuse to launch any variant whose pre-flight fails.
3. **Launch `N` background `python` processes**, one per *free*
   GPU (pinned via `CUDA_VISIBLE_DEVICES=${FREE_GPUS[$y]}`), and
   `wait` for them all to finish:
   ```bash
   LOGDIR="logs/submission/${TAG}_batch_${BATCH}_$(date +%Y%m%d_%H%M%S)"
   mkdir -p "$LOGDIR"
   for y in $(seq 0 $((N - 1))); do
       cfg="submissions/${TAG}/batch_${BATCH}_id${y}.json"
       gpu="${FREE_GPUS[$y]}"
       stem=$(basename "$cfg" .json)
       CUDA_VISIBLE_DEVICES=$gpu \
           python algorithm_scientist/run_submission_aime24.py --config "$cfg" \
           > "$LOGDIR/gpu${gpu}_${stem}.out" \
           2> "$LOGDIR/gpu${gpu}_${stem}.err" &
   done
   wait
   ```
   The id `<y>` is the variant slot (0…N-1), NOT a GPU index —
   the actual GPU comes from `FREE_GPUS[$y]`. Each child writes
   its result into
   `summary_submissions/<tag>/<stem>/<timestamp>__<hash>.json`
   and updates `latest.json` on its own. The runner mirrors the
   config's path under `submissions/` into `summary_submissions/`,
   so `submissions/<tag>/batch_<x>_id<y>.json` becomes
   `summary_submissions/<tag>/batch_<x>_id<y>/...` — per-agent
   isolation, no collisions between agents that happen to use
   the same `batch_x_idy` stem.

4. **One batch at a time on the free GPUs.** Do not launch a
   second batch while the first is still running, and do not
   try to "fill the gaps" by launching extra variants on GPUs
   another user freed mid-batch — both contend for memory and
   either OOM or thrash. Use `jobs` (or `ls -lt
   summary_submissions/<tag>/*/latest.json`) to see how many
   children are still alive while you wait.

## While you wait (8+ hrs per batch)

Idle is not an option. Each polling cycle, do one of:

- **Read.** Priority: `AI/tutorials/` → `AI/developer_guides/` →
  `papers/` → `vortex_torch/flow/algorithms.py` →
  `vortex_torch/{indexer,cache}/*` → `csrc/`. After each file,
  append one insight to `algorithm_scientist/memory.md` §7.
- **Invent.** Open `papers/guide.md` §16 and pick one prompt —
  a paper combination, a knob no paper has tried, a claim
  worth inverting, or a first-principles question. Sketch a
  one-sentence hypothesis + cache/indexer ops. This fills the
  mandatory off-catalog slot in the next batch.
- **Design (don't launch) the rest of the next batch.**
  Pre-flight the `N` candidates so they're ready to fire the
  moment `wait` returns. Concurrent batches would OOM the
  shared GPUs.
- **Analyse children early.** As individual `latest.json` files
  appear (children finish at slightly different times), pull
  their `mean@16` / `throughput` and start filling a §2
  sub-section in memory.md. Close the §1 row when all `N` are
  in, then update §3 (hypotheses) / §4 (anti-patterns) / §5
  (winners).

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
   after a run, and on failure read the per-child log under
   `logs/submission/batch_<TS>/gpu<i>_<stem>.{out,err}` (or the
   `logs/submission/single_<TS>/<stem>.{out,err}` produced by
   `/benchmark`).

## Kickoff prompt for new sessions

To boot a fresh Claude Code session straight into the long-horizon
iterate loop, paste the prompt block from
[algorithm_scientist/iterate_kickoff.md](algorithm_scientist/iterate_kickoff.md)
into the new session's first message. The agent will identify
its tag, bootstrap from this primer + AGENTS.md + tutorials +
papers/guide.md + memory.md, and start the first batch
autonomously. State lives in `algorithm_scientist/memory.md` and
the `submissions/<tag>/` / `summary_submissions/<tag>/` trees,
so any later session resumes cleanly from the same prompt.

## Slash commands available in this session

- `/new-submission <name>` — scaffold a new submission pair.
- `/preflight <name>`      — run the cheap local pre-flight.
- `/batch-benchmark <n1> … <nN>` — launch exactly `N` variants in parallel on the *currently-free* GPUs (`N = $(algorithm_scientist/free_gpus.sh | wc -w)`; the only sanctioned benchmark command).
- `/review <name>`         — audit a submission against AGENTS.md rules.
- `/iterate <name>`        — kick off a full auto-iteration loop (batches that fill every local GPU, one batch at a time, updates memory.md).
- `/benchmark <name>`      — *debug only*: run a single variant directly. Do not use in normal workflow.

## Subagents available

- `vortex-submission-writer` — drafts and iterates on a submission.
- `vortex-submission-reviewer` — audits a submission pair for
  rule violations without editing anything.

Use `Task(subagent_type="vortex-submission-writer", ...)` from the main
agent or invoke via slash command.
