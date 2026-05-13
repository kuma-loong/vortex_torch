---
description: Run the long-horizon centroid-score kernel iterate loop autonomously — design 4 variants, JIT-compile, benchmark, generate reports, analyse, repeat. Stops after --max-iterations batches (default 3).
argument-hint: [--max-iterations <N>] [--q <Q>] [--num-gpus <G>]
---

You are now running the **centroid-score kernel iterate loop**
autonomously. Do not ask for confirmation between steps; execute each
step in sequence and loop until the iteration budget is reached.

Unlike `/iterate`, this loop tunes the *centroid-scoring kernel* used
inside the block-sparse-attention indexer — the kernel that, given a
batch of Q rows and a table of block centroids, scores each request's
candidate pages so the downstream `topk_output` can pick survivors.
Iteration is fast (single GPU, seconds-to-minutes per variant) and
produces a `(geomean speedup, max abs error, per-regime + per-batch
speedup tables)` report per variant.

## Step 0 — parse arguments and activate conda env

Parse `$ARGUMENTS` for `--max-iterations <N>` (default: 3),
`--q <Q>` (default: 4 — `NUM_Q_HEADS`, the Q-axis this session tunes
for), and `--num-gpus <G>` (default: 4 — the batch size, so behavior
matches the historical single-wave run).

- `Q` is the number of query heads every kernel in this session is
  tuned for; it picks the working folder
  `vortex_torch/kernels/centroid_score/q_${Q}/` and is passed to
  `benchmark.py --num-q-heads` so reports use the same axis. One
  `/iterate_centroids_score` session targets a single `Q`; spin
  separate sessions for separate Q values.
- `NUM_GPUS` is the **maximum** number of free GPUs the loop will
  consume per batch *and* the maximum batch size when it exceeds 4:
  - `NUM_GPUS <= 4` → batch = 4 (fixed), the loop runs `⌈4 / NUM_GPUS⌉`
    sequential waves on the cap-sized GPU pool. Floor of 4 keeps the
    "≥1 novelty slot + sweep slots" composition rule intact.
  - `NUM_GPUS > 4`  → batch = `NUM_GPUS`, fully parallel (or waves of
    however many GPUs are actually free if `free_gpus.sh` returns
    fewer than `NUM_GPUS`).

```bash
MAX_ITER=3
Q=4
NUM_GPUS=4
NEXT=
for arg in $ARGUMENTS; do
  case "$NEXT" in
    max-iterations) MAX_ITER=$arg; NEXT= ;;
    q)              Q=$arg;        NEXT= ;;
    num-gpus)       NUM_GPUS=$arg; NEXT= ;;
    *) case "$arg" in
         --max-iterations) NEXT=max-iterations ;;
         --q)              NEXT=q ;;
         --num-gpus)       NEXT=num-gpus ;;
       esac ;;
  esac
done
# Batch size floors at 4; expands with NUM_GPUS when bigger.
BATCH_SIZE=4
[ "$NUM_GPUS" -gt "$BATCH_SIZE" ] && BATCH_SIZE=$NUM_GPUS
echo "max_iterations=$MAX_ITER  q=$Q  num_gpus=$NUM_GPUS  batch_size=$BATCH_SIZE"
```

Activate the conda environment:
```bash
CONDA_BASE=$(conda info --base 2>/dev/null || echo /root/anaconda3)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate vortex_v04
python -c "import sys; print(sys.executable)"
```

## Step 1 — pick your tag and read context (once per session)

Your `<tag>` is a sanitized lowercase form of your model name
(e.g. `claude_sonnet_4_6`, `claude_opus_4_7`). Create the per-`Q`,
per-tag artifact dirs if they don't exist, and scaffold an empty
`memory_centroid_score.md` from the template below on first use of
this `Q`:

```bash
TAG=<your_tag>
QROOT=vortex_torch/kernels/centroid_score/q_${Q}
mkdir -p "$QROOT/sources/$TAG" "$QROOT/reports/$TAG"
# Make the q_${Q}/ tree importable as a Python package.
[ -f "$QROOT/__init__.py" ]                    || : > "$QROOT/__init__.py"
[ -f "$QROOT/sources/__init__.py" ]            || : > "$QROOT/sources/__init__.py"
[ -f "$QROOT/sources/$TAG/__init__.py" ]       || : > "$QROOT/sources/$TAG/__init__.py"

if [ ! -f "$QROOT/memory_centroid_score.md" ]; then
  cat > "$QROOT/memory_centroid_score.md" <<EOF
# memory_centroid_score.md — q=${Q}

Persistent state for \`/iterate_centroids_score --q ${Q}\`. Read at
the start of every session; mutate on every batch launch and every
batch completion. The conversation evaporates; this file does not.

The two reference kernels (against which all proposals are measured)
live at:
- Triton baseline — \`vortex_torch/kernels/centroid_score/baseline.py\`
  (\`.workload_size = 64\`, num_warps=8, num_stages=2; specialises to
  any \`num_q_heads\` via \`tl.constexpr\`).
- CUDA reference  — \`vortex_torch/kernels/centroid_score/baseline_cuda.py\`
  (one block per workload, one thread per page; not perf-tuned).

Submission contract (every proposal MUST satisfy):
\`\`\`python
def centroid_score(q, centroids, ctx, out) -> out
centroid_score.workload_size = <tile>   # positive int
\`\`\`
Implementation language is free — Triton or CUDA-C (compile via
\`cuda_loader.load_cuda_kernel\`). The ctx is sliced to the kernel's
\`.workload_size\`; baseline and proposal can use different tiles.
Submissions in this folder MUST be tuned for **\`num_q_heads=${Q}\`**
(other Q values get their own \`q_*\` bucket).

Benchmark protocol (per proposal): 200 stratified samples over
5 page-distribution regimes (\`N(256,64)\` … \`N(4096,1024)\`) × 5
batch sizes (\`{1,4,16,32,64}\`), \`n_blocks=16384\`, \`head_dim=128\`,
\`num_q_heads=${Q}\`. Reports land at
\`vortex_torch/kernels/centroid_score/q_${Q}/reports/<tag>/<batch_X_idY>.md\`.

Quality bar: **max |err| ≤ 0.1** vs the Triton baseline output
across the sweep (≈ a few bf16 ulps at values around 1.0). Anything
above that is structurally broken — log it in §4 and do not promote.
Speedup is geomean of \`baseline_ms / proposal_ms\`.

---

## §1 RUNNING

In-flight batches. One row per launch; remove the row when the
batch lands (move its summary to §2). If a row sits here at session
start, resume waiting on it before launching anything new.

| tag | batch | launched_at | gpu | variants | status |
| --- | ----- | ----------- | --- | -------- | ------ |
| _none_ | | | | | |

---

## §2 Completed batches

One subsection per completed batch (newest first). Each contains
the per-variant comparison table and a 1–3 sentence takeaway.

---

## §3 Hypotheses (pre-registered before launch)

- batch_X_id_Y — <hypothesis> — <novelty axis>

---

## §4 Anti-patterns / broken variants

- <description> — <reason> — <evidence pointer>

---

## §5 Pareto winners (running best by axis)

Updated after every batch. Each row is the best observed *so far*
on its axis; replace when a new variant strictly dominates.

| axis                                                  | submission | value | notes |
| ----------------------------------------------------- | ---------- | ----- | ----- |
| best overall geomean speedup (max |err| ≤ 0.1)        |            |       |       |
| best speedup on N(256,64) regime                      |            |       |       |
| best speedup on N(4096,1024) regime                   |            |       |       |
| best speedup at batch=1                               |            |       |       |
| best speedup at batch=64                              |            |       |       |

---

## §6 Insights

One bullet per file read or experiment-confirmed observation. Cite
\`file:line\` when applicable so the insight is checkable.

- **Seed: kernel is gather-bound on centroids.** The baseline loads
  ≈16 KB of bf16 centroids per workload from random page IDs; the
  dot product itself (~8K muls) is negligible. Optimisations that
  reduce centroid bandwidth or improve gather efficiency
  (\`num_stages\` pipelining, cache-modifier choice, fewer wasted
  reloads) outperform algorithmic rewrites.
- **Seed: Q is the same across consecutive workloads of one request.**
  \`winfo_q_indices\` often clusters same-batch workloads; today's
  baseline reloads Q every iteration even when \`batch_idx\` is
  unchanged. Cross-workload Q caching saves ~6% of memory traffic
  and the per-iter \`tl.sum\` of ${Q}×128 fp32.
- **Seed: singleton dims bloat the IR.** Baseline carries \`[1,${Q},128]\`
  Q tiles, \`[64,1,128]\` centroid tiles, \`[64,1,1]\` output stores —
  the \`1\`-axes cost real registers via Triton's broadcast handling.
- **Seed: bf16 → fp32 inside the gather doubles register pressure.**
  \`.to(tl.float32)\` on the centroid tile before multiply forces 32 KB
  of fp32 in registers; the multiply happily lowers as bf16×fp32 →
  fp32 accumulator if you cast *after* the elementwise mul. Often
  unblocks \`num_stages=3\`/\`4\`.

---

## §7 Final summaries

(one per \`/iterate_centroids_score --q ${Q}\` session — best
submission path, geomean speedup, worst-case |err|, key design
decisions.)
EOF
  echo "[scaffold] wrote $QROOT/memory_centroid_score.md"
fi
```

Read these files **in order** (skip any already loaded this session):
1. [vortex_torch/kernels/centroid_score/baseline.py](../../vortex_torch/kernels/centroid_score/baseline.py) — the Triton baseline, the submission contract, the tuning knobs (`num_warps`, `num_stages`, cache modifiers).
2. [vortex_torch/kernels/centroid_score/utils.py](../../vortex_torch/kernels/centroid_score/utils.py) — `SyntheticCtx` shape, `make_synthetic_ctx`, the torch reference scorer.
3. [vortex_torch/kernels/centroid_score/benchmark.py](../../vortex_torch/kernels/centroid_score/benchmark.py) — sweep protocol, stratified `(regime, batch_size)` sampling, report format.
4. [vortex_torch/kernels/centroid_score/dispatcher.py](../../vortex_torch/kernels/centroid_score/dispatcher.py) — the `.workload_size` contract enforcement.
5. [vortex_torch/kernels/centroid_score/cuda_loader.py](../../vortex_torch/kernels/centroid_score/cuda_loader.py) — JIT helper for CUDA submissions.
6. [vortex_torch/kernels/centroid_score/baseline_cuda.cu](../../vortex_torch/kernels/centroid_score/baseline_cuda.cu) + [baseline_cuda.py](../../vortex_torch/kernels/centroid_score/baseline_cuda.py) — worked CUDA example.
7. `vortex_torch/kernels/centroid_score/q_${Q}/memory_centroid_score.md` — persistent state across sessions for this `Q`. If missing, scaffold it (see §Memory schema below).

If `memory_centroid_score.md §1` shows a batch RUNNING, skip to Step 6
(wait activities) until it finishes, then resume the loop.

## Step 2 — set the batch index (once per loop iteration)

```bash
BATCH=$(ls vortex_torch/kernels/centroid_score/q_${Q}/sources/$TAG/batch_*_id0.py 2>/dev/null | wc -l)
echo "next batch index: $BATCH"
```

## Step 3 — design the batch (size = `$BATCH_SIZE`, set in Step 0)

**Hard ordering rule:** the next batch's design may only begin once
the previous batch has fully landed and Step 7's analysis has cleared
`memory_centroid_score.md §1`. If §1 still shows a `RUNNING` row at
this point (e.g. you resumed mid-loop), jump to Step 6 (poll + wait),
then Step 7 (analyse), and only then re-enter Step 3. **Do not
pre-author files during the wait window** — keep the GPU and the
agent focused on one batch at a time.

State the batch theme in one short paragraph. Define the knob matrix
— one row per variant (`id0…id$((BATCH_SIZE-1))`).

Each variant is one of:
- **Triton tuning** — copy the baseline's kernel into
  `q_${Q}/sources/$TAG/batch_${BATCH}_id<y>.py` and vary its compile-time
  knobs (`WORKLOAD_SIZE`, `num_warps`, `num_stages`, cache modifiers,
  whether to promote bf16→fp32 before or after multiply, whether to
  cache Q across consecutive same-batch workloads, whether to flatten
  singleton dims, etc.). Keep `NUM_Q_HEADS = ${Q}` hard-coded — this
  bucket is for that `Q` only.
- **New Triton algorithm** — re-author the kernel: e.g. one warp per
  page with cooperative dot product, `tl.dot`-based matmul with
  multiple Q rows packed in the M dim (only useful if amortising the
  16×16 tensor-core minimum), persistent kernel, fused Q-reduction +
  score in a single pass.
- **CUDA submission** — drop a `.cu` next to your `.py` wrapper, load
  it via `cuda_loader.load_cuda_kernel(...)`. The wrapper still
  exposes `centroid_score(q, centroids, ctx, out)` and
  `.workload_size`. Different `WORKLOAD_SIZE` is fine — the benchmark
  builds a per-kernel ctx (Q + centroids are seed-determined, only
  the ctx slicing changes).

Variant composition rule (scales with `$BATCH_SIZE`):
- **Genuinely novel: at least 1, aim for 2, scale up to ~25% of the
  batch for `$BATCH_SIZE >= 8`** (e.g. 2 novel at batch=8, 3–4 at
  batch=16). Place these in the low ids (`id0`, `id1`, …). A novel
  variant is a different algorithm or memory-layout idea, not just
  a knob sweep. Examples worth pursuing if not yet tried (check
  `memory_centroid_score.md §6` first): cross-workload Q caching
  keyed on `batch_idx`, `num_stages=3/4` with bf16-in-regs centroid
  (so the per-stage register cost halves), a CUDA submission using
  vectorised 8-byte centroid loads, a warp-per-page CUDA mapping
  with shuffle-reduce dot product, or a TMA-based centroid gather
  on H100+ if available.
- **Parameter sweeps: the remaining slots** — vary `WORKLOAD_SIZE`
  (32, 64, 128, 256), `num_warps` (4 vs 8), `num_stages`
  (2 vs 3 vs 4), or cache modifier (`.ca`/`.cg`/`.cs`) around the
  current winner in `memory_centroid_score.md §5`. With `$BATCH_SIZE`
  ≥ 8 you can afford a denser 2-D sweep (e.g. WS × num_stages),
  which maps the Pareto curve and gives the measured context to
  judge whether the novel idea is buying something.

Pre-register each novelty hypothesis as a one-sentence row in
`memory_centroid_score.md §3`.

## Step 4 — write files and JIT-compile

For each variant `y ∈ 0..$((BATCH_SIZE - 1))`:
1. Write the Python wrapper at
   `vortex_torch/kernels/centroid_score/q_${Q}/sources/$TAG/batch_${BATCH}_id${y}.py`.
   It MUST export a `centroid_score(q, centroids, ctx, out) -> out`
   function and attach `centroid_score.workload_size = <tile>`.
2. If introducing a CUDA kernel, also write
   `q_${Q}/sources/$TAG/batch_${BATCH}_id${y}.cu` next to the wrapper
   and have the `.py` call `load_cuda_kernel(...)`. The loader
   resolves relative paths against *its own* module directory
   (`kernels/centroid_score/`), not the caller's, so pass an
   absolute path:
   ```python
   from pathlib import Path
   _CU = Path(__file__).with_suffix(".cu")
   _mod = load_cuda_kernel(str(_CU), ...)
   ```

JIT-compile each (imports the module, instantiates the Triton kernel
or compiles the CUDA source, and runs one trivial invocation to
exercise the codegen path):
```bash
for y in $(seq 0 $((BATCH_SIZE - 1))); do
  python - <<PY
import sys, importlib, torch
spec = "vortex_torch.kernels.centroid_score.q_${Q}.sources.$TAG.batch_${BATCH}_id${y}"
m = importlib.import_module(spec)
fn = m.centroid_score
assert callable(fn), f"{spec}.centroid_score is not callable"
ws = fn.workload_size
assert isinstance(ws, int) and ws > 0, f"{spec}.centroid_score.workload_size = {ws!r}"
# Tiny smoke call so JIT codegen runs now, not at benchmark time.
from vortex_torch.kernels.centroid_score import make_synthetic_ctx
q = torch.zeros(1, ${Q}, 128, dtype=torch.bfloat16, device="cuda")
c = torch.zeros(ws, 128, dtype=torch.bfloat16, device="cuda")
ctx = make_synthetic_ctx(1, ws, ws, workload_size=ws, device="cuda")
out = torch.empty(ctx.total_ragged, dtype=torch.bfloat16, device="cuda")
fn(q, c, ctx, out)
torch.cuda.synchronize()
print(f"[ok] id${y}  ws={ws}")
PY
done
```
Fix any failure before continuing. Triton compile errors usually mean
a shape/dtype mismatch or an unsupported `cache_modifier`; nvcc
errors on CUDA submissions usually mean a missing sentinel
substitution or wrong include.

## Step 5 — benchmark the `$BATCH_SIZE` variants in parallel

Detect free GPUs, truncate to the `--num-gpus` cap, then decide
the wave width. `BATCH_SIZE` was set in Step 0 (= `max(4, NUM_GPUS)`).
With the cap applied, `N >= BATCH_SIZE` means all variants run in
parallel; `0 < N < BATCH_SIZE` falls back to sequential waves of `N`;
`N == 0` is a hard wait.
```bash
FREE_GPUS=($(algorithm_scientist/free_gpus.sh)) || {
  echo "no free GPUs — hard wait"; exit 1
}
if [ "${#FREE_GPUS[@]}" -gt "$NUM_GPUS" ]; then
  FREE_GPUS=("${FREE_GPUS[@]:0:$NUM_GPUS}")
fi
N=${#FREE_GPUS[@]}
PARALLEL=$N
[ "$PARALLEL" -gt "$BATCH_SIZE" ] && PARALLEL=$BATCH_SIZE
echo "free GPUs: ${FREE_GPUS[*]}  (N=$N, parallel=$PARALLEL, batch=$BATCH_SIZE, num_gpus_cap=$NUM_GPUS)"
```

Add a row to `memory_centroid_score.md §1` the moment you launch:
`| $TAG | batch_$BATCH | <time> | GPUs=${FREE_GPUS[*]} | batch_${BATCH}_id0…id$((BATCH_SIZE-1)) | RUNNING |`

Each variant runs the centroid-score benchmark with itself as the
sole `--proposal`. The baseline (Triton) is always the comparison
anchor; the report is written to a per-variant path so reports never
collide. Each variant is launched as a background job pinned to its
own GPU via `CUDA_VISIBLE_DEVICES`, polled every **120s**, and killed
if it exceeds **1200s (20 min)** wall clock.

```bash
TS=$(date +%Y%m%d_%H%M%S)
LOGDIR="logs/centroid_score/q${Q}_${TAG}_batch_${BATCH}_${TS}"
mkdir -p "$LOGDIR" "vortex_torch/kernels/centroid_score/q_${Q}/reports/$TAG"

POLL_SECS=120
KILL_AFTER_SECS=1200

declare -A PIDS START_TS GPU_OF

for start in $(seq 0 $PARALLEL $((BATCH_SIZE - 1))); do
  end=$((start + PARALLEL))
  [ "$end" -gt "$BATCH_SIZE" ] && end=$BATCH_SIZE
  echo "=== wave: variants [$start..$((end - 1))] ==="

  for y in $(seq $start $((end - 1))); do
    stem="batch_${BATCH}_id${y}"
    spec="vortex_torch.kernels.centroid_score.q_${Q}.sources.${TAG}.${stem}:centroid_score"
    report="vortex_torch/kernels/centroid_score/q_${Q}/reports/${TAG}/${stem}.md"
    gpu="${FREE_GPUS[$((y - start))]}"
    CUDA_VISIBLE_DEVICES=$gpu \
      python -m vortex_torch.kernels.centroid_score.benchmark \
        --proposal "$spec" \
        --num-q-heads "$Q" \
        --num-samples 200 \
        --report "$report" \
      > "$LOGDIR/${stem}.out" 2> "$LOGDIR/${stem}.err" &
    PIDS[$y]=$!
    START_TS[$y]=$(date +%s)
    GPU_OF[$y]=$gpu
    echo "[launch] $stem on GPU $gpu (pid=${PIDS[$y]})"
  done

  while :; do
    alive=0
    now=$(date +%s)
    for y in $(seq $start $((end - 1))); do
      pid="${PIDS[$y]}"
      [ -z "$pid" ] && continue
      if kill -0 "$pid" 2>/dev/null; then
        alive=1
        elapsed=$((now - START_TS[$y]))
        if [ "$elapsed" -ge "$KILL_AFTER_SECS" ]; then
          stem="batch_${BATCH}_id${y}"
          echo "[kill] $stem (GPU ${GPU_OF[$y]}) exceeded ${KILL_AFTER_SECS}s — SIGTERM"
          kill -TERM "$pid" 2>/dev/null
          sleep 5
          kill -KILL "$pid" 2>/dev/null
          echo "FAILED_TIMEOUT" > "$LOGDIR/${stem}.status"
          PIDS[$y]=""
        fi
      fi
    done
    [ "$alive" -eq 0 ] && break
    echo "[poll] $(date +%H:%M:%S) — wave [$start..$((end - 1))] still running"
    sleep "$POLL_SECS"
  done

  for y in $(seq $start $((end - 1))); do
    pid="${PIDS[$y]}"
    stem="batch_${BATCH}_id${y}"
    if [ -n "$pid" ]; then
      wait "$pid" 2>/dev/null
      rc=$?
      if [ ! -f "$LOGDIR/${stem}.status" ]; then
        if [ "$rc" -eq 0 ]; then
          echo "OK" > "$LOGDIR/${stem}.status"
        else
          echo "FAILED_RC_${rc}" > "$LOGDIR/${stem}.status"
        fi
      fi
    fi
    echo "[$stem] gpu=${GPU_OF[$y]} status=$(cat "$LOGDIR/${stem}.status")"
  done
done
```

The slot `<y>` is the variant index (0…3), NOT a GPU index — the
actual GPU is `FREE_GPUS[$((y - start))]` within the current wave.
When `N >= 4` there is exactly one wave of 4 (fully parallel). When
`N < 4` the outer loop runs ⌈4/N⌉ waves and wall-clock cost scales
accordingly. **Do not** try to grow `PARALLEL` mid-loop if another
user frees a GPU — concurrent oversubscription causes OOM or thrash.

Any variant whose `.status` file is not `OK` did not produce a
report — record it in `memory_centroid_score.md §4` as a broken
variant with the log path, and treat it as failed when filling the
comparison table in Step 7.

## Step 6 — what to do during the 2-minute poll windows

You are **blocked on the running benchmark** between poll ticks.
The only sanctioned wait activities are read-only:

- **Read.** Next file in priority order: the compiled artefact at
  `~/.vortex_compilation_cache/blocksparseattention_*_compiled_func.py`
  (the live JIT'd kernel — different from `baseline.py`!) →
  `csrc/topk_v2.cu` / `csrc/approx_topk.cu` for understanding the
  downstream `topk_output` call → other papers' centroid-style
  scoring tricks in `papers/`. Append one insight bullet to
  `memory_centroid_score.md §6` per poll cycle.
- **Note observations.** If a hypothesis from §3 already looks
  invalidated by the streaming stdout, append a note — but do NOT
  remove the row; the post-batch analysis in Step 7 owns §3 mutation.

**Forbidden during the wait window:**
- Designing or pre-authoring submissions for the next batch.
- Launching any other benchmark (concurrent batches contend for the GPU).
- Editing `memory_centroid_score.md §1` / §2 / §5 (those move only in Step 7).

## Step 7 — analyse, update memory_centroid_score.md, check budget

Read all `$BATCH_SIZE` reports from
`vortex_torch/kernels/centroid_score/q_${Q}/reports/$TAG/batch_${BATCH}_id*.md`.
Each contains:
- **§1 By (μ, σ) regime** — geomean speedup + max |err| per page-
  count distribution.
- **§2 By batch_size** — same axes per batch size.
- **§3 Overall** — single-row geomean across all 200 samples.

Produce a comparison summary:
```
| variant  | ws | overall speedup | min speedup (by regime) | min speedup (by batch) | max |err| |
```

Identify Pareto-non-dominated variants. A variant is interesting if:
- it has higher overall geomean speedup than the running §5 winner
  AND `max |err| ≤ 0.1`, OR
- it dominates on a single regime/batch row that the previous
  winner was weak on (e.g. small-batch latency, or the
  `N(4096,1024)` long-context row), AND `max |err| ≤ 0.1`.

Append the table + 1–3 sentence takeaway to
`memory_centroid_score.md §2`; remove the §1 row; update §3
(hypotheses) / §4 (anti-patterns) / §5 (winners).

**Check iteration budget:**
- Count batches launched this session =
  `ls vortex_torch/kernels/centroid_score/q_${Q}/sources/$TAG/batch_*_id0.py | wc -l`
  minus the count at session start.
- If `MAX_ITER > 0` AND `batches_launched >= MAX_ITER`: write a
  final 2-paragraph summary (best submission path, geomean
  speedup, worst-case |err|, key design decisions), update
  `memory_centroid_score.md §7`, and **stop**.
- Otherwise: go back to Step 2.

## Memory schema — `vortex_torch/kernels/centroid_score/q_${Q}/memory_centroid_score.md`

Step 1 scaffolds this file automatically on first use of a given
`Q`. The literal template lives in that heredoc — when you iterate,
keep these sections and use them as documented below:

```
# memory_centroid_score.md

## §1 RUNNING
| tag | batch | launched_at | gpu | variants | status |
| --- | ----- | ----------- | --- | -------- | ------ |

## §2 Completed batches
(one sub-section per completed batch: comparison table + 1–3
sentence takeaway)

## §3 Hypotheses (pre-registered before launch)
- batch_X_id_Y — <hypothesis> — <novelty axis>

## §4 Anti-patterns / broken variants
- <description> — <reason> — <evidence pointer>

## §5 Pareto winners (running best by axis)
- best overall geomean speedup (max |err| ≤ 0.1): <submission path> — <value>x
- best speedup on N(256,64) regime:   <submission path> — <value>x
- best speedup on N(4096,1024) regime: <submission path> — <value>x
- best speedup at batch=1:   <submission path> — <value>x
- best speedup at batch=64:  <submission path> — <value>x

## §6 Insights (one bullet per file read while waiting)
- <file:line> — <insight>

## §7 Final summaries (one per /iterate_centroids_score session)
```

Mutate this file on every launch and every completion. The
conversation evaporates; `memory_centroid_score.md` does not.
