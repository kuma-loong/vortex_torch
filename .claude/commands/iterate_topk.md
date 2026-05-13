---
description: Run the long-horizon top-k kernel iterate loop autonomously — design 4 variants, JIT-compile, benchmark, generate reports, analyse, repeat. Stops after --max-iterations batches (default 3).
argument-hint: [--max-iterations <N>] [--k <K>] [--num-gpus <G>]
---

You are now running the **top-k kernel iterate loop** autonomously.
Do not ask for confirmation between steps; execute each step in
sequence and loop until the iteration budget is reached.

Unlike `/iterate`, this loop tunes the *top-k selection kernel* used
inside the sparse-attention indexer — a CUDA kernel that, given
per-segment scores, returns the top-K indices. Iteration is fast
(single GPU, seconds-to-minutes per variant) and produces a
`(geomean speedup, recall@189, recall@253)` report per variant.

## Step 0 — parse arguments and activate conda env

Parse `$ARGUMENTS` for `--max-iterations <N>` (default: 3),
`--k <K>` (default: 256), and `--num-gpus <G>` (default: 4 — the
batch size, so behavior matches the historical single-wave run).

- `K` is the top-k size every kernel in this session is tuned for;
  it picks the working folder `vortex_torch/kernels/topk/k_${K}/`
  and is passed to `benchmark.py --k` so reports use the same
  eval points.
- `NUM_GPUS` is the **maximum** number of free GPUs the loop will
  consume per batch. The batch is always 4 variants; with
  `NUM_GPUS < 4` the loop runs `⌈4 / NUM_GPUS⌉` sequential waves
  on the cap-sized GPU pool. With `NUM_GPUS >= 4` (and ≥4 free
  GPUs available) the whole batch fires in a single wave.
```bash
MAX_ITER=3
K=256
NUM_GPUS=4
NEXT=
for arg in $ARGUMENTS; do
  case "$NEXT" in
    max-iterations) MAX_ITER=$arg; NEXT= ;;
    k)              K=$arg;        NEXT= ;;
    num-gpus)       NUM_GPUS=$arg; NEXT= ;;
    *) case "$arg" in
         --max-iterations) NEXT=max-iterations ;;
         --k)              NEXT=k ;;
         --num-gpus)       NEXT=num-gpus ;;
       esac ;;
  esac
done
echo "max_iterations=$MAX_ITER  k=$K  num_gpus=$NUM_GPUS"
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
(e.g. `claude_sonnet_4_6`, `claude_opus_4_7`). Create the per-tag
artifact dirs if they don't exist, and scaffold an empty
`memory_topk.md` from the template below on first use of this K:
```bash
TAG=<your_tag>
KROOT=vortex_torch/kernels/topk/k_${K}
mkdir -p "$KROOT/configs/$TAG" "$KROOT/sources/$TAG" "$KROOT/reports/$TAG"

if [ ! -f "$KROOT/memory_topk.md" ]; then
  cat > "$KROOT/memory_topk.md" <<EOF
# memory_topk.md — k=${K}

Persistent state for \`/iterate_topk --k ${K}\`. Read at the start of
every session; mutate on every batch launch and every batch
completion. The conversation evaporates; this file does not.

The two reference kernels (against which all proposals are measured)
live at:
- baseline — \`vortex_torch/kernels/topk/configs/sort_default.json\` (CUB BlockRadixSort)
- proposal seed — \`vortex_torch/kernels/topk/configs/radix_default.json\` (8-bit radix)

Benchmark protocol (per proposal): 100 samples × 4 batch sizes ×
4 seq_lens, scores drawn from a 14-distribution rotation, top-k
size **K=${K}**. Recall is reported at \`floor(K*3/4)\` and \`K\`.
Reports land at \`vortex_torch/kernels/topk/k_${K}/reports/<tag>/<batch_X_idY>.md\`.

Quality bar: **min recall@K ≥ 0.98** on every distribution.
Anything below that is structurally broken — log it in §4 and do
not promote.

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

| axis                                          | config | value | notes |
| --------------------------------------------- | ------ | ----- | ----- |
| best geomean speedup                          |        |       |       |
| best worst-case recall@K with speedup ≥ 1.0   |        |       |       |

---

## §6 Insights

One bullet per file read or experiment-confirmed observation. Cite
\`file:line\` when applicable so the insight is checkable.

- **Seed: read prior K buckets first.** Before designing batch 0,
  scan \`vortex_torch/kernels/topk/k_*/memory_topk.md\` for any
  sibling bucket — especially \`k_256/memory_topk.md\` if present.
  Prior experiments expose anti-patterns (§4), alignment / smem
  occupancy caveats (§6), and Pareto winners (§5) that frequently
  transfer across K. Carry the relevant bullets into this file's
  §6 with a "from k_<other>:" prefix instead of re-deriving them.
- **Seed: the algorithm space is wider than radix.** The current
  proposal seed is 8-bit radix (\`radix_topk.cu\`); the baseline is
  CUB \`BlockRadixSort\` (\`sort_topk.cu\`). Other families are
  unexplored and have very different occupancy / recall tradeoffs:
  \`bitonic\` (warp-cooperative sort, no smem candidate buffer),
  hybrid \`radix → bitonic refinement\` for the threshold bin,
  \`heap\` (per-warp min-heap of size K), \`approxTopK\`-style
  atomic-arrival within the threshold bin, and 16-bit-radix
  attacking the bf16-bottom-zeros half of the baseline's sort
  width (see \`csrc/topk.cu\`). Reserve novelty slots (id0–id1)
  for the unexplored family this K most needs.

---

## §7 Final summaries

(one per \`/iterate_topk\` session — best config path, geomean
speedup, worst-case recall, key design decisions.)
EOF
  echo "[scaffold] wrote $KROOT/memory_topk.md"
fi
```

Read these files **in order** (skip any already loaded this session):
1. [vortex_torch/kernels/topk/dispatcher.py](../../vortex_torch/kernels/topk/dispatcher.py) — the JIT compile / cache layer; understand `get_kernel` and `load_submission`.
2. [vortex_torch/kernels/topk/benchmark.py](../../vortex_torch/kernels/topk/benchmark.py) — the benchmark protocol, distribution rotation, report format.
3. [vortex_torch/kernels/topk/radix_topk.cu](../../vortex_torch/kernels/topk/radix_topk.cu) — the 8-bit radix proposal kernel (the current default).
4. [vortex_torch/kernels/topk/sort_topk.cu](../../vortex_torch/kernels/topk/sort_topk.cu) — the CUB `BlockRadixSort` baseline kernel.
5. [vortex_torch/kernels/topk/configs/radix_default.json](../../vortex_torch/kernels/topk/configs/radix_default.json) and [vortex_torch/kernels/topk/configs/sort_default.json](../../vortex_torch/kernels/topk/configs/sort_default.json) — the reference configs and what their substitutions / cflags look like.
6. [vortex_torch/kernels/topk/k_${K}/memory_topk.md](../../vortex_torch/kernels/topk/k_${K}/memory_topk.md) — persistent state across sessions. If missing, scaffold it (see §Memory schema below).

If `memory_topk.md §1` shows a batch RUNNING, skip to Step 6 (wait
activities) until it finishes, then resume the loop.

## Step 2 — set the batch index (once per loop iteration)

```bash
BATCH=$(ls vortex_torch/kernels/topk/k_${K}/configs/$TAG/batch_*_id0.json 2>/dev/null | wc -l)
echo "next batch index: $BATCH"
```

## Step 3 — design the 4-variant batch

**Hard ordering rule:** the next batch's design may only begin once
the previous batch has fully landed and Step 7's analysis has cleared
`memory_topk.md §1`. If §1 still shows a `RUNNING` row at this point
(e.g. you resumed mid-loop), jump to Step 6 (poll + wait), then Step 7
(analyse), and only then re-enter Step 3. **Do not pre-author configs
during the wait window** — keep the GPU and the agent focused on one
batch at a time.

State the batch theme in one short paragraph. Define the knob matrix
— one row per variant (id0…id3).

Each variant is one of:
- **Substitution sweep** — reuse an existing `.cu` file
  (`radix_topk.cu` or a previously-introduced source) and vary its
  compile-time knobs (`__THREADS_PER_BLOCK__`, `__VORTEX_MAX_TOPK__`,
  `__SMEM_BYTES__`, or whatever sentinels you added to a new source).
- **New algorithm** — write a fresh `.cu` file under
  `vortex_torch/kernels/topk/k_${K}/sources/$TAG/batch_${BATCH}_id<y>.cu` exporting
  `void topk(...)` with the same signature as the existing kernels.
  Add sentinels for any compile-time knob you want sweep-able.

Variant composition rule:
- **id0–id1 (aim for 2): genuinely novel** — a different algorithm
  (e.g. heap-based, bitonic, warp-cooperative split-K, hybrid
  radix→sort refinement), a different memory layout, or an
  unexplored knob axis. One sentence defending each, naming the
  specific kernel mechanism exploited.
- **id2–id3: parameter sweeps** — vary `threads_per_block`,
  `max_topk`, `smem_bytes`, or other sentinels around the current
  best in `memory_topk.md §5`. These map the Pareto curve and give
  the measured context to judge whether the novel idea is buying
  something.

Pre-register each novelty hypothesis as a one-sentence row in
`memory_topk.md §3`.

## Step 4 — write files and JIT-compile

For each variant `y ∈ 0..3`:
1. Write `vortex_torch/kernels/topk/k_${K}/configs/$TAG/batch_${BATCH}_id${y}.json`. The
   JSON's `"file"` must be a path resolvable from the config's
   directory (use a path like `"../../sources/$TAG/..."` for new
   sources, or `"../../radix_topk.cu"` to reuse).
2. If introducing a new algorithm, also write
   `vortex_torch/kernels/topk/k_${K}/sources/$TAG/batch_${BATCH}_id${y}.cu`. It must
   `#include` the same ATen/CUDA headers as `radix_topk.cu` and
   export `void topk(...)` with the shared signature.

JIT-compile each (CPU happens during nvcc, GPU not required to
*compile*, but the toolchain expects a CUDA-capable host):
```bash
for y in 0 1 2 3; do
  python -c "
from vortex_torch.kernels.topk.dispatcher import load_submission
load_submission('vortex_torch/kernels/topk/k_${K}/configs/$TAG/batch_${BATCH}_id${y}.json', verbose=False)
print('[ok] id${y}')
" || echo "[FAIL] id${y}"
done
```
Fix any compile failure before continuing. nvcc errors usually
mean a missing sentinel substitution, wrong include, or a CUB API
mismatch — re-read the source side-by-side with `radix_topk.cu`.

## Step 5 — benchmark the 4 variants in parallel

Detect free GPUs, truncate to the `--num-gpus` cap, then decide
the wave width. With the cap applied, `N >= 4` means all 4
variants run in parallel; `0 < N < 4` falls back to sequential
waves of `N`; `N == 0` is a hard wait.
```bash
FREE_GPUS=($(algorithm_scientist/free_gpus.sh)) || {
  echo "no free GPUs — hard wait"; exit 1
}
# Truncate to the --num-gpus cap (default 4) so the loop never
# consumes more than the requested number of GPUs even if more
# are free.
if [ "${#FREE_GPUS[@]}" -gt "$NUM_GPUS" ]; then
  FREE_GPUS=("${FREE_GPUS[@]:0:$NUM_GPUS}")
fi
N=${#FREE_GPUS[@]}
BATCH_SIZE=4
PARALLEL=$N
[ "$PARALLEL" -gt "$BATCH_SIZE" ] && PARALLEL=$BATCH_SIZE
echo "free GPUs: ${FREE_GPUS[*]}  (N=$N, parallel=$PARALLEL, batch=$BATCH_SIZE, num_gpus_cap=$NUM_GPUS)"
```

Add a row to `memory_topk.md §1` the moment you launch:
`| $TAG | batch_$BATCH | <time> | GPUs=${FREE_GPUS[*]} | batch_${BATCH}_id0…id3 | RUNNING |`

Each variant is launched as a background job pinned to its own GPU
via `CUDA_VISIBLE_DEVICES`, polled every **120s**, and killed if it
exceeds **1200s (20 min)** wall clock. The script blocks here until
every variant in the wave has either landed or been killed before
starting the next wave. `benchmark.py` mirrors the config's path
under `configs/` into `reports/`, so a config at
`k_${K}/configs/$TAG/batch_X_idY.json` produces a report at
`k_${K}/reports/$TAG/batch_X_idY.md` automatically — no post-processing
needed.

```bash
TS=$(date +%Y%m%d_%H%M%S)
LOGDIR="logs/topk/${TAG}_batch_${BATCH}_${TS}"
mkdir -p "$LOGDIR"

POLL_SECS=120
KILL_AFTER_SECS=1200

declare -A PIDS START_TS GPU_OF

for start in $(seq 0 $PARALLEL $((BATCH_SIZE - 1))); do
  end=$((start + PARALLEL))
  [ "$end" -gt "$BATCH_SIZE" ] && end=$BATCH_SIZE
  echo "=== wave: variants [$start..$((end - 1))] ==="

  # Launch this wave — one child per (slot, GPU) pair.
  for y in $(seq $start $((end - 1))); do
    cfg="vortex_torch/kernels/topk/k_${K}/configs/$TAG/batch_${BATCH}_id${y}.json"
    stem="batch_${BATCH}_id${y}"
    gpu="${FREE_GPUS[$((y - start))]}"
    CUDA_VISIBLE_DEVICES=$gpu \
      python -m vortex_torch.kernels.topk.benchmark \
        --proposal-config "$cfg" \
        --num-samples 100 \
        --k "$K" \
      > "$LOGDIR/${stem}.out" 2> "$LOGDIR/${stem}.err" &
    PIDS[$y]=$!
    START_TS[$y]=$(date +%s)
    GPU_OF[$y]=$gpu
    echo "[launch] $stem on GPU $gpu (pid=${PIDS[$y]})"
  done

  # Poll every POLL_SECS until every child in this wave is gone.
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

  # Reap exit codes for variants that finished naturally.
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
report — record it in `memory_topk.md §4` as a broken variant with
the log path, and treat it as failed when filling the comparison
table in Step 7.

## Step 6 — what to do during the 2-minute poll windows

You are **blocked on the running benchmark** between poll ticks.
The only sanctioned wait activities are read-only:

- **Read.** Next file in priority order: dispatcher's `load_inline`
  cache logic → benchmark distribution sampler → existing `.cu`
  sources → `csrc/topk_v2.cu` / `csrc/approx_topk.cu` (the upstream
  references). Append one insight bullet to `memory_topk.md §6` per
  poll cycle.
- **Note observations.** If a hypothesis from §3 already looks
  invalidated by the streaming stdout, append a note — but do NOT
  remove the row; the post-batch analysis in Step 7 owns §3 mutation.

**Forbidden during the wait window:**
- Designing or pre-authoring configs for the next batch.
- Launching any other benchmark (concurrent batches contend for the GPU).
- Editing `memory_topk.md §1` / §2 / §5 (those move only in Step 7).

## Step 7 — analyse, update memory_topk.md, check budget

Read all 4 reports from `vortex_torch/kernels/topk/k_${K}/reports/$TAG/batch_${BATCH}_id*.md`.
Each contains the per-`(bs, seq_len)` and per-`(seq_len, distribution)`
tables.

Produce a comparison summary:
```
| variant       | geomean speedup | min R@189 | min R@253 |
```

Identify Pareto-non-dominated variants on the
`(geomean_speedup, min_recall@253)` plane. Append the table + 1–3
sentence takeaway to `memory_topk.md §2`; remove the §1 row;
update §3 (hypotheses) / §4 (anti-patterns) / §5 (winners).

**Check iteration budget:**
- Count batches launched this session =
  `ls vortex_torch/kernels/topk/k_${K}/configs/$TAG/batch_*_id0.json | wc -l` minus the
  count at session start.
- If `MAX_ITER > 0` AND `batches_launched >= MAX_ITER`: write a
  final 2-paragraph summary (best config path, geomean speedup,
  worst-case recall, key design decisions), update
  `memory_topk.md §7`, and **stop**.
- Otherwise: go back to Step 2.

## Memory schema — `vortex_torch/kernels/topk/k_${K}/memory_topk.md`

Step 1 scaffolds this file automatically on first use of a given
`K`. The literal template lives in that heredoc — when you iterate,
keep these sections and use them as documented below:

```
# memory_topk.md

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
- best geomean speedup: <config path> — <value>x @ R@253=<value>
- best worst-case R@253 with speedup ≥ 1.0: <config path> — ...

## §6 Insights (one bullet per file read while waiting)
- <file:line> — <insight>

## §7 Final summaries (one per /iterate_topk session)
```

Mutate this file on every launch and every completion. The
conversation evaporates; `memory_topk.md` does not.
