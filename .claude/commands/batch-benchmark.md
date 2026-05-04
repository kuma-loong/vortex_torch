---
description: Launch one submission per *free* local GPU in parallel (the only sanctioned benchmark form). Pass exactly N submission names, where N is the count returned by algorithm_scientist/free_gpus.sh.
argument-hint: <name1> <name2> ... <nameN>   (N = number of currently-free GPUs)
---

Run **one submission per *free* local GPU**, in parallel, against
the AIME24 benchmark. This is the *only* benchmark command
sanctioned by the protocol — single-variant runs are debug-only.

The user passes **submission names** (not JSON paths); this command
expands each `<nameI>` into `submissions/<tag>/<nameI>.json`,
where `<tag>` is the session's agent identifier. For the standard
iterate-loop layout, the names look like
`batch_<x>_id0 batch_<x>_id1 … batch_<x>_idN-1`.

Step 0 — resolve `<tag>`, detect *free* GPUs (not the physical
count — other users may be sharing this host), then count
arguments:
```bash
TAG=<your_agent_tag>           # sanitized model name, set once per session
FREE_GPUS=($(algorithm_scientist/free_gpus.sh)) || {
    echo "no free GPUs — wait, do not launch" >&2; exit 1
}
N=${#FREE_GPUS[@]}
NAMES=($ARGUMENTS)
if [ "${#NAMES[@]}" -ne "$N" ]; then
    echo "expected $N variants (one per FREE GPU: ${FREE_GPUS[*]}), got ${#NAMES[@]}" >&2
    exit 1
fi
```
If the user supplied a different number, refuse and explain:
"Batches must contain exactly `$N` variants — one per *currently-
free* GPU (`${FREE_GPUS[*]}`). The number of free GPUs can change
between batches; re-detect and re-design before re-invoking."
Do not silently shrink or pad the batch.

Step 1 — concurrency cap. Only **one** batch may run at a time
on the GPUs you target:
```bash
jobs -r | wc -l
```
If background jobs from a previous batch are still alive, refuse
to launch — tell the user to wait, or to use the wait-time
activities in `algorithm_scientist/memory.md`. (Other users'
processes are already filtered out by `free_gpus.sh`.)

Step 2 — pre-flight every config locally first (cheap, no GPU):
```bash
for name in "${NAMES[@]}"; do
    python -c "from vortex_torch.engine.sgl import check_engine_config; check_engine_config('submissions/${TAG}/${name}.json')" \
        || echo "[preflight] FAILED: ${TAG}/${name}"
done
```
Refuse to launch any submission whose preflight failed — fix the
failing variant first.

Step 3 — fork `N` background `python` processes pinned to the
*free* GPU indices (NOT 0…N-1) and `wait` for them all:
```bash
LOGDIR="logs/submission/${TAG}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"
for y in $(seq 0 $((N - 1))); do
    name="${NAMES[$y]}"
    gpu="${FREE_GPUS[$y]}"
    CUDA_VISIBLE_DEVICES=$gpu \
        python algorithm_scientist/run_submission_aime24.py --config "submissions/${TAG}/${name}.json" \
        > "$LOGDIR/gpu${gpu}_${name}.out" \
        2> "$LOGDIR/gpu${gpu}_${name}.err" &
done
wait
```
Each child writes its result into
`summary_submissions/<tag>/<name>/<timestamp>__<hash>.json` and
updates `summary_submissions/<tag>/<name>/latest.json` itself.
(The runner mirrors the config's path under `submissions/` into
`summary_submissions/`, so `submissions/<tag>/batch_<x>_id<y>.json`
becomes `summary_submissions/<tag>/batch_<x>_id<y>/...` — per-agent
isolation, no collisions across agents that pick the same stem.)

Step 4 — append a row to `algorithm_scientist/memory.md` §1
*In-flight batches* the moment you launch:
`| <tag> | <batch_id> | <UTC time> | <LOGDIR> | <name1>,…,<nameN> | RUNNING |`

Step 5 — while waiting (the batch takes **8+ hours**), do NOT
idle. Spend the time reading tutorials / developer guides /
source, or designing the next batch (don't launch — concurrent
batches OOM the shared GPUs), or analysing children that have
already produced their `latest.json`. See AGENTS.md §5d.

Step 6 — once `wait` returns:

- **Success** → for each `<nameI>`, read
  `summary_submissions/<tag>/<nameI>/latest.json`. Produce a comparison
  table with columns: `name | content_hash | mean@16 | pass@16 |
  throughput | e2e_time`. Recommend the one with the highest
  `throughput` **that also clears the quality floor**. Append the
  table + a 1-3 sentence takeaway to `algorithm_scientist/memory.md`
  §2 *Completed batches*; remove the row you added in step 4 from §1.
- **Failure** → any child whose process exited non-zero (or
  whose `latest.json` is missing / older than `$LOGDIR`'s
  timestamp) wrote its traceback to
  `$LOGDIR/gpu<i>_<name>.err`. Open the failing child's `.err`,
  summarise the error, and record the lesson in
  `algorithm_scientist/memory.md` §4 *Anti-patterns*.
