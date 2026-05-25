---
description: Run the long-horizon iterate loop autonomously — design 4 variants, pre-flight, RULER filter, launch AIME24, wait, analyse, repeat. Stops after --max-iterations batches (default 3).
argument-hint: [--max-iterations <N>]
---

You are now running the **vortex_torch iterate loop** autonomously.
Do not ask for confirmation between steps; execute each step in
sequence and loop until the iteration budget is reached.

## Step 0 — parse arguments and activate conda env

Parse `$ARGUMENTS` for `--max-iterations <N>` (default: 3).
Extract N:
```bash
MAX_ITER=3
for arg in $ARGUMENTS; do
  case "$arg" in --max-iterations) MAX_ITER_NEXT=1 ;; *)
    [ "$MAX_ITER_NEXT" = "1" ] && { MAX_ITER=$arg; MAX_ITER_NEXT=0; } ;; esac
done
echo "max_iterations=$MAX_ITER"
```

Activate the conda environment:
```bash
CONDA_BASE=$(conda info --base 2>/dev/null || echo /root/anaconda3)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate vortex_v1
python -c "import sys; print(sys.executable)"
```

## Step 1 — pick your tag and read context (once per session)

Your `<tag>` is a sanitized lowercase form of your model name
(e.g. `claude_sonnet_4_6`, `claude_opus_4_7`). Create the
submissions dir if it doesn't exist:
```bash
TAG=<your_tag>
mkdir -p submissions/$TAG
```

Read these files **in order** (skip any already loaded this session):
1. [AI/AGENTS.md](../AI/AGENTS.md)
2. [AI/tutorials/overview.md](../AI/tutorials/overview.md)
3. [AI/tutorials/program_create_cache.md](../AI/tutorials/program_create_cache.md)
4. [AI/tutorials/program_forward_cache.md](../AI/tutorials/program_forward_cache.md)
5. [AI/tutorials/program_forward_indexer.md](../AI/tutorials/program_forward_indexer.md)
6. [AI/tutorials/cache_op.md](../AI/tutorials/cache_op.md)
7. [AI/tutorials/indexer_op.md](../AI/tutorials/indexer_op.md)
8. [vortex_torch/flow/algorithms.py](../vortex_torch/flow/algorithms.py)
9. [papers/guide.md](../papers/guide.md) — especially §14, §16
10. [algorithm_scientist/memory.md](../algorithm_scientist/memory.md)

If `memory.md §1` shows a batch RUNNING, skip to Step 5 (wait
activities) until it finishes, then resume the loop.

## Step 2 — set the batch index (once per loop iteration)

```bash
BATCH=$(ls submissions/$TAG/batch_*_id0.json 2>/dev/null | wc -l)
echo "next batch index: $BATCH"
```

## Step 3 — design the 4-variant batch

State the batch theme in one short paragraph. Define the knob
matrix — one row per variant (id0…id3).

Variant composition rule:
- **id0–id1 (aim for 2): genuinely novel** — §16.2 (untried knob),
  §16.3 (inversion), §16.4 (first-principles), or an op-set idea
  that doesn't fit any §16 sub-bucket. One sentence defending each,
  naming the specific op or behaviour exploited.
- **id2–id3: §16.5 parameter sweeps** — `vortex_topk_val`,
  `approxTopK` vs `topK`, layer-skip patterns, fp8/bf16 KV, etc.
  Explicitly encouraged for non-novelty slots.

Pre-register each novelty hypothesis as a one-sentence row in
`algorithm_scientist/memory.md` §3.

## Step 4 — write 8 files and pre-flight

Write `submissions/$TAG/batch_${BATCH}_id{0,1,2,3}.{py,json}`.
Each `.py` uses `@register("${TAG}_batch_${BATCH}_id<y>_cls")`.
Each `.json` sets `vortex_module_path` and `vortex_module_name`.

Pre-flight all 4 (CPU-only):
```bash
for y in 0 1 2 3; do
  python -c "from vortex_torch.engine.sgl import check_engine_config; \
check_engine_config('submissions/${TAG}/batch_${BATCH}_id${y}.json')" \
    && echo "[ok] id$y" || echo "[FAIL] id$y"
done
```
Fix any failure before continuing.

## Step 5 — RULER pre-filter (≥ 0.85)

Detect one free GPU, then run sequentially:
```bash
FREE_GPUS=($(algorithm_scientist/free_gpus.sh)) || {
  echo "no free GPUs — waiting"; exit 1
}
for y in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=${FREE_GPUS[0]} \
    python algorithm_scientist/run_ruler.py \
      --config "submissions/${TAG}/batch_${BATCH}_id${y}.json"
done
```
Any variant with `accuracy < 0.85` has broken attention — fix it
(widen `vortex_topk_val`/`vortex_topk_ratio` or revise the indexer),
re-pre-flight, and re-run RULER until all 4 pass.

## Step 6 — detect free GPUs and launch AIME24

Re-detect free GPUs immediately (set may have shifted):
```bash
FREE_GPUS=($(algorithm_scientist/free_gpus.sh)) || {
  echo "no free GPUs — hard wait"; exit 1
}
N=${#FREE_GPUS[@]}
BATCH_SIZE=4
PARALLEL=$N
[ "$PARALLEL" -gt "$BATCH_SIZE" ] && PARALLEL=$BATCH_SIZE
echo "free GPUs: ${FREE_GPUS[*]}  (N=$N, parallel=$PARALLEL)"
LOGDIR="logs/submission/${TAG}_batch_${BATCH}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"
for start in $(seq 0 $PARALLEL $((BATCH_SIZE - 1))); do
    end=$((start + PARALLEL))
    [ "$end" -gt "$BATCH_SIZE" ] && end=$BATCH_SIZE
    for y in $(seq $start $((end - 1))); do
        cfg="submissions/${TAG}/batch_${BATCH}_id${y}.json"
        gpu="${FREE_GPUS[$((y - start))]}"
        stem=$(basename "$cfg" .json)
        CUDA_VISIBLE_DEVICES=$gpu \
            python algorithm_scientist/run_submission_aime24.py \
              --config "$cfg" \
            > "$LOGDIR/gpu${gpu}_${stem}.out" \
            2> "$LOGDIR/gpu${gpu}_${stem}.err" &
    done
    wait
done
```
Add a row to `algorithm_scientist/memory.md` §1 the moment you launch:
`| $TAG | batch_$BATCH | <time> | $LOGDIR | batch_${BATCH}_id0…id3 | RUNNING |`

## Step 7 — wait (20–60 min) and do productive work

**Kill any child still running after 60 minutes** (`kill %<job>`),
log the error in memory.md §4, and treat that variant as failed.

On each polling cycle (check `jobs` for alive children), do ONE of:

(a) **Read.** Next file in priority order:
    `AI/tutorials/` → `AI/developer_guides/` → `papers/` →
    `vortex_torch/flow/algorithms.py` →
    `vortex_torch/{indexer,cache}/*` → `csrc/`.
    Append one insight bullet to memory.md §7.

(b) **Invent.** Pick a §16.2/§16.3/§16.4 prompt from
    `papers/guide.md` (NOT §16.1 combinations; NOT §16.5 sweeps —
    those fill slots but don't count as novelty). Sketch a
    one-sentence hypothesis naming the specific op or behaviour
    exploited. Aim for two sketches per wait cycle.

(c) **Design next batch.** Pre-flight all 4 candidates so they're
    ready the moment `wait` returns. Do NOT launch — concurrent
    batches OOM the shared GPUs.

(d) **Analyse children early.** Each child writes its
    `summary_submissions/$TAG/batch_${BATCH}_id<y>/latest.json`
    as soon as it finishes. Read landed summaries and start
    filling memory.md §2.

## Step 8 — analyse, update memory.md, check budget

Read all 4 `summary_submissions/$TAG/batch_${BATCH}_id<y>/latest.json`.
Produce a comparison table:

```
| variant       | content_hash | RULER acc | mean@16 | pass@16 | throughput | e2e_time |
```

Identify Pareto-non-dominated variants on the `(throughput, mean@16)`
plane. Append the table + 1–3 sentence takeaway to memory.md §2;
remove the §1 row; update §3 (hypotheses) / §4 (anti-patterns) /
§5 (winners).

**Check iteration budget:**
- Count batches launched this session =
  `ls submissions/$TAG/batch_*_id0.json | wc -l` minus the count
  at session start.
- If `MAX_ITER > 0` AND `batches_launched >= MAX_ITER`: write a
  final 2-paragraph summary (best submission name, content_hash,
  mean@16, throughput, key design decisions), update memory.md §8,
  and **stop**.
- Otherwise: go back to Step 2.
