# Algorithm-scientist memory

Persistent working notebook for the vortex_torch sparse-attention
submission workflow. **Every agent run reads this file at start and
updates it before stopping.** It survives across sessions; the
conversation does not.

**Hard constraints** (restate every time you open this file):

- Submissions **always run in a batch sized to the free local GPUs,
  with a hard minimum of `N = 4`**. Detect with
  `algorithm_scientist/free_gpus.sh` (returns space-separated
  indices of GPUs with no compute process and memory.used <
  1024 MiB); `N = ${#FREE_GPUS[@]}`. **If `N < 4`, hard wait** —
  do not launch a 1/2/3-variant batch. Below 4 the batch loses the
  analytical width needed for orthogonal-knob sweeps and
  Pareto-frontier mapping. If you have only one core idea, fill
  the other `N - 1` slots with orthogonal knob sweeps.
- **One batch at a time** on the free GPUs. Concurrent batches contend
  for memory and OOM/thrash. The cadence is: detect free → launch
  N-variant batch → `wait` → analyse → re-detect → next batch.
- **At least one *genuinely novel* variant per batch** — and aim
  for two when `N >= 6`. "Genuinely novel" excludes both paper
  replications **and** combinations of two papers (combinations
  are catalog-adjacent — see `papers/guide.md` §16.1). Acceptable
  origins: `papers/guide.md` §16.2 (untried knobs), §16.3
  (inversions), §16.4 (first-principles), or — best — ideas
  derived from the framework's op set itself that don't fit any
  §16 sub-bucket. Defend each novelty in one sentence naming the
  specific framework op or behaviour exploited (not "combine
  paper A with paper B"). Pre-register each novelty hypothesis in
  §3 the moment you launch.
- **File layout.** Submissions live under `submissions/<tag>/`, where
  `<tag>` is your sanitized model name (e.g. `claude_opus_4_7`).
  Batched runs use `submissions/<tag>/batch_<x>_id<y>.{py,json}`
  (`<x>` = batch index, `<y>` = variant slot 0…N-1). Summaries land
  at `summary_submissions/<tag>/batch_<x>_id<y>/latest.json` (the
  runner mirrors the source path).
- **Objective**: strike the best tradeoff between AIME24 `mean@16`
  and `throughput`. Both are objectives — there is no fixed quality
  floor. Pick winners by where they sit on the
  `(throughput, mean@16)` Pareto frontier in §5, not by clearing a
  fixed bar.

---

## 1. In-flight batches  (at most 1 — every targeted GPU is consumed)

> Append one row the moment you launch a batch; remove it once all
> `N` finished rows land in §2. A batch counts as "in-flight" from
> the `wait` start until the last child writes its `latest.json`.

| tag | batch_x | launched_at | logdir | gpus | submissions | off-cat hyp § | status |
|---|---|---|---|---|---|---|---|
| _none_ |  |  |  |  |  |  |  |

---

## 2. Completed batches

> Oldest first. When a batch completes, copy headline metrics from
> `summary_submissions/<tag>/<stem>/latest.json` for each variant
> and distil 1-3 sentences of takeaway — what moved, what didn't,
> why. Always cite the off-catalog variant and what its result said
> about the §3 hypothesis it pre-registered.

<!--
### <tag>/batch_<x> — <one-line theme>  (launched YYYY-MM-DD HH:MM, free GPUs: 0,3,5,7)

| variant         | content_hash | mean@16 | pass@16 | throughput (tok/s) | e2e_time (s) | notes |
|---|---|---|---|---|---|---|
| batch_<x>_id0   |  |  |  |  |  |  |
| batch_<x>_id1   |  |  |  |  |  |  |
| …               |  |  |  |  |  |  |
| batch_<x>_idN-1 |  |  |  |  |  | ← off-catalog (§3 hyp #?) |

**Knob matrix:** id0=…, id1=…, …, idN-1=… (off-catalog: …)

**Off-catalog hypothesis tested:** §3 row #? — verdict: confirmed/refuted/inconclusive.

**What moved:** …

**Takeaway:** …
-->
