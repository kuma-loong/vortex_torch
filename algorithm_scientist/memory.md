# Algorithm-scientist memory

Persistent working notebook for the vortex_torch sparse-attention
submission workflow. **Every agent run reads this file at start and
updates it before stopping.** It survives across sessions; the
conversation does not.

**Hard constraints** (restate every time you open this file):

- **Batch size is always 4 variants.** Parallelism = `min(N, 4)`
  where `N` = number of free GPUs detected by
  `algorithm_scientist/free_gpus.sh` (space-separated indices of
  GPUs with no compute process and memory.used < 1024 MiB).
  - `N >= 4` → all 4 variants run in parallel, one per GPU.
  - `0 < N < 4` → run the 4 variants in **waves of N**
    (sequential fallback). With `N = 1` this is fully serial;
    `N = 2` runs 2 + 2; `N = 3` runs 3 + 1.
  - `N == 0` → **hard wait**, do not launch.
  If you have only one core idea, fill the other 3 slots with
  orthogonal knob sweeps.
- **RULER pre-filter before AIME24.** Run
  `algorithm_scientist/run_ruler.py` on each variant
  sequentially on one free GPU before launching AIME24. Any
  variant scoring **< 0.85 accuracy** on
  `examples/validation.jsonl` has structurally broken attention
  — fix or replace it before launching AIME24.
- **One batch at a time** on the free GPUs. Concurrent batches
  contend for memory and OOM/thrash. The cadence is: detect free
  → run RULER filter → launch 4-variant batch → `wait` →
  analyse → re-detect → next batch.
- **At least one *genuinely novel* variant per batch** — aim for
  two when slots allow. "Genuinely novel" excludes both paper
  replications **and** combinations of two papers (combinations
  are catalog-adjacent — see `papers/guide.md` §16.1). Acceptable
  origins: `papers/guide.md` §16.2 (untried knobs), §16.3
  (inversions), §16.4 (first-principles), or — best — ideas
  derived from the framework's op set itself that don't fit any
  §16 sub-bucket. Defend each novelty in one sentence naming the
  specific framework op or behaviour exploited (not "combine
  paper A with paper B"). Pre-register each novelty hypothesis in
  §3 the moment you launch.
- **File layout.** Submissions live under `submissions/<tag>/`,
  where `<tag>` is your sanitized model name (e.g.
  `claude_opus_4_7`). Batched runs use
  `submissions/<tag>/batch_<x>_id<y>.{py,json}` (`<x>` = batch
  index, `<y>` = variant slot 0…3). Summaries land at
  `summary_submissions/<tag>/batch_<x>_id<y>/latest.json`; RULER
  results at
  `summary_ruler_submissions/<tag>/batch_<x>_id<y>/latest.json`.
- **Objective**: strike the best tradeoff between AIME24 `mean@16`
  and `throughput`. Both are objectives — there is no fixed quality
  floor. Pick winners by where they sit on the
  `(throughput, mean@16)` Pareto frontier in §5, not by clearing a
  fixed bar.

---

## §1. In-flight batches  (at most 1 — every targeted GPU is consumed)

> Append one row the moment you launch a batch; remove it once all
> 4 finished rows land in §2. A batch counts as "in-flight" from
> the first wave's launch until the last child writes its
> `latest.json`.

| tag | batch_x | launched_at | logdir | gpus | submissions | off-cat hyp § | status |
|---|---|---|---|---|---|---|---|
| _none_ |  |  |  |  |  |  |  |

---

## §2. Completed batches

> Oldest first. When a batch completes, copy headline metrics from
> `summary_submissions/<tag>/<stem>/latest.json` for each variant
> and distil 1-3 sentences of takeaway — what moved, what didn't,
> why. Always cite the off-catalog variant and what its result said
> about the §3 hypothesis it pre-registered.

<!--
### <tag>/batch_<x> — <one-line theme>  (launched YYYY-MM-DD HH:MM, free GPUs: 0,3,5,7)

| variant       | content_hash | RULER acc | mean@16 | pass@16 | throughput (tok/s) | e2e_time (s) | notes |
|---|---|---|---|---|---|---|---|
| batch_<x>_id0 |  |  |  |  |  |  |  |
| batch_<x>_id1 |  |  |  |  |  |  |  |
| batch_<x>_id2 |  |  |  |  |  |  |  |
| batch_<x>_id3 |  |  |  |  |  |  | ← off-catalog (§3 hyp #?) |

**Knob matrix:** id0=…, id1=…, id2=…, id3=… (off-catalog: …)

**Off-catalog hypothesis tested:** §3 row #? — verdict: confirmed/refuted/inconclusive.

**What moved:** …

**Takeaway:** …
-->

---

## §3. Design hypotheses

> Pre-register each novelty hypothesis here the moment its batch
> launches. Record the specific op or behaviour exploited, the
> predicted direction of effect, and the verdict after results land.

| # | hypothesis (one sentence, names op/behaviour) | §16 bucket | batch | verdict |
|---|---|---|---|---|
| — | _none yet_ | | | |

---

## §4. Anti-patterns

> One-liners of things that didn't work, so future iterations don't
> retry them. Include WHY it failed when known.

- _none yet_

---

## §5. Patterns that worked (Pareto frontier)

> Confirmed winners worth carrying forward. Each entry should give
> the submission name, its `(throughput, mean@16)` coordinates, and
> why it works. A new entry replaces an older one only if it
> dominates on both axes; otherwise both stay (they are on different
> parts of the frontier).

| submission | mean@16 | throughput (tok/s) | RULER acc | notes |
|---|---|---|---|---|
| _none yet_ | | | | |

---

## §6. Open questions / backlog

> Ideas and experiments that didn't fit the current batch but are
> worth trying. Pick from the top when a slot opens.

- _none yet_

---

## §7. Reading log

> Timestamped notes from reading tutorials, developer guides, source.
> One bullet per file, recording the single most useful insight.

- _none yet_

---

## §8. Session notes

> Per-session freeform, append-only. Record the date, what was
> attempted, and any context that doesn't fit the structured sections.

- _none yet_
