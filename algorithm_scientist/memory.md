# Algorithm-scientist memory

Persistent working notebook for the vortex_torch sparse-attention
submission workflow. **Every agent run reads this file at start and
updates it before stopping.** It survives across sessions; the
conversation does not.

**Hard constraints** (restate every time you open this file):

- Submissions **always run in a batch sized to the free local GPUs**.
  Detect with `algorithm_scientist/free_gpus.sh` (returns space-
  separated indices of GPUs with no compute process and
  memory.used < 1024 MiB); `N = ${#FREE_GPUS[@]}`. Never launch a
  single variant — if you have only one idea, fill the other `N - 1`
  slots with orthogonal knob sweeps.
- **One batch at a time** on the free GPUs. Concurrent batches contend
  for memory and OOM/thrash. The cadence is: detect free → launch
  N-variant batch → `wait` → analyse → re-detect → next batch.
- **At least one off-catalog variant per batch** (`papers/guide.md §16`):
  paper combinations, knob inversions, untried-knob experiments, or
  first-principles answers. Pure parameter sweeps and direct paper
  replications do not count. Pre-register the off-catalog hypothesis
  in §3 the moment you launch.
- **File layout.** Submissions live under `submissions/<tag>/`, where
  `<tag>` is your sanitized model name (e.g. `claude_opus_4_7`).
  Batched runs use `submissions/<tag>/batch_<x>_id<y>.{py,json}`
  (`<x>` = batch index, `<y>` = variant slot 0…N-1). Summaries land
  at `summary_submissions/<tag>/batch_<x>_id<y>/latest.json` (the
  runner mirrors the source path).
- **Objective**: maximise AIME24 `throughput` while keeping `mean@16`
  at or above the quality floor set for this session.

---

## 1. In-flight batches  (at most 1 — every targeted GPU is consumed)

> Append one row the moment you launch a batch; remove it once all
> `N` finished rows land in §2. A batch counts as "in-flight" from
> the `wait` start until the last child writes its `latest.json`.

| tag | batch_x | launched_at | logdir | gpus | submissions | off-cat hyp § | status |
|---|---|---|---|---|---|---|---|
| _none yet_ |  |  |  |  |  |  |  |

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

_none yet_

---

## 3. Design hypotheses (running)

> Keep one row per idea under investigation. Close a row with a
> verdict when you have enough evidence. A refuted hypothesis is
> just as valuable as a confirmed one — record it. Number rows so
> the §1 "off-cat hyp §" column and §2 "off-catalog hypothesis"
> notes can cite them.

| # | hypothesis | source | evidence for (batches) | evidence against | verdict |
|---|---|---|---|---|---|
| _ex_ | _"fp8_e5m2 kv cache keeps mean@16 within 5% of bf16 on AIME24"_ | _§16.4 / Double Sparsity §4_ | _e.g. <tag>/batch_3, batch_5_ | _e.g. batch_4_ | _open / confirmed / refuted_ |

---

## 4. Anti-patterns (things tried that did not work)

> One line each. Brief. Future-you will thank you for the crisp list.

- _none yet_

---

## 5. Patterns that worked (reuse these)

> Confirmed throughput / accuracy wins worth carrying into every
> future submission.

- _none yet_

---

## 6. Open questions / next directions

> The agent's own backlog of ideas it hasn't tested. When the
> current batch finishes (no in-flight rows in §1), pick from the
> top of this list. At least one slot in every new batch must be
> off-catalog — promote items here into pre-registered §3
> hypotheses on launch.

- _none yet_

---

## 7. Reading log

> Timestamped notes from tutorials / developer_guides / source code.
> One bullet per file read, with the single most useful insight.

- _none yet_

---

## 8. Session notes

> Freeform per-session summary: what the agent did, what was surprising,
> what's left for next time. Append — do not overwrite.

- _none yet_
