# Algorithm-scientist memory

Persistent working notebook for the vortex_torch sparse-attention
submission workflow. **Every agent run reads this file at start and
updates it before stopping.** It survives across sessions; the
conversation does not.

**Hard constraints** (restate every time you open this file):

- Submissions **always run in batches of 8** via
  `sbatch algorithm_scientist/run_submission_batch.slurm …`. Never
  submit a single variant — if you have only one idea, fill the other
  seven slots with orthogonal knob sweeps.
- **At most 24 experiments in flight** at once (i.e. ≤ 3 batches
  queued+running simultaneously). Check `squeue -u $USER -h` before
  each `sbatch`.
- **Objective**: maximise AIME24 `throughput` while keeping `mean@16`
  at or above the quality floor set for this session.

---

## 1. In-flight batches  (≤ 3, equivalent to ≤ 24 experiments)

> Append one row when a batch is submitted; remove it once all 8
> finished rows land in §2. A batch counts against the ceiling from
> `sbatch` until the last child in the batch lands a summary JSON.

| batch_id | submitted_at | slurm_job_id | submissions (8) | status |
|---|---|---|---|---|
| _none yet_ |  |  |  |  |

---

## 2. Completed batches

> Oldest first. When a batch completes, copy headline metrics from
> `summary_submissions/<name>/latest.json` and distil 1-3 sentences
> of takeaway — what moved, what didn't, why.

<!--
### Batch <id> — <one-line label>  (submitted YYYY-MM-DD HH:MM, slurm <JOBID>)

| submission | content_hash | mean@16 | pass@16 | throughput (tok/s) | e2e_time (s) |
|---|---|---|---|---|---|
| v1 |  |  |  |  |  |
| v2 |  |  |  |  |  |
| …  |  |  |  |  |  |
| v8 |  |  |  |  |  |

**Hypothesis this batch tested:** …

**What moved:** …

**Takeaway:** …
-->

_none yet_

---

## 3. Design hypotheses (running)

> Keep one row per idea under investigation. Close a row with a
> verdict when you have enough evidence. A refuted hypothesis is
> just as valuable as a confirmed one — record it.

| hypothesis | evidence for (batch ids) | evidence against | verdict |
|---|---|---|---|
| _example: "fp8_e5m2 kv cache keeps mean@16 within 5% of bf16 on AIME24"_ | _e.g. batch 3, 5_ | _e.g. batch 4_ | _open / confirmed / refuted_ |

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

> The agent's own backlog of ideas it hasn't tested. When a slot
> opens up (< 24 in flight), pick from the top of this list.

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
