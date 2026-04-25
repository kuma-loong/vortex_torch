---
description: Submit exactly 8 submissions in parallel on a single 8-GPU node (the only sanctioned benchmark form).
argument-hint: <name1> <name2> <name3> <name4> <name5> <name6> <name7> <name8>
---

Submit **exactly 8 submissions** to the AIME24 benchmark as one
batched Slurm job. This is the *only* benchmark command sanctioned
by the protocol — single-variant runs are debug-only.

The user passes **submission names** (not JSON paths); this command
expands each `<nameI>` into `submissions/<nameI>.json`.

Step 0 — count arguments. If the user supplied fewer than 8 names,
refuse and explain: "Batches must contain exactly 8 variants. Fill
the remaining slots with orthogonal knob sweeps (different topk,
kv_dtype, layers_skip, block_size, …) before re-invoking." Do
not silently shrink the batch.

Step 1 — check the in-flight ceiling (≤ 24 experiments, i.e. ≤ 3
concurrent batches):
```bash
squeue -u $USER -h -o '%j' | grep -c '^vortex_submission_batch$'
```
If the count is already 3, refuse to submit — tell the user to
wait for one to finish (or to use the wait-time activities in
`algorithm_scientist/memory.md`).

Step 2 — pre-flight every config locally first (cheap, no GPU):
```bash
for name in $ARGUMENTS; do
    python -c "from vortex_torch.engine.sgl import check_engine_config; check_engine_config('submissions/${name}.json')" \
        || echo "[preflight] FAILED: ${name}"
done
```
Refuse to sbatch any submission whose preflight failed — fix the
failing variant first.

Step 3 — build the argument list and submit (exactly 8 paths):
```bash
CFGS=()
for name in $ARGUMENTS; do
    CFGS+=("submissions/${name}.json")
done
sbatch algorithm_scientist/run_submission_batch.slurm "${CFGS[@]}"
```
Capture the returned `Submitted batch job <JOBID>`.

Step 4 — append a row to `algorithm_scientist/memory.md` §1
*In-flight batches*:
`| <batch_id> | <UTC time> | <JOBID> | <name1>,…,<name8> | PENDING |`

Step 5 — poll:
```bash
sacct -j <JOBID> --format=JobID,State,ExitCode -X -n -P
```
Each batch may run **8+ hours**. While polling, do NOT idle —
spend the time reading tutorials / developer guides / source, or
designing the next batch (subject to the ≤ 3 in-flight ceiling),
or analysing earlier completed batches. See AGENTS.md §5d.

Step 6 — once terminal:

- **Success** → for each `<nameI>`, read
  `summary_submissions/<nameI>/latest.json`. Produce a comparison
  table with columns: `name | content_hash | mean@16 | pass@16 |
  throughput | e2e_time`. Recommend the one with the highest
  `throughput` **that also clears the quality floor**. Append the
  table + a 1-3 sentence takeaway to `algorithm_scientist/memory.md`
  §2 *Completed batches*; remove the row you added in step 4 from §1.
- **Failure** → any child that failed wrote its traceback to
  `logs/submission/batch_<JOBID>/gpu<i>_<stem>.err`. Open the
  failing child's `.err`, summarise the error, and record the
  lesson in `algorithm_scientist/memory.md` §4 *Anti-patterns*.
