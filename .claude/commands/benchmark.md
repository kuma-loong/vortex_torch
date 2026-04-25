---
description: (DEBUG ONLY) sbatch a single-variant AIME24 run. Forbidden in the standard agent workflow — use /batch-benchmark instead.
argument-hint: <submission-name>
---

> **Debug-only command.** The sanctioned benchmark protocol runs
> exclusively in batches of 8 via `/batch-benchmark`. Use this
> single-variant form ONLY for human debugging (e.g. confirming a
> new flow boots end-to-end). Do not chain it as part of an
> automated workflow.

Submit `submissions/$1.json` to the AIME24 benchmark via Slurm.

**Do not run `python algorithm_scientist/run_submission_aime24.py` directly** —
the host environment has no GPU. Use Slurm:

Step 1 — submit:
```bash
sbatch algorithm_scientist/run_submission.slurm submissions/$1.json
```
Capture the returned `Submitted batch job <JOBID>` and echo the
JOBID back to the user.

Step 2 — poll every ~60s until the job reaches a terminal state
(`COMPLETED` / `FAILED` / `TIMEOUT` / `CANCELLED` / `OUT_OF_MEMORY`):
```bash
sacct -j <JOBID> --format=JobID,State,ExitCode -X -n -P
```

Step 3 — once terminal:

- **Success** → Read `summary_submissions/$1/latest.json` (a
  symlink to the newest run). Print `mean@16`, `pass@16`,
  `e2e_time`, `total_tokens`, `throughput`, and the
  `content_hash`. For run-to-run comparisons, `cat
  summary_submissions/$1/INDEX.jsonl | jq -c '{finished_at,
  content_hash, "mean@16", throughput}'`.
- **Failure** → Read `logs/submission/vortex_submission_<JOBID>.err`
  (and `.out` if needed). Summarise the error in 1-2 sentences and
  recommend a fix. Do NOT re-submit without pre-flight passing first.
