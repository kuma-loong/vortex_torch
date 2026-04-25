---
description: Run check_engine_config locally (CPU-only) on a submission.
argument-hint: <submission-name>
---

Run the cheap pre-flight check for `submissions/$1.json`:

```bash
python -c "from vortex_torch.engine.sgl import check_engine_config; check_engine_config('submissions/$1.json')"
```

Capture the output. If it raises, paste the relevant traceback line
and cross-reference it with [AI/AGENTS.md §5](AI/AGENTS.md)
"Common failures" table. Suggest the minimum edit needed to make it
pass, but do **not** edit files unless the user asks you to.

If it passes, remind the user that the next step is `/benchmark $1`
(which submits to Slurm — no GPU needed here).
