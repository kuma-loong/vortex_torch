---
description: Launch the long-horizon auto-iteration loop (batches of 8, ≤ 24 in flight, persists state in memory.md).
argument-hint: <theme-tag> [--min-mean-at-16 <float>] [--max-iterations <int>] [--baseline-throughput <float>]
---

Kick off a full auto-iteration loop tagged **$1** using
[algorithm_scientist/auto_agent.py](algorithm_scientist/auto_agent.py).
This drives a dedicated Claude agent that:

- Reads `AI/AGENTS.md` + the six tutorials as a cached system prompt.
- Reads / appends to `algorithm_scientist/memory.md` for persistent
  state across sessions (in-flight ledger, completed-batch table,
  hypotheses, anti-patterns, reading log).
- Writes submission pairs `submissions/$1_v1..v8.{py,json}`.
- Pre-flights all 8 locally, then sbatches the **batch of 8** via
  `algorithm_scientist/run_submission_batch.slurm`.
- Honours the in-flight ceiling: ≤ 3 concurrent batches (≤ 24
  experiments).
- During Slurm's 8+ hr wait, alternates between reading source,
  preparing the next batch, and analysing completed batches.
- Reads `summary_submissions/<name>/latest.json` and updates
  memory.md.

```bash
ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:?ANTHROPIC_API_KEY must be set} \
  python algorithm_scientist/auto_agent.py --submission-name $1 $ARGUMENTS
```

(`$ARGUMENTS` forwards every extra flag the user supplies, e.g.
`--min-mean-at-16 0.62 --baseline-throughput 5500`.)

Tail the live log printed to stdout. Every turn, tool call, and
tool result is also persisted to `logs/auto_agent/$1_*.jsonl`.
