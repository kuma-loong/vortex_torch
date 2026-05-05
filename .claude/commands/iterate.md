---
description: Launch the long-horizon auto-iteration loop (4 variants per batch, parallel across free GPUs when N>=4 else sequential waves, one batch at a time, persists state in memory.md).
argument-hint: <theme-tag> [--max-iterations <int>] [--baseline-throughput <float>]
---

Kick off a full auto-iteration loop tagged **$1** using
[algorithm_scientist/auto_agent.py](algorithm_scientist/auto_agent.py).
This drives a dedicated Claude agent that:

- Reads `AI/AGENTS.md` + the six tutorials as a cached system prompt.
- Reads / appends to `algorithm_scientist/memory.md` for persistent
  state across sessions (in-flight ledger, completed-batch table,
  hypotheses, anti-patterns, reading log).
- Detects the *currently-free* GPU set at the start of every
  batch via `algorithm_scientist/free_gpus.sh` (NOT the physical
  GPU count — other users may share this host). **Every batch is
  exactly 4 variants**; parallelism is `min(N, 4)`. With `N >= 4`
  the 4 variants run in parallel (one per GPU); with `0 < N < 4`
  they run in **waves of `N`** on the available GPUs (sequential
  fallback). Only `N == 0` triggers a hard wait. Writes submission
  pairs `submissions/<tag>/batch_<x>_id<y>.{py,json}` for
  `<y> ∈ {0, 1, 2, 3}`, where `<tag>` is the agent's identifier
  and `<x>` is the batch index (0-indexed, increments per batch
  this session). The positional `$1` argument is the *theme tag*
  for memory.md attribution; it is **not** the agent tag (the
  agent tag is auto-detected from the model name).
- Pre-flights all 4 variants locally, then runs them in waves of
  `min(N, 4)` background `python algorithm_scientist/run_submission_aime24.py`
  processes pinned to the free GPU indices
  (`CUDA_VISIBLE_DEVICES=${FREE_GPUS[$((y - start))]}`) with
  `wait` between waves. Re-detects free GPUs immediately before
  launch in case the set shifted during pre-flight.
- Honours the concurrency cap: **one** batch at a time on the
  GPUs it targets; if `free_gpus.sh` returns nothing, hard wait.
- During the 8+ hr wait, alternates between reading source,
  designing the next batch (without launching), and analysing
  children whose `latest.json` has already landed.
- Reads `summary_submissions/<name>/latest.json` and updates
  memory.md.

```bash
ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:?ANTHROPIC_API_KEY must be set} \
  python algorithm_scientist/auto_agent.py --submission-name $1 $ARGUMENTS
```

(`$ARGUMENTS` forwards every extra flag the user supplies, e.g.
`--max-iterations 12 --baseline-throughput 5500`.)

Tail the live log printed to stdout. Every turn, tool call, and
tool result is also persisted to `logs/auto_agent/$1_*.jsonl`.
