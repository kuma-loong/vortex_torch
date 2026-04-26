# `algorithm_scientist/` — self-contained bundle for agentic submissions

Everything you need to turn a fresh idea into an AIME24 score, in one
folder. All four files here are ready to run from the repo root
(`vortex_torch`) and reference each other using
`algorithm_scientist/…` paths.

## What lives here

| File | Role |
|---|---|
| [`auto_agent.py`](auto_agent.py) | Drives Claude through an agentic loop (Anthropic SDK). Reads `AI/AGENTS.md` + the six tutorials as a cached system prompt, writes `submissions/<name>.{py,json}`, runs the local pre-flight, and sbatches the AIME24 benchmark. |
| [`run_submission_aime24.py`](run_submission_aime24.py) | The fixed-protocol AIME24 runner (16 trials, Qwen3-4B, tp=1). Writes per-submission summaries to `summary_submissions/<name>/<timestamp>__<hash>.json`, with a `latest.json` symlink and an `INDEX.jsonl` index. |
| [`run_submission.slurm`](run_submission.slurm) | Single-variant Slurm job — 1 GPU, one submission. |
| [`run_submission_batch.slurm`](run_submission_batch.slurm) | Batched Slurm job — whole 8-GPU node, backgrounds up to 8 submissions with distinct `CUDA_VISIBLE_DEVICES=0..7`. |

## Authoritative instructions

The text instructions the agent reads remain in [`AI/`](../AI/):

- [`AI/AGENTS.md`](../AI/AGENTS.md) — submission contract, rules, benchmark protocol, objective.
- [`AI/tutorials/`](../AI/tutorials/) — six user-facing tutorials.
- [`AI/developer_guides/`](../AI/developer_guides/) — framework-internal deep dives.

The `.claude/` folder at the repo root provides slash commands
(`/new-submission`, `/preflight`, `/benchmark`, `/batch-benchmark`,
`/review`, `/iterate`) that wire these scripts into a Claude Code
session.

## Usage

From the repo root:

### Single variant

```bash
sbatch algorithm_scientist/run_submission.slurm submissions/<name>.json
```

### 2-8 variants in parallel on one 8-GPU node

```bash
sbatch algorithm_scientist/run_submission_batch.slurm \
    submissions/v1.json submissions/v2.json … submissions/v8.json
```

### Autonomous agent

```bash
export ANTHROPIC_API_KEY=…
python algorithm_scientist/auto_agent.py \
    --submission-name my_v1 \
    --min-mean-at-16 0.09 \
    --baseline-throughput 11962 \
    --max-iterations 5
```

The agent writes submission pairs, pre-flights them, sbatches the
benchmark, reads `summary_submissions/<name>/latest.json`, and
iterates. Event log at `logs/auto_agent/<name>_<ts>.jsonl`.

## Relation to `examples/` and `experiments/`

The originals under [`examples/run_submission_aime24.py`](../examples/run_submission_aime24.py),
[`examples/auto_agent.py`](../examples/auto_agent.py),
[`experiments/run_submission.slurm`](../experiments/run_submission.slurm),
[`experiments/run_submission_batch.slurm`](../experiments/run_submission_batch.slurm)
are retained for backwards compatibility and are kept in sync with the
copies here. When you edit one, mirror the change in the other, or
delete the duplicates you don't use.
