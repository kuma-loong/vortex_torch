---
description: Draft N novel sparse-attention submissions in one shot — algorithm-innovation focus, all must compile, no benchmark loop, no memory.md mutation.
argument-hint: <N> [theme-hint]
---

Generate **N novel submissions** in a single shot. This is the
*innovation-draft* mode — the complement to `/iterate`. There is
**no benchmark loop**, **no memory.md mutation**, and **no batch
sizing rule** (`N` is whatever the user passes, not capped to 4 or
the free-GPU count).

Every variant must satisfy two non-negotiable requirements:

1. **Genuinely novel algorithm.** Drawn from
   [papers/guide.md](../papers/guide.md) §16.2 (untried knobs),
   §16.3 (inversions), §16.4 (first-principles), or — best —
   from a hypothesis derived from the framework's own op set
   that doesn't fit any §16 sub-bucket. **Not** a paper replica,
   **not** a combination of two papers (those are §16.1
   catalog-adjacent and disqualified here), **not** a pure
   parameter sweep over an existing flow.
2. **Compiles.** Every variant must pass the local pre-flight
   (`check_engine_config`). A novel idea that does not compile is
   not an output of this mode — fix it or drop it before
   reporting completion.

The optional second argument is a free-form theme hint
(e.g. `page-level EMA tracking`, `channel-band gating`,
`Save/Load-state across decode steps`) that biases all N variants
toward a related family. Without a hint, draw freely from §16 or
the op set.

## Output layout

The N submission pairs land at:
```
submissions/<tag>/innovate_<x>_id<y>.py
submissions/<tag>/innovate_<x>_id<y>.json
```
for `y ∈ {0 … N-1}`, where `<tag>` is your session's agent
identifier and `<x>` is the next innovate-run index:
```bash
TAG=<your_agent_tag>           # sanitized model name
X=$(ls submissions/${TAG}/innovate_*_id0.json 2>/dev/null | wc -l)
```
The `innovate_<x>_id<y>` prefix is intentionally distinct from
the `batch_<x>_id<y>` prefix used by `/iterate` and
`/batch-benchmark`, so the two modes never collide on filenames
within the same tag.

## Steps

Step 0 — **validate args**:
```bash
NARGS=($ARGUMENTS)
N=${NARGS[0]:-}
[ -z "$N" ] && { echo "usage: /innovate <N> [theme-hint]" >&2; exit 1; }
case "$N" in *[!0-9]*) echo "N must be a positive integer" >&2; exit 1 ;; esac
[ "$N" -lt 1 ] && { echo "N must be >= 1" >&2; exit 1; }
THEME="${NARGS[@]:1}"   # may be empty
```

Step 1 — **resolve `<tag>` and run index `<x>`**, and create the
agent dir if needed:
```bash
mkdir -p "submissions/${TAG}"
X=$(ls submissions/${TAG}/innovate_*_id0.json 2>/dev/null | wc -l)
echo "innovate run: tag=${TAG} x=${X} N=${N} theme=\"${THEME}\""
```

Step 2 — **read the canonical context once** (skip files already
loaded this session):
- [AI/AGENTS.md](../AI/AGENTS.md) §1-§5 — contract + hard rules.
- [AI/tutorials/overview.md](../AI/tutorials/overview.md) and the
  five op/program tutorials.
- [papers/guide.md](../papers/guide.md) §16 — off-catalog prompts.
- [vortex_torch/flow/algorithms.py](../vortex_torch/flow/algorithms.py)
  — six reference flows for op patterns.

Step 3 — **for each variant `y ∈ {0 … N-1}`**, in one short
paragraph state:
- the **novelty hypothesis** (one sentence, naming the specific
  framework op or behaviour exploited — *not* "combine paper A
  with paper B"),
- the §16 sub-bucket or op-set thread it draws from,
- the cache fields it will declare and the indexer ops it will
  call.

If the user supplied a theme hint, every variant must connect to
it (different angles on the same family). Without a hint, the N
variants must be **orthogonal** — different mechanisms, not N
copies of the same idea.

Step 4 — **write each variant**. For each `y`:

`submissions/${TAG}/innovate_${X}_id${y}.py`:
- Globally-unique decorator: `@register("${TAG}_innovate_${X}_id${y}_cls")`.
- vFlow subclass implementing `create_cache` / `forward_cache` /
  `forward_indexer`.
- Use **only** `vortex_torch.indexer.*` and `vortex_torch.cache.*`
  ops. No native torch ops inside the three methods.
- Each op instance is a single call site (don't share).
- Don't declare `"k"` or `"v"` in `create_cache`.
- `forward_indexer` ends in `topK(score, o, ctx=ctx)` or
  `approxTopK(tolerate_ratio=...)(score, o, ctx=ctx)`; `score`
  must be ragged `[S, 1, 1]`.
- Cache-side reductions only on `dim ∈ {1, 2}`.
- Any field touched by `Load`/`Save` across steps must be
  `CFill(0.0)`-initialised in `forward_cache`.

`submissions/${TAG}/innovate_${X}_id${y}.json`:
- `vortex_module_path: "submissions/${TAG}/innovate_${X}_id${y}.py"`
- `vortex_module_name: "${TAG}_innovate_${X}_id${y}_cls"`
- `disable_radix_cache: true` **iff** `forward_indexer` uses
  `Save(...)` (the pre-flight enforces this).

Step 5 — **mandatory pre-flight gate**. Compile every variant:
```bash
FAILED=()
for y in $(seq 0 $((N - 1))); do
    cfg="submissions/${TAG}/innovate_${X}_id${y}.json"
    if python -c "from vortex_torch.engine.sgl import check_engine_config; check_engine_config('${cfg}')" 2>&1; then
        echo "[preflight] ok: ${cfg}"
    else
        echo "[preflight] FAIL: ${cfg}"
        FAILED+=("$y")
    fi
done
```
For every variant in `FAILED`: open the traceback, identify the
offending op or config field, fix it **in that variant's
.py/.json** (do *not* delete the variant — the goal is N
compiling submissions), and re-run the pre-flight on it. Repeat
until every variant passes.

If after a reasonable fix attempt a variant still cannot be made
to compile, surface the residual error to the user and stop —
do **not** silently report a partial run. The user can either
loosen the theme, lower `N`, or accept the partial output
explicitly.

Step 6 — **report**. Print a table with one row per variant:

```
| y | file                                              | bucket | hypothesis (≤1 sentence)                | preflight |
|---|---------------------------------------------------|--------|-----------------------------------------|-----------|
| 0 | submissions/<tag>/innovate_<x>_id0.{py,json}      | §16.2  | <one-sentence>                           | ok        |
| 1 | submissions/<tag>/innovate_<x>_id1.{py,json}      | op-set | <one-sentence>                           | ok        |
| … | …                                                 | …      | …                                       | …         |
```

Then **stop**. Do **not** call `/batch-benchmark`. Do **not**
touch `algorithm_scientist/memory.md`. Hand the user the N paths;
they decide whether to feed any of them into a future
`/batch-benchmark` (groups of 4) or `/iterate` run.

## What this mode is *not*

- **Not a benchmark.** No `run_submission_aime24.py` invocation;
  no GPU is touched.
- **Not iterative.** One shot, then return. No "wait + analyse +
  next batch" loop.
- **Not bound to batch=4.** `N` is the user's choice — useful
  values are typically 2-12 depending on the theme's surface area.
- **Not memory.md-aware.** Does not read or write the persistent
  notebook. (If the user wants a draft to feed into the iterate
  loop later, they can copy it from `submissions/<tag>/` into a
  `batch_<x>_id<y>` slot themselves.)

## Hard refusals

- `N` missing, non-numeric, or `< 1` → reject with `usage:
  /innovate <N> [theme-hint]`.
- A drafted variant produces only catalog-adjacent ideas (paper
  combinations, parameter sweeps over `example_*`) → reject the
  variant and re-draft it with a §16.2/§16.3/§16.4 or op-set
  hypothesis. Do **not** report the run as complete with a
  catalog-adjacent variant in the output.
- A variant cannot be made to pre-flight after a fix attempt →
  surface the error and stop the entire run.
