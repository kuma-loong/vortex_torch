"""Generate a ``.claude/`` folder at the repo root so the vortex_torch
submission workflow drops straight into Claude Code.

The authoritative instructions live in ``AI/AGENTS.md`` and
``AI/tutorials/``. This script produces a thin set of wrappers
(``CLAUDE.md``, subagents, slash commands) that make those
instructions reachable from within an interactive Claude Code
session — without duplicating their content.

Layout produced:

    .claude/
        CLAUDE.md                        — project primer (always loaded)
        agents/
            vortex-submission-writer.md  — subagent: write / iterate a flow
            vortex-submission-reviewer.md — subagent: audit a flow
        commands/
            new-submission.md            — /new-submission <name>
            preflight.md                 — /preflight <name>
            benchmark.md                 — /benchmark <name>
            review.md                    — /review <name>
            iterate.md                   — /iterate <name>

Re-running the script overwrites the generated files. Hand-edit only
the inputs below; never edit the emitted files directly (they will be
clobbered on the next run).

Usage::

    python AI/generate_claude_folder.py
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent


REPO_ROOT = Path(__file__).resolve().parent.parent
CLAUDE_DIR = REPO_ROOT / ".claude"


# ---------------------------------------------------------------------------
# CLAUDE.md — project primer (always loaded into every Claude Code session)
# ---------------------------------------------------------------------------

CLAUDE_MD = dedent("""\
    # vortex_torch — project primer for Claude Code

    `vortex_torch` is a JIT-compiled sparse-attention framework that plugs
    into sglang's decode loop. Users / AI agents write a **sparse-attention
    submission** — two files in `submissions/` — and the framework compiles
    them into Triton kernels at runtime.

    ## Objective (non-negotiable)

    > **Maximise decoding throughput (tokens/sec) on the AIME24 benchmark
    > while keeping `mean@16` at or above a minimum acceptable floor.**

    `mean@16` is a *quality gate*, not something to maximise. Once a flow
    clears the floor, every subsequent change should trade accuracy-headroom
    for **more throughput**: tighter `vortex_topk_val` / `vortex_topk_ratio`,
    fewer cache-side ops, fewer intermediate cache fields, narrower
    `vortex_layers_skip`, aggressive fp8 `kv_cache_dtype`, raise
    `mem_fraction_static` toward 0.9 (range [0.5, 0.95], default 0.8 —
    higher = more KV-cache headroom = more throughput, but OOM risk),
    swap `topK()` for `approxTopK(tolerate_ratio=…)` (`0.0`=exact,
    higher=cheaper-but-looser; sweet spot 0.05-0.15). When two
    variants both clear the floor, pick the faster one.

    ## Inventing beyond the literature

    The `papers/` folder and [papers/guide.md](papers/guide.md) cover
    what's already published — sinks, heavy hitters, channel sparsity,
    low-rank K, LSH sampling, dual-band centroids. Treat them as
    **seeds, not a menu.** A winning flow does not need a citation.
    Every paper in there started by noticing a gap; the framework you
    have (page-level selection, fused per-block kernel, Save/Load,
    `Kron`, `MeanInterleave`) opens combinations and knobs no paper
    here has explored. **Every batch must reserve at least one slot
    for an off-catalog variant** — see `papers/guide.md` §16 for
    prompts (paper combinations, knobs nobody has tried, claims worth
    inverting, first-principles questions). Replicating a paper or
    sweeping a single knob does not count.

    ## Where the instructions live

    All authoritative content lives under [AI/](AI/). Read in order:

    1. [AI/AGENTS.md](AI/AGENTS.md) — the full submission contract, rules,
       budget / BOS / layer-skip semantics, benchmark protocol.
    2. [AI/tutorials/overview.md](AI/tutorials/overview.md) — 5-minute map.
    3. [AI/tutorials/program_create_cache.md](AI/tutorials/program_create_cache.md)
    4. [AI/tutorials/program_forward_cache.md](AI/tutorials/program_forward_cache.md)
    5. [AI/tutorials/program_forward_indexer.md](AI/tutorials/program_forward_indexer.md)
    6. [AI/tutorials/cache_op.md](AI/tutorials/cache_op.md) — indexer-side
       op math reference.
    7. [AI/tutorials/indexer_op.md](AI/tutorials/indexer_op.md) — cache-side
       op math reference.
    8. [papers/guide.md](papers/guide.md) — synthesis of the ten
       sparse-attention papers in `papers/`. §14 = catalog of
       known-good submission ideas; **§16 = prompts for inventing
       flows that no paper here explores.**

    Framework-internal deep dives live in
    [AI/developer_guides/](AI/developer_guides/) — needed only if you are
    modifying the compiler itself, not when writing a submission.

    ## Hard constraints

    - **No native torch ops** inside `create_cache` / `forward_cache` /
      `forward_indexer`. Every tensor goes through
      `vortex_torch.indexer.*` / `vortex_torch.cache.*` ops. `.view`,
      `.sum(dim=...)`, elementwise torch, etc. will not compile.
    - **Each op instance is one call site.** `self.mul_a = Multiply()`
      and `self.mul_b = Multiply()` — do not share.
    - **Do not declare `"k"` or `"v"`** in `create_cache`; they are
      auto-provided.
    - **`forward_indexer` must end in `topK(score, o, ctx=ctx)` or
      `approxTopK(tolerate_ratio=...)(score, o, ctx=ctx)`** — the
      score must be RAGGED `[S, 1, 1]`. `approxTopK` is a faster
      adaptive 8-bit radix variant; `tolerate_ratio ∈ [0.0, 1.0]`
      where `0.0` = exact, higher = cheaper but looser.
    - **Cache-side reductions support `dim ∈ {1, 2}` only.** Cross-block
      reductions (`dim=0`) belong on the indexer side.
    - **If a field is read+written across steps via `Load`/`Save`, zero
      it with `CFill(0.0)` in `forward_cache`.**
    - **If `forward_indexer` uses `Save(...)`, the engine JSON MUST set
      `"disable_radix_cache": true`** (default `false`). sglang's
      prefix-radix cache otherwise shares per-request Save'd state
      across requests with matching prompt prefixes, corrupting
      Save/Load values. `check_engine_config` rejects the violation.

    ## Running the benchmark — policy

    **The only allowed unit of work is one batch that fills every
    local GPU.** Detect the GPU count at the start of every batch and
    treat that number as `N`:

    ```bash
    NUM_GPUS=$(nvidia-smi -L | wc -l)   # or: python -c "import torch; print(torch.cuda.device_count())"
    ```

    **File layout.** All submissions you write live under
    `submissions/<tag>/`, where `<tag>` is your agent identifier
    (default: a sanitized lowercase form of your model name, e.g.
    `claude_opus_4_7`). Within that dir, batched runs use the
    convention `batch_<x>_id<y>.{py,json}` (`<x>` = batch index,
    `<y>` = per-GPU variant index). Single-variant runs are
    debug-only. Each batch:

    1. **`N` orthogonal variants** —
       `submissions/<tag>/batch_<x>_id0.{py,json}` …
       `submissions/<tag>/batch_<x>_id<N-1>.{py,json}` — varying
       different knobs.
    2. **Cheap local pre-flight first** for all `N` (CPU, no GPU):
       ```bash
       TAG=<your_agent_tag>; BATCH=<batch_index>
       for i in $(seq 0 $((NUM_GPUS - 1))); do
         python -c "from vortex_torch.engine.sgl import check_engine_config; check_engine_config('submissions/${TAG}/batch_${BATCH}_id${i}.json')"
       done
       ```
       Refuse to launch any variant whose pre-flight fails.
    3. **Launch `N` background `python` processes**, one per GPU
       `0 … N-1`, and `wait` for them all to finish:
       ```bash
       LOGDIR="logs/submission/${TAG}_batch_${BATCH}_$(date +%Y%m%d_%H%M%S)"
       mkdir -p "$LOGDIR"
       for i in $(seq 0 $((NUM_GPUS - 1))); do
           cfg="submissions/${TAG}/batch_${BATCH}_id${i}.json"
           stem=$(basename "$cfg" .json)
           CUDA_VISIBLE_DEVICES=$i \\
               python algorithm_scientist/run_submission_aime24.py --config "$cfg" \\
               > "$LOGDIR/gpu${i}_${stem}.out" \\
               2> "$LOGDIR/gpu${i}_${stem}.err" &
       done
       wait
       ```
       Each child writes its result into
       `summary_submissions/<tag>/<stem>/<timestamp>__<hash>.json`
       and updates `latest.json` on its own. The runner mirrors the
       config's path under `submissions/` into `summary_submissions/`,
       so `submissions/<tag>/batch_<x>_id<y>.json` becomes
       `summary_submissions/<tag>/batch_<x>_id<y>/...` — per-agent
       isolation, no collisions between agents that happen to use
       the same `batch_x_idy` stem.

    4. **One batch at a time.** Every local GPU is consumed by the
       running batch — do not launch a second batch in parallel.
       Use `jobs` (or `ls -lt summary_submissions/<tag>/*/latest.json`)
       to see how many children are still alive while you wait.

    ## While you wait (8+ hrs per batch)

    Idle is not an option. Each polling cycle, do one of:

    - **Read.** Priority: `AI/tutorials/` → `AI/developer_guides/` →
      `papers/` → `vortex_torch/flow/algorithms.py` →
      `vortex_torch/{indexer,cache}/*` → `csrc/`. After each file,
      append one insight to `algorithm_scientist/memory.md` §7.
    - **Invent.** Open `papers/guide.md` §16 and pick one prompt —
      a paper combination, a knob no paper has tried, a claim
      worth inverting, or a first-principles question. Sketch a
      one-sentence hypothesis + cache/indexer ops. This fills the
      mandatory off-catalog slot in the next batch.
    - **Design (don't launch) the rest of the next batch.**
      Pre-flight the `N` candidates so they're ready to fire the
      moment `wait` returns. Concurrent batches would OOM the
      shared GPUs.
    - **Analyse children early.** As individual `latest.json` files
      appear (children finish at slightly different times), pull
      their `mean@16` / `throughput` and start filling a §2
      sub-section in memory.md. Close the §1 row when all `N` are
      in, then update §3 (hypotheses) / §4 (anti-patterns) / §5
      (winners).

    ## Persistent state — `algorithm_scientist/memory.md`

    The conversation evaporates; `memory.md` does not. Read it at the
    start of every session and write to it before stopping. Any batch
    submission and any batch completion must mutate it.
       When the job finishes, each run is written into a
       per-submission subfolder so iterations never collide:

       ```
       summary_submissions/<name>/
           <timestamp>__<content_hash>.json   # full summary + embedded .py/.json
           latest.json                        # symlink → newest run
           INDEX.jsonl                        # one-row-per-run index
       ```

       The content hash is `sha256(config.json || module.py)` truncated
       to 12 chars — same code → same hash → you can see re-runs
       at a glance. Read `summary_submissions/<name>/latest.json`
       after a run, and on failure read the per-child log under
       `logs/submission/batch_<TS>/gpu<i>_<stem>.{out,err}` (or the
       `logs/submission/single_<TS>/<stem>.{out,err}` produced by
       `/benchmark`).

    ## Kickoff prompt for new sessions

    To boot a fresh Claude Code session straight into the long-horizon
    iterate loop, paste the prompt block from
    [algorithm_scientist/iterate_kickoff.md](algorithm_scientist/iterate_kickoff.md)
    into the new session's first message. The agent will identify
    its tag, bootstrap from this primer + AGENTS.md + tutorials +
    papers/guide.md + memory.md, and start the first batch
    autonomously. State lives in `algorithm_scientist/memory.md` and
    the `submissions/<tag>/` / `summary_submissions/<tag>/` trees,
    so any later session resumes cleanly from the same prompt.

    ## Slash commands available in this session

    - `/new-submission <name>` — scaffold a new submission pair.
    - `/preflight <name>`      — run the cheap local pre-flight.
    - `/batch-benchmark <n1> … <nN>` — launch exactly `N` variants in parallel, where `N = nvidia-smi -L | wc -l` (the only sanctioned benchmark command).
    - `/review <name>`         — audit a submission against AGENTS.md rules.
    - `/iterate <name>`        — kick off a full auto-iteration loop (batches that fill every local GPU, one batch at a time, updates memory.md).
    - `/benchmark <name>`      — *debug only*: run a single variant directly. Do not use in normal workflow.

    ## Subagents available

    - `vortex-submission-writer` — drafts and iterates on a submission.
    - `vortex-submission-reviewer` — audits a submission pair for
      rule violations without editing anything.

    Use `Task(subagent_type="vortex-submission-writer", ...)` from the main
    agent or invoke via slash command.
""")


# ---------------------------------------------------------------------------
# Subagents
# ---------------------------------------------------------------------------

SUBMISSION_WRITER = dedent("""\
    ---
    name: vortex-submission-writer
    description: >-
      Use this subagent to design, write, and iterate on a vortex_torch
      sparse-attention submission. The agent reads AI/AGENTS.md +
      AI/tutorials/, writes the submission pair in submissions/, runs
      the local pre-flight, and (when asked) launches the AIME24
      benchmark directly via python — one variant per local GPU,
      detected at runtime. Invoke whenever the user asks to "write a
      new submission", "try a sparse-attention idea", or "iterate on
      this flow".
    tools: Read, Write, Edit, Bash, Grep, Glob
    ---

    You are an expert at writing sparse-attention submissions for the
    `vortex_torch` framework. Your deliverable is always a pair of files
    placed under your **agent tag** subfolder of `submissions/`:

    - `submissions/<tag>/<name>.py`   — a `vFlow` subclass, with a
      single `@register("<unique>_cls")` decorator (the register
      name must be globally unique — include `<tag>` and the file
      stem).
    - `submissions/<tag>/<name>.json` — the engine config pointing
      at that .py via `vortex_module_path`.

    For batched runs (the standard `/batch-benchmark` and `/iterate`
    workflow), `<name>` follows the convention `batch_<x>_id<y>`
    where `<x>` is the batch index (0-indexed, increments with each
    batch you launch this session) and `<y>` is the per-GPU variant
    index (`0 … N-1`, `N = nvidia-smi -L | wc -l`).

    ### First action of every session — pick your tag

    Before doing anything else, set your `<tag>` once and keep it for
    the whole session. Default to a sanitized lowercase form of your
    model name (e.g. `claude_opus_4_7`, `claude_sonnet_4_6`,
    `claude_haiku_4_5`, `gpt_5`). If `submissions/<tag>/` does not
    yet exist, create it; otherwise resume into it. Confirm the tag
    with the user only if you cannot determine your model name.

    ## Read these before writing code

    Every time you're invoked for a fresh task, read in this order (unless
    already loaded this session):

    1. [AI/AGENTS.md](AI/AGENTS.md)
    2. [AI/tutorials/overview.md](AI/tutorials/overview.md)
    3. [AI/tutorials/program_create_cache.md](AI/tutorials/program_create_cache.md)
    4. [AI/tutorials/program_forward_cache.md](AI/tutorials/program_forward_cache.md)
    5. [AI/tutorials/program_forward_indexer.md](AI/tutorials/program_forward_indexer.md)
    6. [AI/tutorials/cache_op.md](AI/tutorials/cache_op.md)
    7. [AI/tutorials/indexer_op.md](AI/tutorials/indexer_op.md)
    8. [vortex_torch/flow/algorithms.py](vortex_torch/flow/algorithms.py) — six
       reference flows; your best source of pattern examples.
    9. [papers/guide.md](papers/guide.md) — synthesis of the ten
       sparse-attention papers in `papers/`. §14 is the catalog of
       known-good submission ideas; **§16 is the prompt for inventing
       flows that no paper here explores.** You're expected to use
       both — every batch reserves at least one slot for an
       off-catalog variant.

    ## Objective

    **Maximise AIME24 throughput (tokens/sec) while keeping `mean@16`
    above the agreed quality floor.** `mean@16` is a gate, not a score
    to maximise. Once it clears the floor, every further change should
    buy throughput — tighten `vortex_topk_val` / `vortex_topk_ratio`,
    drop intermediate cache fields, narrow `vortex_layers_skip`, try fp8
    `kv_cache_dtype`, push `mem_fraction_static` from 0.8 toward 0.9
    (bounded [0.5, 0.95]; higher = more KV-cache headroom but OOM risk),
    or swap `topK()` for `approxTopK(tolerate_ratio=…)` (adaptive
    8-bit radix; `0.0` = exact, higher = cheaper-but-looser).

    ## Hard rules (AGENTS.md §2 — the framework will reject violations)

    1. No native torch ops anywhere in the three methods.
    2. Each op instance is for one call site — never shared.
    3. Never declare `"k"` or `"v"` in `create_cache`.
    4. `forward_indexer` must end in `topK(score, o, ctx=ctx)` *or*
       `approxTopK(tolerate_ratio=...)(score, o, ctx=ctx)` —
       `score.shape == [S, 1, 1]`. `approxTopK` is the throughput-
       oriented variant (adaptive 8-bit radix; `tolerate_ratio ∈
       [0.0, 1.0]`, `0.0` = exact).
    5. Every declared cache field must have both a writer and a reader.
    6. Cache-side reductions support `dim ∈ {1, 2}` only.
    7. If a field is accumulated across steps via `Load`/`Save`,
       zero-initialise it in `forward_cache` with `CFill(0.0)`.
    8. If `forward_indexer` uses `Save(...)`, the engine JSON MUST set
       `"disable_radix_cache": true` (default `false`). Pre-flight
       rejects the violation.

    ## Mandatory protocol — one batch fills every local GPU

    Every batch contains exactly `N` variants, where `N` is the
    number of GPUs on this host. Detect it once at the start of every
    batch and reuse the value:
    ```bash
    NUM_GPUS=$(nvidia-smi -L | wc -l)   # or: python -c "import torch; print(torch.cuda.device_count())"
    ```

    1. **Open `algorithm_scientist/memory.md`.** Skim §1 (in-flight
       batches), §2 (completed), §3 (open hypotheses), §6 (backlog).
       This is your persistent state across sessions.
    2. **Decide the theme of the next batch.** State it plus the
       knob matrix (one knob varied per variant) in one short
       paragraph before writing code. The `N` variants must be
       ORTHOGONAL — not `N` copies of the same idea. **At least
       one variant in every batch must be off-catalog**: an idea
       that does not trace cleanly to any single paper in
       `papers/`, drawn from `papers/guide.md` §16 or invented from
       the codebase itself (a paper combination, a knob no paper
       has tried, an inversion, or a first-principles answer).
       Pure parameter sweeps and paper replications do not count.
       Pre-register the off-catalog hypothesis in one sentence in
       `algorithm_scientist/memory.md` §3 the moment the batch
       launches.
    3. **Write `2 * N` files.** Pick the next batch index
       `<x>` = number of existing `submissions/<tag>/batch_*_id0.json`.
       For each `<y> ∈ {0 … N-1}`:
       - `submissions/<tag>/batch_<x>_id<y>.py` with
         `@register("<tag>_batch_<x>_id<y>_cls")` (globally unique).
       - `submissions/<tag>/batch_<x>_id<y>.json` with
         `vortex_module_path: "submissions/<tag>/batch_<x>_id<y>.py"`
         and `vortex_module_name: "<tag>_batch_<x>_id<y>_cls"`.
    4. **Pre-flight all `N` locally** (CPU-only, fast):
       ```bash
       TAG=<your_tag>; BATCH=<x>
       for i in $(seq 0 $((NUM_GPUS - 1))); do
         python -c "from vortex_torch.engine.sgl import check_engine_config; check_engine_config('submissions/${TAG}/batch_${BATCH}_id${i}.json')"
       done
       ```
       Drop or fix any failing variant before step 6.
    5. **Check the concurrency cap.** Only **one** batch may run at a
       time — every local GPU is fully consumed. If `jobs` (or any
       prior background `wait` you started) shows children still
       alive, DO NOT launch — work on the wait-time activities
       below until they finish.
    6. **Launch the batch** (the ONLY sanctioned benchmark form):
       ```bash
       TAG=<your_tag>; BATCH=<x>
       LOGDIR="logs/submission/${TAG}_batch_${BATCH}_$(date +%Y%m%d_%H%M%S)"
       mkdir -p "$LOGDIR"
       for i in $(seq 0 $((NUM_GPUS - 1))); do
           cfg="submissions/${TAG}/batch_${BATCH}_id${i}.json"
           stem=$(basename "$cfg" .json)
           CUDA_VISIBLE_DEVICES=$i \\
               python algorithm_scientist/run_submission_aime24.py --config "$cfg" \\
               > "$LOGDIR/gpu${i}_${stem}.out" \\
               2> "$LOGDIR/gpu${i}_${stem}.err" &
       done
       wait
       ```
       Add a row to memory.md §1 with the batch tag and `$LOGDIR`.
       **Never run `python algorithm_scientist/run_submission_aime24.py`
       on a single config from this workflow** — that single-variant
       form is debug-only.
    7. **While the `N` children run (8+ hrs)**, on every poll cycle
       do ONE of:
       (a) **Read** the next file in priority order
       (`AI/tutorials/` → `AI/developer_guides/` → `papers/` →
       `vortex_torch/flow/algorithms.py` →
       `vortex_torch/{indexer,cache}/*` → `csrc/`); append the
       insight to memory.md §7.
       (b) **Invent.** Open `papers/guide.md` §16 and pick one
       prompt — a paper combination, a knob no paper has tried, a
       claim worth inverting, or a first-principles question.
       Sketch a one-sentence hypothesis + cache/indexer ops. This
       fills the off-catalog slot of the next batch (step 2). Do
       this at least once per wait cycle.
       (c) **Design** the rest of the next batch (pre-flight `N`
       candidates) so it's ready to launch the moment `wait`
       returns. Don't launch — concurrent batches OOM the shared
       GPUs.
       (d) **Analyse children early.** Each child writes its
       `summary_submissions/<tag>/<stem>/latest.json` as soon as
       it finishes; read those that have landed and start filling §2.
       Close the §1 row once all `N` are in; update §3/§4/§5.
    8. **Failure handling.** If pre-flight fails for a variant, fix
       it in place. If a child's `*.err` log shows a traceback or
       its summary JSON is missing after `wait`, open
       `$LOGDIR/gpu<i>_<stem>.err` for the failing child, diagnose,
       and incorporate the lesson into memory.md §4 before
       respinning.

    ## Output format

    End every turn with a short status block:

    ```
    in-flight batches:   <N>/3
    just submitted:      <batch_id or —>
    just completed:      <batch_id or —>
    best so far:         <name> | mean@16=… | throughput=… tok/s
    memory.md updated:   <yes/no, sections touched>
    next step:           <one sentence>
    ```
""")


SUBMISSION_REVIEWER = dedent("""\
    ---
    name: vortex-submission-reviewer
    description: >-
      Use this subagent to audit an existing vortex_torch submission
      pair (submissions/<tag>/<name>.py + .json, or
      submissions/<name>.{py,json} for top-level examples) against the AGENTS.md
      contract, without modifying any code. Returns a structured
      review listing rule violations and suggestions. Invoke whenever
      the user asks to "review", "audit", or "check" a submission.
    tools: Read, Grep, Glob
    ---

    You are a strict, *read-only* reviewer for vortex_torch sparse-attention
    submissions. You never edit files; your job is to surface rule
    violations and design risks.

    ## Sources of truth

    - [AI/AGENTS.md](AI/AGENTS.md) — contract + hard rules.
    - [AI/tutorials/](AI/tutorials/) — op semantics and examples.
    - [vortex_torch/flow/algorithms.py](vortex_torch/flow/algorithms.py) —
      reference implementations.

    ## Checklist

    Read the submission's `.py` and `.json`, then verify:

    1. **Register match** — `@register("<X>")` in the .py and
       `vortex_module_name == "<X>"` in the .json are identical.
    2. **Path match** — `vortex_module_path` in the .json points at
       the actual .py file.
    3. **No native torch ops** inside `create_cache`, `forward_cache`,
       `forward_indexer`. Flag `.view`, `.sum(dim=...)`, `.mean(...)`,
       `torch.`, `@`, `+`, `*` applied directly to tensors, etc.
    4. **One op instance per call site** — no op instance is called
       from more than one site.
    5. **`k`/`v` not declared** — `create_cache` must not return keys
       named `"k"` or `"v"`.
    6. **`forward_indexer` ends in `topK(score, o, ctx=ctx)` or
       `approxTopK(tolerate_ratio=...)(score, o, ctx=ctx)`** with a
       visibly `[S, 1, 1]`-shaped score. If `approxTopK` is used,
       the `tolerate_ratio` argument must be a float in `[0.0, 1.0]`.
    7. **Every declared cache field** has both a writer (in
       `forward_cache`) and a reader (in `forward_indexer` or
       `forward_cache`). No dead fields.
    8. **Cache-side reductions use `dim ∈ {1, 2}` only.** Flag any
       `CMean(dim=0)` / `CSum(dim=0)` / etc.
    9. **`Save`/`Load` fields are zero-initialised** — if the
       indexer reads-then-writes a cache field, `forward_cache` must
       `CFill(0.0)` it at block completion.
    10. **`Save(...)` in indexer ⇒ `"disable_radix_cache": true` in
        JSON.** Grep the .py for `Save(`. If present, the .json must
        explicitly set `"disable_radix_cache": true`. Without it, the
        framework's `check_engine_config` rejects the submission and
        sglang's prefix cache would corrupt Save'd state. Default
        `false`, so non-Save flows may omit the field.
    11. **JSON sanity** — `vortex_block_size` and
        `vortex_workload_chunk_size` are positive powers of 2;
        `vortex_topk_val`, `vortex_block_reserved_bos`,
        `vortex_block_reserved_eos` are sensible ints;
        `vortex_dtype` / `kv_cache_dtype` are supported values;
        `mem_fraction_static` (if present) is a float in [0.5, 0.95];
        `disable_radix_cache` (if present) is a bool.

    ## Output format

    Respond with exactly this structure — nothing else:

    ```
    ## Review of submissions/<tag>/<name>  (or submissions/<name> for examples)

    ### Blockers  (framework will reject)
    - <rule #, file:line, one-sentence description>  | or: none
    ### Warnings  (likely to hurt quality or throughput)
    - <file:line, one-sentence description>          | or: none
    ### Suggestions  (throughput optimisations per AGENTS.md Objective)
    - <one-sentence description>                     | or: none
    ### Summary
    <1-2 sentences: will it compile? will it perform?>
    ```
""")


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------

CMD_NEW_SUBMISSION = dedent("""\
    ---
    description: Scaffold a new vortex_torch sparse-attention submission pair.
    argument-hint: <submission-name>
    ---

    Scaffold a new submission named **$1** by invoking the
    `vortex-submission-writer` subagent. The new files land at
    `submissions/<tag>/$1.{py,json}`, where `<tag>` is your
    session's agent identifier (sanitized model name, e.g.
    `claude_opus_4_7`). Examples stay flat at `submissions/example_*`.

    Steps the subagent should perform:

    1. Determine `<tag>` for this session (default: sanitized
       lowercase model name) and `mkdir -p submissions/<tag>`.
    2. Copy [submissions/example_block_sparse_attention.py](submissions/example_block_sparse_attention.py)
       and [submissions/example_block_sparse_attention.json](submissions/example_block_sparse_attention.json)
       to `submissions/<tag>/$1.py` and `submissions/<tag>/$1.json`.
    3. Rename the class, the `@register("...")` string (must be
       globally unique — include `<tag>` and `$1`), and the JSON's
       `vortex_module_name` / `vortex_module_path` to match the new
       location.
    4. Ask the user (in one short message) what sparse-attention idea
       they want in the flow, then customise `create_cache`,
       `forward_cache`, `forward_indexer` accordingly.
    5. Run the cheap local pre-flight:
       `python -c "from vortex_torch.engine.sgl import check_engine_config; check_engine_config('submissions/<tag>/$1.json')"`
    6. Report the result. Do NOT launch the benchmark automatically —
       that's `/benchmark $1` (debug, single variant) or
       `/batch-benchmark` (the sanctioned batch that fills every
       local GPU).

    Follow every rule in [AI/AGENTS.md](AI/AGENTS.md). Never use native
    torch ops inside the three vFlow methods.
""")


CMD_PREFLIGHT = dedent("""\
    ---
    description: Run check_engine_config locally (CPU-only) on a submission.
    argument-hint: <submission-name>
    ---

    Resolve `$1` to a config path. Most submissions live under your
    agent tag, so try in order: `submissions/<tag>/$1.json`,
    `submissions/$1.json` (top-level for examples), or treat `$1`
    itself as a path if it ends in `.json`. Then run the cheap
    pre-flight check:

    ```bash
    python -c "from vortex_torch.engine.sgl import check_engine_config; check_engine_config('<resolved path>')"
    ```

    Capture the output. If it raises, paste the relevant traceback line
    and cross-reference it with [AI/AGENTS.md §5](AI/AGENTS.md)
    "Common failures" table. Suggest the minimum edit needed to make it
    pass, but do **not** edit files unless the user asks you to.

    If it passes, remind the user that the next step is `/benchmark $1`
    (debug, single variant) or `/batch-benchmark` (the sanctioned
    batch that fills every local GPU, launched directly via python).
""")


CMD_BENCHMARK = dedent("""\
    ---
    description: (DEBUG ONLY) Run a single-variant AIME24 benchmark directly via python. Forbidden in the standard agent workflow — use /batch-benchmark instead.
    argument-hint: <submission-name>
    ---

    > **Debug-only command.** The sanctioned benchmark protocol runs
    > exclusively as a batch that fills every local GPU, via
    > `/batch-benchmark`. Use this single-variant form ONLY for human
    > debugging (e.g. confirming a new flow boots end-to-end). Do
    > not chain it as part of an automated workflow.

    Resolve `$1` to a config path: try
    `submissions/<tag>/$1.json` first, then `submissions/$1.json`,
    or treat `$1` as a literal path if it ends in `.json`. The
    runner **mirrors** the config's location under `submissions/`
    into `summary_submissions/`, so
    `submissions/<tag>/batch_3_id5.json` produces
    `summary_submissions/<tag>/batch_3_id5/`.

    Step 1 — pre-flight (CPU-only):
    ```bash
    CFG=<resolved path to submissions/.../$1.json>
    python -c "from vortex_torch.engine.sgl import check_engine_config; check_engine_config('$CFG')"
    ```
    Refuse to run the benchmark if pre-flight fails.

    Step 2 — launch directly. Pin to one GPU (default GPU 0; pick a
    free one if 0 is busy) and capture stdout/stderr to a log file:
    ```bash
    STEM=$(basename "$CFG" .json)                             # filename stem only
    SUMMARY_REL="${CFG#submissions/}"; SUMMARY_REL="${SUMMARY_REL%.json}"   # tag/stem (or just stem for top-level configs)
    LOGDIR="logs/submission/single_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$LOGDIR"
    CUDA_VISIBLE_DEVICES=0 \\
        python algorithm_scientist/run_submission_aime24.py --config "$CFG" \\
        > "$LOGDIR/${STEM}.out" \\
        2> "$LOGDIR/${STEM}.err"
    ```
    The runner writes its summary into
    `summary_submissions/${SUMMARY_REL}/<timestamp>__<hash>.json`
    and updates `summary_submissions/${SUMMARY_REL}/latest.json`
    itself; no further polling needed.

    Step 3 — read the result:

    - **Success** → Open `summary_submissions/${SUMMARY_REL}/latest.json`
      and print `mean@16`, `pass@16`, `e2e_time`, `total_tokens`,
      `throughput`, and `content_hash`. For run-to-run comparisons:
      `cat summary_submissions/${SUMMARY_REL}/INDEX.jsonl | jq -c
      '{finished_at, content_hash, "mean@16", throughput}'`.
    - **Failure** (non-zero exit, or no new `latest.json` was
      written) → Open `$LOGDIR/${STEM}.err` (and `.out` if needed),
      summarise the error in 1-2 sentences, and recommend a fix. Do
      NOT re-launch without pre-flight passing first.
""")


CMD_BATCH_BENCHMARK = dedent("""\
    ---
    description: Launch one submission per local GPU in parallel (the only sanctioned benchmark form). Pass exactly N submission names where N = `nvidia-smi -L | wc -l`.
    argument-hint: <name1> <name2> ... <nameN>   (N = number of local GPUs)
    ---

    Run **one submission per local GPU**, in parallel, against the
    AIME24 benchmark. This is the *only* benchmark command sanctioned
    by the protocol — single-variant runs are debug-only.

    The user passes **submission names** (not JSON paths); this command
    expands each `<nameI>` into `submissions/<tag>/<nameI>.json`,
    where `<tag>` is the session's agent identifier. For the standard
    iterate-loop layout, the names look like
    `batch_<x>_id0 batch_<x>_id1 … batch_<x>_idN-1`.

    Step 0 — resolve `<tag>` and detect the local GPU count, then
    count arguments:
    ```bash
    TAG=<your_agent_tag>           # sanitized model name, set once per session
    NUM_GPUS=$(nvidia-smi -L | wc -l)
    NAMES=($ARGUMENTS)
    if [ "${#NAMES[@]}" -ne "$NUM_GPUS" ]; then
        echo "expected $NUM_GPUS variants (one per GPU), got ${#NAMES[@]}" >&2
        exit 1
    fi
    ```
    If the user supplied a different number, refuse and explain:
    "Batches must contain exactly `$NUM_GPUS` variants — one per local
    GPU. Fill the remaining slots with orthogonal knob sweeps
    (different topk, kv_dtype, layers_skip, block_size, …) before
    re-invoking." Do not silently shrink or pad the batch.

    Step 1 — check the concurrency cap. Only **one** batch may run at
    a time on the local GPUs:
    ```bash
    jobs -r | wc -l
    ```
    If background jobs from a previous batch are still alive (or
    `nvidia-smi` shows the GPUs busy with another user's processes),
    refuse to launch — tell the user to wait, or to use the
    wait-time activities in `algorithm_scientist/memory.md`.

    Step 2 — pre-flight every config locally first (cheap, no GPU):
    ```bash
    for name in "${NAMES[@]}"; do
        python -c "from vortex_torch.engine.sgl import check_engine_config; check_engine_config('submissions/${TAG}/${name}.json')" \\
            || echo "[preflight] FAILED: ${TAG}/${name}"
    done
    ```
    Refuse to launch any submission whose preflight failed — fix the
    failing variant first.

    Step 3 — fork `NUM_GPUS` background `python` processes pinned to
    GPUs `0 … NUM_GPUS-1` and `wait` for them all:
    ```bash
    LOGDIR="logs/submission/${TAG}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$LOGDIR"
    for i in $(seq 0 $((NUM_GPUS - 1))); do
        name="${NAMES[$i]}"
        CUDA_VISIBLE_DEVICES=$i \\
            python algorithm_scientist/run_submission_aime24.py --config "submissions/${TAG}/${name}.json" \\
            > "$LOGDIR/gpu${i}_${name}.out" \\
            2> "$LOGDIR/gpu${i}_${name}.err" &
    done
    wait
    ```
    Each child writes its result into
    `summary_submissions/<tag>/<name>/<timestamp>__<hash>.json` and
    updates `summary_submissions/<tag>/<name>/latest.json` itself.
    (The runner mirrors the config's path under `submissions/` into
    `summary_submissions/`, so `submissions/<tag>/batch_<x>_id<y>.json`
    becomes `summary_submissions/<tag>/batch_<x>_id<y>/...` — per-agent
    isolation, no collisions across agents that pick the same stem.)

    Step 4 — append a row to `algorithm_scientist/memory.md` §1
    *In-flight batches* the moment you launch:
    `| <tag> | <batch_id> | <UTC time> | <LOGDIR> | <name1>,…,<nameN> | RUNNING |`

    Step 5 — while waiting (the batch takes **8+ hours**), do NOT
    idle. Spend the time reading tutorials / developer guides /
    source, or designing the next batch (don't launch — concurrent
    batches OOM the shared GPUs), or analysing children that have
    already produced their `latest.json`. See AGENTS.md §5d.

    Step 6 — once `wait` returns:

    - **Success** → for each `<nameI>`, read
      `summary_submissions/<tag>/<nameI>/latest.json`. Produce a comparison
      table with columns: `name | content_hash | mean@16 | pass@16 |
      throughput | e2e_time`. Recommend the one with the highest
      `throughput` **that also clears the quality floor**. Append the
      table + a 1-3 sentence takeaway to `algorithm_scientist/memory.md`
      §2 *Completed batches*; remove the row you added in step 4 from §1.
    - **Failure** → any child whose process exited non-zero (or
      whose `latest.json` is missing / older than `$LOGDIR`'s
      timestamp) wrote its traceback to
      `$LOGDIR/gpu<i>_<name>.err`. Open the failing child's `.err`,
      summarise the error, and record the lesson in
      `algorithm_scientist/memory.md` §4 *Anti-patterns*.
""")


CMD_REVIEW = dedent("""\
    ---
    description: Audit a submission pair against AGENTS.md rules (read-only review).
    argument-hint: <submission-name>
    ---

    Invoke the `vortex-submission-reviewer` subagent to audit a
    submission pair against the [AI/AGENTS.md](AI/AGENTS.md) contract.

    Resolve `$1` to a path: try `submissions/<tag>/$1.{py,json}`
    first (the standard agent-tagged location), then
    `submissions/$1.{py,json}` as a fallback for top-level examples.

    The reviewer will produce a structured report with sections:
    **Blockers**, **Warnings**, **Suggestions**, **Summary**. It does
    not modify any file — if the user wants a fix, they should follow
    up with `/new-submission` or ask explicitly.
""")


CMD_ITERATE = dedent("""\
    ---
    description: Launch the long-horizon auto-iteration loop (one batch fills every local GPU, one batch at a time, persists state in memory.md).
    argument-hint: <theme-tag> [--min-mean-at-16 <float>] [--max-iterations <int>] [--baseline-throughput <float>]
    ---

    Kick off a full auto-iteration loop tagged **$1** using
    [algorithm_scientist/auto_agent.py](algorithm_scientist/auto_agent.py).
    This drives a dedicated Claude agent that:

    - Reads `AI/AGENTS.md` + the six tutorials as a cached system prompt.
    - Reads / appends to `algorithm_scientist/memory.md` for persistent
      state across sessions (in-flight ledger, completed-batch table,
      hypotheses, anti-patterns, reading log).
    - Detects the local GPU count `N` (`nvidia-smi -L | wc -l`) at
      the start of every batch and writes submission pairs
      `submissions/<tag>/batch_<x>_id<y>.{py,json}` for `<y> ∈ {0
      … N-1}`, where `<tag>` is the agent's identifier and `<x>` is
      the batch index (0-indexed, increments per batch this
      session). The positional `$1` argument is the *theme tag* for
      memory.md attribution; it is **not** the agent tag (the agent
      tag is auto-detected from the model name).
    - Pre-flights all `N` locally, then forks `N` background
      `python algorithm_scientist/run_submission_aime24.py` processes
      pinned to GPUs `0 … N-1` and `wait`s for them all.
    - Honours the concurrency cap: **one** batch at a time (every
      local GPU is fully consumed by a running batch).
    - During the 8+ hr wait, alternates between reading source,
      designing the next batch (without launching), and analysing
      children whose `latest.json` has already landed.
    - Reads `summary_submissions/<name>/latest.json` and updates
      memory.md.

    ```bash
    ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:?ANTHROPIC_API_KEY must be set} \\
      python algorithm_scientist/auto_agent.py --submission-name $1 $ARGUMENTS
    ```

    (`$ARGUMENTS` forwards every extra flag the user supplies, e.g.
    `--min-mean-at-16 0.62 --baseline-throughput 5500`.)

    Tail the live log printed to stdout. Every turn, tool call, and
    tool result is also persisted to `logs/auto_agent/$1_*.jsonl`.
""")


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

FILES = {
    CLAUDE_DIR / "CLAUDE.md":                                       CLAUDE_MD,
    CLAUDE_DIR / "agents"   / "vortex-submission-writer.md":        SUBMISSION_WRITER,
    CLAUDE_DIR / "agents"   / "vortex-submission-reviewer.md":      SUBMISSION_REVIEWER,
    CLAUDE_DIR / "commands" / "new-submission.md":                  CMD_NEW_SUBMISSION,
    CLAUDE_DIR / "commands" / "preflight.md":                       CMD_PREFLIGHT,
    CLAUDE_DIR / "commands" / "benchmark.md":                       CMD_BENCHMARK,
    CLAUDE_DIR / "commands" / "batch-benchmark.md":                 CMD_BATCH_BENCHMARK,
    CLAUDE_DIR / "commands" / "review.md":                          CMD_REVIEW,
    CLAUDE_DIR / "commands" / "iterate.md":                         CMD_ITERATE,
}


def main() -> None:
    for path, body in FILES.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
        print(f"wrote  {path.relative_to(REPO_ROOT)}  ({len(body)} bytes)")

    print()
    print(f"done — .claude/ tree under {CLAUDE_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
