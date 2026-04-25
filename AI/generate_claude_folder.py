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
    `vortex_layers_skip`, aggressive fp8 `kv_cache_dtype`. When two variants
    both clear the floor, pick the faster one.

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
    - **`forward_indexer` must end in `topK(score, o, ctx=ctx)`** with
      `score.shape == [S, 1, 1]`.
    - **Cache-side reductions support `dim ∈ {1, 2}` only.** Cross-block
      reductions (`dim=0`) belong on the indexer side.
    - **If a field is read+written across steps via `Load`/`Save`, zero
      it with `CFill(0.0)` in `forward_cache`.**

    ## Running the benchmark — policy

    **The only allowed unit of work is a batch of 8.** Single-variant
    runs (`run_submission.slurm`) are debug-only and off-limits to the
    automated workflow. Each batch:

    1. **8 orthogonal variants** — `submissions/<tag>_v1.{py,json}` …
       `submissions/<tag>_v8.{py,json}` — varying different knobs.
    2. **Cheap local pre-flight first** for all 8 (CPU, no GPU):
       ```bash
       for i in 1 2 3 4 5 6 7 8; do
         python -c "from vortex_torch.engine.sgl import check_engine_config; check_engine_config('submissions/<tag>_v${i}.json')"
       done
       ```
       Refuse to sbatch any variant whose pre-flight fails.
    3. **Submit on Slurm** — the host has no GPU, so direct invocation
       of `algorithm_scientist/run_submission_aime24.py` will fail.
       Always use the batched slurm file (whole 8-GPU node, each child
       pinned to its own `CUDA_VISIBLE_DEVICES=0..7`):
       ```bash
       sbatch algorithm_scientist/run_submission_batch.slurm \\
           submissions/<tag>_v1.json submissions/<tag>_v2.json \\
           submissions/<tag>_v3.json submissions/<tag>_v4.json \\
           submissions/<tag>_v5.json submissions/<tag>_v6.json \\
           submissions/<tag>_v7.json submissions/<tag>_v8.json
       ```
       Per-child logs land under
       `logs/submission/batch_<JOBID>/gpu<i>_<stem>.{out,err}`.

    4. **In-flight ceiling = 24 experiments (3 batches).** Before every
       sbatch:
       ```bash
       squeue -u $USER -h -o '%i %j %T'
       ```
       If 3 `vortex_submission_batch` rows are already PENDING/RUNNING,
       do not submit — work on the wait-time protocol below until a
       slot frees up.

    5. Poll:
       ```bash
       sacct -j <JOBID> --format=JobID,State,ExitCode -X -n -P
       ```

    ## While you wait (8+ hrs per batch)

    Idle is not an option. Each polling cycle, do one of:

    - **Read.** Priority: `AI/tutorials/` → `AI/developer_guides/` →
      `vortex_torch/flow/algorithms.py` →
      `vortex_torch/{indexer,cache}/*` → `csrc/`. After each file,
      append one insight to `algorithm_scientist/memory.md` §7.
    - **Prepare the next batch.** If fewer than 3 batches in flight,
      design 8 orthogonal variants for a different theme, pre-flight
      them, submit, record in `algorithm_scientist/memory.md` §1.
    - **Analyse.** For any terminal batch, read all 8 `latest.json`
      files, fill a §2 sub-section (table + takeaway), remove the §1
      row, update §3 (hypotheses) / §4 (anti-patterns) / §5 (winners).

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
       after a run, and on failure read
       `logs/submission/vortex_submission_<JOBID>.{out,err}`.

    ## Slash commands available in this session

    - `/new-submission <name>` — scaffold a new submission pair.
    - `/preflight <name>`      — run the cheap local pre-flight.
    - `/batch-benchmark <n1> … <n8>` — sbatch exactly 8 variants on one node (the only sanctioned benchmark command).
    - `/review <name>`         — audit a submission against AGENTS.md rules.
    - `/iterate <name>`        — kick off a full auto-iteration loop (batches of 8, ≤ 24 in flight, updates memory.md).
    - `/benchmark <name>`      — *debug only*: sbatch a single-variant AIME24 run. Do not use in normal workflow.

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
      AI/tutorials/, writes the submission pair in submissions/, runs the
      local pre-flight, and (when asked) submits the AIME24 benchmark to
      Slurm. Invoke whenever the user asks to "write a new submission",
      "try a sparse-attention idea", or "iterate on this flow".
    tools: Read, Write, Edit, Bash, Grep, Glob
    ---

    You are an expert at writing sparse-attention submissions for the
    `vortex_torch` framework. Your deliverable is always a pair of files
    placed in `submissions/`:

    - `submissions/<name>.py`   — a `vFlow` subclass, with a single
      `@register("<name>_cls")` decorator.
    - `submissions/<name>.json` — the engine config pointing at that .py.

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

    ## Objective

    **Maximise AIME24 throughput (tokens/sec) while keeping `mean@16`
    above the agreed quality floor.** `mean@16` is a gate, not a score
    to maximise. Once it clears the floor, every further change should
    buy throughput — tighten `vortex_topk_val` / `vortex_topk_ratio`,
    drop intermediate cache fields, narrow `vortex_layers_skip`, try fp8
    `kv_cache_dtype`.

    ## Hard rules (AGENTS.md §2 — the framework will reject violations)

    1. No native torch ops anywhere in the three methods.
    2. Each op instance is for one call site — never shared.
    3. Never declare `"k"` or `"v"` in `create_cache`.
    4. `forward_indexer` must end in `topK(score, o, ctx=ctx)` with
       `score.shape == [S, 1, 1]`.
    5. Every declared cache field must have both a writer and a reader.
    6. Cache-side reductions support `dim ∈ {1, 2}` only.
    7. If a field is accumulated across steps via `Load`/`Save`,
       zero-initialise it in `forward_cache` with `CFill(0.0)`.

    ## Mandatory protocol — batches of exactly 8

    1. **Open `algorithm_scientist/memory.md`.** Skim §1 (in-flight
       batches), §2 (completed), §3 (open hypotheses), §6 (backlog).
       This is your persistent state across sessions.
    2. **Decide the theme of the next batch of 8.** State it plus the
       knob matrix (one knob varied per variant) in one short paragraph
       before writing code. The 8 variants must be ORTHOGONAL — not
       eight copies of the same idea.
    3. **Write 16 files** — for each `vI ∈ {v1..v8}`:
       - `submissions/<tag>_vI.py`  with `@register("<tag>_vI_cls")`.
       - `submissions/<tag>_vI.json` pointing at the .py.
    4. **Pre-flight all 8 locally** (CPU-only, fast):
       ```bash
       for i in 1 2 3 4 5 6 7 8; do
         python -c "from vortex_torch.engine.sgl import check_engine_config; check_engine_config('submissions/<tag>_v${i}.json')"
       done
       ```
       Drop or fix any failing variant before step 6.
    5. **Check the in-flight ceiling.** If `squeue -u $USER -h -o '%j'`
       already shows 3 `vortex_submission_batch` rows, DO NOT submit —
       work on the wait-time activities below until a slot frees up.
    6. **Submit the batch of 8** (the ONLY sanctioned benchmark form):
       ```bash
       sbatch algorithm_scientist/run_submission_batch.slurm \\
           submissions/<tag>_v1.json submissions/<tag>_v2.json \\
           submissions/<tag>_v3.json submissions/<tag>_v4.json \\
           submissions/<tag>_v5.json submissions/<tag>_v6.json \\
           submissions/<tag>_v7.json submissions/<tag>_v8.json
       ```
       Add a row to memory.md §1 with the new JOBID.
       **Never use `algorithm_scientist/run_submission.slurm`** — that
       single-variant slurm file is debug-only and forbidden in this
       workflow.
    7. **While Slurm is running (8+ hrs)**, on every poll cycle do ONE of:
       (a) **Read** the next file in priority order
       (`AI/tutorials/` → `AI/developer_guides/` →
       `vortex_torch/flow/algorithms.py` →
       `vortex_torch/{indexer,cache}/*` → `csrc/`); append the
       insight to memory.md §7.
       (b) **Prepare another batch** if < 3 are in flight: pick a new
       theme, write 16 files, pre-flight, submit, record in §1.
       (c) **Analyse** any terminal batch: read 8 `latest.json`
       files, fill a §2 sub-section, remove the §1 row, update
       §3/§4/§5.
    8. **Failure handling.** If pre-flight fails for a variant, fix
       it in place. If the Slurm job's `sacct` state is
       FAILED/TIMEOUT/etc., open
       `logs/submission/batch_<JOBID>/gpu<i>_<stem>.err` for the
       failing children, diagnose, and incorporate the lesson into
       memory.md §4 before respinning.

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
      pair (submissions/<name>.py + .json) against the AGENTS.md
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
    6. **`forward_indexer` ends in `topK(score, o, ctx=ctx)`** with a
       visibly `[S, 1, 1]`-shaped score.
    7. **Every declared cache field** has both a writer (in
       `forward_cache`) and a reader (in `forward_indexer` or
       `forward_cache`). No dead fields.
    8. **Cache-side reductions use `dim ∈ {1, 2}` only.** Flag any
       `CMean(dim=0)` / `CSum(dim=0)` / etc.
    9. **`Save`/`Load` fields are zero-initialised** — if the
       indexer reads-then-writes a cache field, `forward_cache` must
       `CFill(0.0)` it at block completion.
    10. **JSON sanity** — `vortex_block_size` and
        `vortex_workload_chunk_size` are positive powers of 2;
        `vortex_topk_val`, `vortex_block_reserved_bos`,
        `vortex_block_reserved_eos` are sensible ints;
        `vortex_dtype` / `kv_cache_dtype` are supported values.

    ## Output format

    Respond with exactly this structure — nothing else:

    ```
    ## Review of submissions/<name>

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
    `vortex-submission-writer` subagent.

    Steps the subagent should perform:

    1. Copy [submissions/example_block_sparse_attention.py](submissions/example_block_sparse_attention.py)
       and [submissions/example_block_sparse_attention.json](submissions/example_block_sparse_attention.json)
       to `submissions/$1.py` and `submissions/$1.json`.
    2. Rename the class, the `@register("...")` string, and the JSON's
       `vortex_module_name` / `vortex_module_path` to match **$1**.
    3. Ask the user (in one short message) what sparse-attention idea
       they want in the flow, then customise `create_cache`,
       `forward_cache`, `forward_indexer` accordingly.
    4. Run the cheap local pre-flight:
       `python -c "from vortex_torch.engine.sgl import check_engine_config; check_engine_config('submissions/$1.json')"`
    5. Report the result. Do NOT submit to Slurm automatically —
       that's `/benchmark $1`.

    Follow every rule in [AI/AGENTS.md](AI/AGENTS.md). Never use native
    torch ops inside the three vFlow methods.
""")


CMD_PREFLIGHT = dedent("""\
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
""")


CMD_BENCHMARK = dedent("""\
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
""")


CMD_BATCH_BENCHMARK = dedent("""\
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
        python -c "from vortex_torch.engine.sgl import check_engine_config; check_engine_config('submissions/${name}.json')" \\
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
""")


CMD_REVIEW = dedent("""\
    ---
    description: Audit a submission pair against AGENTS.md rules (read-only review).
    argument-hint: <submission-name>
    ---

    Invoke the `vortex-submission-reviewer` subagent to audit
    `submissions/$1.py` and `submissions/$1.json` against the
    [AI/AGENTS.md](AI/AGENTS.md) contract.

    The reviewer will produce a structured report with sections:
    **Blockers**, **Warnings**, **Suggestions**, **Summary**. It does
    not modify any file — if the user wants a fix, they should follow
    up with `/new-submission` or ask explicitly.
""")


CMD_ITERATE = dedent("""\
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
