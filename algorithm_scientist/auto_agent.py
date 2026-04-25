"""Autonomous submission writer for the vortex_torch sparse-attention competition.

Drives Claude through an agentic loop (SDK: ``anthropic``) that:

  1. Reads the AGENTS.md brief + the six tutorials in ``AI/tutorials/``
     (all bundled into a cached system prompt).
  2. Writes a new submission pair
     ``submissions/<name>.py`` + ``submissions/<name>.json``.
  3. Submits the AIME24 benchmark to **Slurm** via
     :file:`algorithm_scientist/run_submission.slurm`, polls ``squeue`` /
     ``sacct`` until the job terminates, then parses the summary
     JSON under ``summary_submissions/``.
  4. Iterates on failures (missing op, bad dtype, low mean@16, etc.)
     until either the target metric is reached or ``--max-iterations``
     is exhausted.

The benchmark itself MUST run on Slurm — the login environment
where this script is launched does not necessarily have a GPU. The
agent uses the ``bash`` tool to ``sbatch``, poll, and read the
Slurm stdout/stderr logs.

The optimisation objective is **maximise throughput (tokens/sec)
subject to `mean@16 >= --min-mean-at-16`** — the same contract
described in `AI/AGENTS.md`.

Usage
-----
::

    export ANTHROPIC_API_KEY=...
    python examples/auto_agent.py \\
        --submission-name my_agent_v1 \\
        --max-iterations 6 \\
        --min-mean-at-16 0.62 \\
        --baseline-throughput 5500

Only ``ANTHROPIC_API_KEY`` must be set externally.  Everything else has
a safe default.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import anthropic
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "The 'anthropic' package is required.  "
        "Install with: pip install 'anthropic>=0.79'"
    ) from e


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
AGENTS_MD = REPO_ROOT / "AI" / "AGENTS.md"
TUTORIAL_DIR = REPO_ROOT / "AI" / "tutorials"
TUTORIAL_FILES = [
    "overview.md",
    "program_create_cache.md",
    "program_forward_cache.md",
    "program_forward_indexer.md",
    "cache_op.md",
    "indexer_op.md",
]
ALGORITHMS_PY = REPO_ROOT / "vortex_torch" / "flow" / "algorithms.py"
EXAMPLE_JSON = REPO_ROOT / "submissions" / "example_block_sparse_attention.json"
EXAMPLE_PY = REPO_ROOT / "submissions" / "example_block_sparse_attention.py"
SUMMARY_DIR = REPO_ROOT / "summary_submissions"
MEMORY_MD = REPO_ROOT / "algorithm_scientist" / "memory.md"


# ---------------------------------------------------------------------------
# Claude config
# ---------------------------------------------------------------------------

MODEL = "claude-opus-4-7"
MAX_TOKENS = 16384


# ---------------------------------------------------------------------------
# Tools exposed to Claude
# ---------------------------------------------------------------------------

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "read_file",
        "description": (
            "Read a UTF-8 text file from the vortex_torch repo. "
            "Paths may be absolute or relative to the repo root. "
            "Returns up to 400 KB of text."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Write (overwriting) a UTF-8 text file. Intended for "
            "submissions/<name>.py and submissions/<name>.json. "
            "Parent directories are created automatically."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path":    {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_dir",
        "description": "List filenames in a directory (non-recursive).",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "bash",
        "description": (
            "Run a shell command from the repo root. "
            "Used to run check_engine_config / the AIME24 benchmark / "
            "quick python -c probes. Combined stdout+stderr are returned, "
            "truncated to 20 KB."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command":     {"type": "string"},
                "timeout_sec": {
                    "type": "integer",
                    "description": "Optional timeout (default 1800).",
                },
            },
            "required": ["command"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def _tool_read_file(path: str) -> str:
    p = _resolve(path)
    if not p.is_file():
        return f"ERROR: not a file: {p}"
    data = p.read_bytes()
    if len(data) > 400_000:
        data = data[:400_000] + b"\n... [truncated] ..."
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return f"ERROR: file is not utf-8 text ({p})"


def _tool_write_file(path: str, content: str) -> str:
    p = _resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} bytes to {p}"


def _tool_list_dir(path: str) -> str:
    p = _resolve(path)
    if not p.is_dir():
        return f"ERROR: not a directory: {p}"
    return "\n".join(sorted(x.name for x in p.iterdir()))


def _tool_bash(command: str, timeout_sec: int = 1800) -> str:
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as e:
        return f"TIMEOUT after {timeout_sec}s\n--- partial stdout ---\n{e.stdout or ''}\n--- partial stderr ---\n{e.stderr or ''}"
    out = (proc.stdout or "") + (proc.stderr or "")
    if len(out) > 20_000:
        out = out[:10_000] + "\n... [truncated middle] ...\n" + out[-10_000:]
    return f"exit={proc.returncode}\n{out}"


def _dispatch_tool(name: str, args: Dict[str, Any]) -> str:
    try:
        if name == "read_file":
            return _tool_read_file(args["path"])
        if name == "write_file":
            return _tool_write_file(args["path"], args["content"])
        if name == "list_dir":
            return _tool_list_dir(args["path"])
        if name == "bash":
            return _tool_bash(args["command"], args.get("timeout_sec", 1800))
        return f"ERROR: unknown tool {name!r}"
    except KeyError as e:
        return f"ERROR: missing required arg {e} for tool {name}"
    except Exception as e:  # pragma: no cover — surface to model
        return f"ERROR: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# System prompt assembly (with prompt caching)
# ---------------------------------------------------------------------------

def _load_bundle() -> str:
    """Concatenate AGENTS.md + tutorials + reference example into one string."""
    parts: List[str] = []

    def _section(title: str, path: Path) -> None:
        parts.append(f"\n\n===== {title} ({path.relative_to(REPO_ROOT)}) =====\n")
        parts.append(path.read_text(encoding="utf-8"))

    _section("AGENTS.md",            AGENTS_MD)
    for name in TUTORIAL_FILES:
        _section(f"tutorial: {name}", TUTORIAL_DIR / name)
    _section("reference flows",      ALGORITHMS_PY)
    _section("example submission .py",   EXAMPLE_PY)
    _section("example submission .json", EXAMPLE_JSON)
    return "".join(parts)


def _build_system_blocks() -> List[Dict[str, Any]]:
    preamble = (
        "You are an expert autonomous algorithm scientist working on "
        "the vortex_torch sparse-attention competition. Your job is "
        "to design, submit, and iterate on `vFlow` submissions over "
        "very long wall-clock horizons (a single batch of experiments "
        "can take 8+ hours). You manage your own backlog — do not "
        "ask the user to plan for you.\n\n"
        "Hard rules — read carefully, these are non-negotiable:\n"
        "  1. **SUBMISSIONS ALWAYS RUN IN BATCHES OF EXACTLY 8.** "
        "     Single-variant runs are disallowed in this workflow. If "
        "     you have only one real idea, you MUST still fill the other "
        "     seven slots with orthogonal knob sweeps of that idea "
        "     (different `vortex_topk_val`, different `kv_cache_dtype`, "
        "     different `mem_fraction_static` ∈ [0.5, 0.95] (0.8-0.9 sweet spot, higher = more throughput but OOM risk), "
        "     different `vortex_layers_skip`, different block sizes, "
        "     different scoring-op combinations, etc.). Submit via:\n"
        "         sbatch algorithm_scientist/run_submission_batch.slurm \\\n"
        "             submissions/v1.json submissions/v2.json ... "
        "submissions/v8.json\n"
        "     Do NOT use `algorithm_scientist/run_submission.slurm` — "
        "     the single-variant file is retained for human debugging "
        "     only and is off-limits to you.\n"
        "  2. **AT MOST 24 EXPERIMENTS IN FLIGHT AT ONCE.** That is "
        "     ≤ 3 concurrent batches. Before every `sbatch`, check "
        "     in-flight batches with\n"
        "         squeue -u $USER -h -o '%i %j %T'\n"
        "     If 3 batches are already PENDING/RUNNING, DO NOT submit — "
        "     spend the time on the wait-time activities below.\n"
        "  3. **EVERY BENCHMARK RUN GOES THROUGH SLURM.** The host you "
        "     are running on may not have a GPU; invoking "
        "     `python algorithm_scientist/run_submission_aime24.py` "
        "     directly will fail or hang. Use sbatch + sacct polling.\n"
        "  4. **ALWAYS PRE-FLIGHT LOCALLY BEFORE SBATCH.** Cheap, CPU-only:\n"
        "         python -c \"from vortex_torch.engine.sgl import "
        "check_engine_config; check_engine_config('submissions/<name>.json')\"\n"
        "     Refuse to sbatch any of the 8 configs whose pre-flight "
        "     fails — fix or drop them from the batch first.\n"
        "  5. **MEMORY LIVES IN "
        "`algorithm_scientist/memory.md`** — not in the chat. Read it "
        "     at start and update it before stopping. It is your only "
        "     persistent state across sessions; the conversation is not.\n\n"
        "What to do while Slurm jobs are running (this is MOST of your time):\n"
        "  Batches take 8+ hours. Idle is forbidden. In each polling "
        "  turn, do one of three things — spread time across all of "
        "  them, don't fixate:\n"
        "  (a) **Deepen understanding.** Priority order:\n"
        "         AI/tutorials/  →  AI/developer_guides/  →  "
        "vortex_torch/flow/algorithms.py  →  "
        "vortex_torch/{indexer,cache}/*  →  csrc/.\n"
        "      After reading a file, append one bullet to memory.md "
        "      §7 'Reading log' with the single most useful insight.\n"
        "  (b) **Prepare the next batch.** If fewer than 3 batches are "
        "      in flight, design 8 new orthogonal variants, write the "
        "      16 files (`submissions/<tag>_v{1..8}.{py,json}`), "
        "      pre-flight all 8, and submit via "
        "      run_submission_batch.slurm. Before submitting, add a row "
        "      to memory.md §1 'In-flight batches'.\n"
        "  (c) **Analyse finished batches.** For any batch whose "
        "      `sacct` state is terminal, read "
        "      `summary_submissions/<name>/latest.json` for each of the "
        "      8 children, add a results table + 1-3 sentence takeaway "
        "      to memory.md §2, then remove the batch's row from §1. "
        "      Update §3 (hypotheses) / §4 (anti-patterns) / §5 "
        "      (patterns that worked) as the evidence lands.\n\n"
        "Other rules:\n"
        "  - Never touch files outside `submissions/`, "
        "    `algorithm_scientist/memory.md`, and "
        "    `algorithm_scientist/` logs. Reading is fine everywhere.\n"
        "  - Each finished run writes to\n"
        "        summary_submissions/<name>/\n"
        "            <timestamp>__<content_hash>.json   # embeds .py/.json text\n"
        "            latest.json                        # symlink → newest run\n"
        "            INDEX.jsonl                        # one-row-per-run index\n"
        "    `content_hash = sha256(config.json || module.py)[:12]` so "
        "    identical code → identical hash; re-runs are visible.\n"
        "  - Poll with "
        "    `sacct -j <JOBID> --format=JobID,State,ExitCode -X -n -P`. "
        "    On failure read `logs/submission/batch_<JOBID>/gpu<i>_<stem>.err`.\n"
        "  - Final report (when you stop): emit a terse summary naming "
        "    the best submission (highest `throughput` that cleared the "
        "    `mean@16` floor), its `content_hash`, and its headline "
        "    numbers. The full audit trail is in memory.md.\n"
        "  - Use tools efficiently — cache what you've already read, "
        "    don't re-read the same file more than once per session."
    )
    bundle = _load_bundle()
    return [
        {"type": "text", "text": preamble},
        {
            "type": "text",
            "text": bundle,
            "cache_control": {"type": "ephemeral"},
        },
    ]


# ---------------------------------------------------------------------------
# Initial user prompt
# ---------------------------------------------------------------------------

def _build_initial_user_prompt(
    submission_name: str,
    min_mean: float,
    baseline_throughput: Optional[float],
    max_iterations: int,
) -> str:
    baseline_line = (
        f"Reference: `example_block_sparse_attention` delivers roughly "
        f"`throughput ≈ {baseline_throughput:.0f} tok/s`. You must "
        f"comfortably exceed it.\n"
        if baseline_throughput is not None else ""
    )
    return (
        f"## Objective\n"
        f"**Maximise AIME24 `throughput` (tokens/sec) subject to "
        f"`mean@16 >= {min_mean}`.** `mean@16` is a floor, not a score "
        f"to maximise. When two variants both clear the floor, pick the "
        f"faster one.\n"
        f"{baseline_line}\n"
        f"## Hard constraints (restated)\n"
        f"- **Batches of exactly 8.** Submit only via "
        f"`run_submission_batch.slurm`, never the single-variant form.\n"
        f"- **≤ 24 experiments in flight** (≤ 3 concurrent batches). "
        f"Check with `squeue -u $USER -h -o '%i %j %T'` before each "
        f"`sbatch`.\n"
        f"- **Persistent state lives in "
        f"`algorithm_scientist/memory.md`**. Read it now (if it already "
        f"has content), and update it after every batch submission and "
        f"every completion.\n\n"
        f"## Protocol\n"
        f"0. **Open memory.md** and skim every section. If this is a "
        f"   fresh session, there will only be template scaffolding; if "
        f"   earlier sessions ran, §1 (in-flight) and §2 (completed) "
        f"   will have real rows.\n"
        f"1. **Decide the theme of your first batch of 8.** Call it "
        f"   `{submission_name}` (tag). The 8 variants should be "
        f"   orthogonal — `{submission_name}_v1` through "
        f"   `{submission_name}_v8` — each varying a different knob "
        f"   (topk_val, topk_ratio, layers_skip pattern, kv_cache_dtype, "
        f"   mem_fraction_static ∈ [0.5, 0.95], "
        f"   block_size, scoring op choice, etc.), NOT 8 copies of the "
        f"   same idea. State the theme and the 8 variants' knob "
        f"   matrix in one short paragraph before writing code.\n"
        f"2. **Write 16 files** — for each `vI` in `v1..v8`:\n"
        f"   - `submissions/{submission_name}_vI.py`  — the vFlow subclass "
        f"with `@register(\"{submission_name}_vI_cls\")`.\n"
        f"   - `submissions/{submission_name}_vI.json` — config pointing "
        f"at the .py.\n"
        f"3. **Pre-flight all 8 locally** (CPU-only, fast):\n"
        f"   ```bash\n"
        f"   for i in 1 2 3 4 5 6 7 8; do\n"
        f"     python -c \"from vortex_torch.engine.sgl import "
        f"check_engine_config; "
        f"check_engine_config('submissions/{submission_name}_v${{i}}.json')\"\n"
        f"   done\n"
        f"   ```\n"
        f"   Any failing variant must be fixed or replaced before step 4.\n"
        f"4. **Check the in-flight ceiling.** If "
        f"`squeue -u $USER -h -o '%j'` already shows 3 running/pending "
        f"`vortex_submission_batch` jobs, DO NOT submit — jump straight "
        f"to step 6 and spend the time reading / analysing.\n"
        f"5. **Submit the batch of 8**:\n"
        f"   ```bash\n"
        f"   sbatch algorithm_scientist/run_submission_batch.slurm \\\n"
        f"     submissions/{submission_name}_v1.json "
        f"submissions/{submission_name}_v2.json \\\n"
        f"     submissions/{submission_name}_v3.json "
        f"submissions/{submission_name}_v4.json \\\n"
        f"     submissions/{submission_name}_v5.json "
        f"submissions/{submission_name}_v6.json \\\n"
        f"     submissions/{submission_name}_v7.json "
        f"submissions/{submission_name}_v8.json\n"
        f"   ```\n"
        f"   Capture the JOBID. **Add a row to memory.md §1** "
        f"(batch_id, submitted_at, slurm_job_id, submissions, status=PENDING).\n"
        f"6. **While the batch runs (8+ hrs), don't idle.** "
        f"Each polling turn, choose one of (a)/(b)/(c) — spread "
        f"attention across all three over the wait:\n"
        f"   (a) **Read and learn.** One file per turn. Order of "
        f"priority: AI/tutorials/ → AI/developer_guides/ → "
        f"vortex_torch/flow/algorithms.py → "
        f"vortex_torch/{{indexer,cache}}/* → csrc/. "
        f"Append one bullet to memory.md §7 with the insight.\n"
        f"   (b) **Prepare the next batch.** If < 3 batches in flight, "
        f"draft 8 more orthogonal variants for a different theme, "
        f"pre-flight them, submit, and record in §1.\n"
        f"   (c) **Analyse.** For any batch whose `sacct` state is "
        f"terminal, read the 8 `summary_submissions/<name>/latest.json` "
        f"files, fill in a §2 sub-section (results table + takeaway), "
        f"remove the row from §1, and update §3/§4/§5 as evidence lands.\n"
        f"7. **Iteration budget:** up to {max_iterations} batches total "
        f"this session. A batch = 8 experiments, so up to "
        f"{max_iterations * 8} experiments over the session. "
        f"Spend them on evidence-based decisions: every new batch "
        f"should be informed by something you wrote down in §3-§5.\n"
        f"8. **Stop condition.** When you have a single submission that "
        f"clearly dominates (highest `throughput` while clearing the "
        f"`mean@16` floor) AND a tight memory.md, produce a final "
        f"2-paragraph report naming that submission, its "
        f"`content_hash`, its numbers, and the key design decisions.\n\n"
        f"Begin with step 0 — open and summarise memory.md, then "
        f"propose the theme + knob matrix for the first batch."
    )


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def _pretty_content_blocks(blocks) -> str:
    """Short human-readable summary of the assistant's response blocks."""
    out: List[str] = []
    for b in blocks:
        t = getattr(b, "type", None)
        if t == "text":
            txt = b.text.strip()
            if txt:
                out.append(f"[text] {txt[:800]}")
        elif t == "thinking":
            out.append(f"[thinking ~{len(b.thinking or '')} chars]")
        elif t == "tool_use":
            args_preview = json.dumps(b.input)[:500]
            out.append(f"[tool_use name={b.name} id={b.id} args={args_preview}]")
        else:
            out.append(f"[{t}]")
    return "\n".join(out)


def _blocks_for_messages(blocks) -> List[Dict[str, Any]]:
    """Serialize the SDK's content blocks back into dict form for the next turn."""
    msg_blocks: List[Dict[str, Any]] = []
    for b in blocks:
        t = b.type
        if t == "text":
            msg_blocks.append({"type": "text", "text": b.text})
        elif t == "thinking":
            msg_blocks.append({
                "type": "thinking",
                "thinking": b.thinking,
                "signature": getattr(b, "signature", ""),
            })
        elif t == "tool_use":
            msg_blocks.append({
                "type": "tool_use",
                "id": b.id,
                "name": b.name,
                "input": b.input,
            })
        # other block types (e.g. compaction) get passed through verbatim
        else:
            msg_blocks.append(b.model_dump())
    return msg_blocks


def run_agent(
    submission_name: str,
    min_mean: float,
    baseline_throughput: Optional[float],
    max_iterations: int,
    max_tool_calls: int,
    log_path: Path,
) -> Dict[str, Any]:
    client = anthropic.Anthropic()
    system_blocks = _build_system_blocks()

    initial_user_text = _build_initial_user_prompt(
        submission_name,
        min_mean,
        baseline_throughput,
        max_iterations,
    )
    if MEMORY_MD.is_file():
        try:
            memory_snapshot = MEMORY_MD.read_text(encoding="utf-8")
            initial_user_text += (
                "\n\n---\n\n"
                "## Current contents of `algorithm_scientist/memory.md`\n"
                "(this is your persistent notebook; update it during the session)\n\n"
                "```markdown\n"
                f"{memory_snapshot}\n"
                "```\n"
            )
        except OSError:
            pass
    messages: List[Dict[str, Any]] = [
        {"role": "user", "content": initial_user_text}
    ]

    turns = 0
    stop_reason: Optional[str] = None
    log_fp = log_path.open("w", encoding="utf-8")

    def _log(obj: Dict[str, Any]) -> None:
        log_fp.write(json.dumps(obj, ensure_ascii=False) + "\n")
        log_fp.flush()

    _log({
        "event": "start",
        "submission": submission_name,
        "objective": "maximise throughput subject to mean@16 >= floor",
        "min_mean@16": min_mean,
        "baseline_throughput": baseline_throughput,
        "max_iterations": max_iterations,
    })

    while turns < max_tool_calls:
        turns += 1

        # --- call Claude ---
        for attempt in range(4):
            try:
                with client.messages.stream(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    system=system_blocks,
                    tools=TOOLS,
                    thinking={"type": "adaptive"},
                    output_config={"effort": "high"},
                    messages=messages,
                ) as stream:
                    response = stream.get_final_message()
                break
            except anthropic.RateLimitError as e:
                wait = 2 ** attempt * 5
                print(f"[rate-limit] sleeping {wait}s …  ({e})", file=sys.stderr)
                time.sleep(wait)
            except anthropic.APIError as e:
                wait = 2 ** attempt * 5
                print(f"[api-error attempt {attempt}] {e} — retrying in {wait}s",
                      file=sys.stderr)
                time.sleep(wait)
        else:
            _log({"event": "api_failure_terminal"})
            raise SystemExit("Claude API kept failing after 4 retries.")

        stop_reason = response.stop_reason
        usage = response.usage.model_dump() if response.usage else {}
        print(f"\n========== turn {turns} "
              f"(stop_reason={stop_reason}, "
              f"usage={usage}) ==========")
        print(_pretty_content_blocks(response.content))

        _log({
            "event": "assistant_turn",
            "turn": turns,
            "stop_reason": stop_reason,
            "usage": usage,
            "content_summary": _pretty_content_blocks(response.content),
        })

        # --- append assistant message ---
        messages.append({
            "role": "assistant",
            "content": _blocks_for_messages(response.content),
        })

        # --- done? ---
        if stop_reason == "end_turn":
            print("\n[agent] model ended its turn — stopping.")
            break
        if stop_reason not in ("tool_use",):
            print(f"\n[agent] non-tool stop_reason={stop_reason!r} — stopping.")
            break

        # --- execute any tool_use blocks ---
        tool_results: List[Dict[str, Any]] = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            result = _dispatch_tool(block.name, block.input or {})
            print(f"\n[tool {block.name}] -> {result[:500]}"
                  f"{' …' if len(result) > 500 else ''}")
            _log({
                "event": "tool_result",
                "turn": turns,
                "tool": block.name,
                "tool_use_id": block.id,
                "input": block.input,
                "output_preview": result[:2000],
            })
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })

        if not tool_results:
            print("[agent] stop_reason=tool_use but no tool blocks — stopping.")
            break

        messages.append({"role": "user", "content": tool_results})

    log_fp.close()

    return {
        "turns": turns,
        "stop_reason": stop_reason,
        "log_path": str(log_path),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Claude-driven sparse-attention submission writer."
    )
    p.add_argument(
        "--submission-name",
        default=f"agent_{uuid.uuid4().hex[:8]}",
        help="Stem for submissions/<name>.py and .json.",
    )
    p.add_argument(
        "--min-mean-at-16",
        type=float,
        default=0.62,
        help=(
            "Minimum acceptable mean@16 on AIME24 (quality floor). "
            "The agent maximises throughput subject to this floor."
        ),
    )
    p.add_argument(
        "--baseline-throughput",
        type=float,
        default=5500,
        help=(
            "Optional tokens/sec reference (e.g. "
            "example_block_sparse_attention's throughput) to give the "
            "agent a target to beat."
        ),
    )
    p.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="How many benchmark runs the agent may attempt.",
    )
    p.add_argument(
        "--max-tool-calls",
        type=int,
        default=80,
        help="Hard cap on total LLM turns (tool calls + final answer).",
    )
    p.add_argument(
        "--logs-dir",
        type=Path,
        default=REPO_ROOT / "logs" / "auto_agent",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set.")

    args.logs_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_path = args.logs_dir / f"{args.submission_name}_{ts}.jsonl"

    print(f"[auto_agent] objective: maximise throughput s.t. "
          f"mean@16 >= {args.min_mean_at_16}")
    print(f"[auto_agent] submission_name     = {args.submission_name}")
    print(f"[auto_agent] min mean@16 (floor) = {args.min_mean_at_16}")
    if args.baseline_throughput is not None:
        print(f"[auto_agent] baseline_throughput = "
              f"{args.baseline_throughput:.0f} tok/s")
    print(f"[auto_agent] max_iterations      = {args.max_iterations}")
    print(f"[auto_agent] log                 -> {log_path}")

    result = run_agent(
        submission_name=args.submission_name,
        min_mean=args.min_mean_at_16,
        baseline_throughput=args.baseline_throughput,
        max_iterations=args.max_iterations,
        max_tool_calls=args.max_tool_calls,
        log_path=log_path,
    )

    print("\n[auto_agent] done.")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
