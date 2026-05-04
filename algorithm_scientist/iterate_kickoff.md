# Iterate kickoff prompt

A copy-pasteable prompt for booting a fresh Claude Code session
straight into the long-horizon iterate loop on this repo. Paste the
fenced block below into a new Claude Code session running in this
directory; the agent will identify its tag, bootstrap from
`AI/AGENTS.md` + tutorials + `papers/guide.md` + `memory.md`, and
start the first batch.

## How to use

1. Open a fresh Claude Code session in this repo (`cd
   /home/zhuominc/new_envs/vortex_new/vortex_torch`, then `claude`).
2. Paste the entire block below as your first message.
3. The agent will reply with its chosen tag, then proceed
   autonomously: design batch → pre-flight → `/batch-benchmark` →
   wait + read/invent/design/analyse → loop.

You don't need to add anything else. The prompt is self-contained.

---

## Prompt — paste this into Claude Code

```
You are an algorithm scientist iterating on vortex_torch sparse-attention
submissions. Your goal: maximise AIME24 decoding throughput while
keeping mean@16 above the quality floor. Operate the full
long-horizon iterate loop autonomously — do not ask me to confirm
each step.

Bootstrap (do these in order; each file once):

1. Pick your agent tag. Default to a sanitized lowercase form of
   your model name (e.g. claude_opus_4_7, claude_sonnet_4_6,
   claude_haiku_4_5). State the tag back in your first reply, then
   reuse it for the entire session. mkdir -p submissions/<tag> if it
   does not exist.

2. Read these files in this order:
   - .claude/CLAUDE.md
   - AI/AGENTS.md
   - AI/tutorials/overview.md
   - AI/tutorials/program_create_cache.md
   - AI/tutorials/program_forward_cache.md
   - AI/tutorials/program_forward_indexer.md
   - AI/tutorials/cache_op.md
   - AI/tutorials/indexer_op.md
   - papers/guide.md  (especially §11 axes, §14 catalog, §16 off-catalog prompts)
   - algorithm_scientist/memory.md  (persistent state)

3. If memory.md §1 already shows a batch RUNNING, do NOT launch
   another — every local GPU is consumed by it. Jump to step 7 and
   spend the wait time on read/invent/design/analyse activities,
   then resume from §1 once it terminates.

Iterate loop (repeat until I say stop):

4. Design the next batch.
   - N = `nvidia-smi -L | wc -l`  (one variant per local GPU).
   - State the theme in one short paragraph.
   - List the knob matrix — one knob varied per variant.
   - **At least one variant must be off-catalog** (papers/guide.md §16):
     a paper combination, an inversion, an untried knob, or a
     first-principles answer. Pure parameter sweeps and direct
     paper replications do not count.
   - Pre-register the off-catalog hypothesis as a one-sentence row
     in memory.md §3 the moment you launch.

5. Write the 2*N files at:
       submissions/<tag>/batch_<x>_id<y>.py
       submissions/<tag>/batch_<x>_id<y>.json
   for y ∈ {0 .. N-1}, where <x> = number of existing
   `submissions/<tag>/batch_*_id0.json` files.
   Each .py uses `@register("<tag>_batch_<x>_id<y>_cls")` (must be
   globally unique). Each .json sets:
       "vortex_module_path": "submissions/<tag>/batch_<x>_id<y>.py"
       "vortex_module_name": "<tag>_batch_<x>_id<y>_cls"

6. Pre-flight all N locally (CPU-only):
       for i in $(seq 0 $((N - 1))); do
         python -c "from vortex_torch.engine.sgl import check_engine_config; check_engine_config('submissions/<tag>/batch_<x>_id${i}.json')"
       done
   Drop or fix any failing variant before launch.

7. Launch via the /batch-benchmark slash command, passing the N
   names: `batch_<x>_id0 batch_<x>_id1 ... batch_<x>_id<N-1>`.
   Add a row to memory.md §1.

8. While the batch runs (8+ hours), on each polling cycle do
   exactly ONE of:
   (a) read the next file in priority order
       (AI/tutorials → AI/developer_guides → papers/ →
        vortex_torch/flow/algorithms.py → vortex_torch/{indexer,cache}/* → csrc/);
       append one bullet to memory.md §7.
   (b) invent — pick a prompt from papers/guide.md §16 and sketch
       the off-catalog variant for the next batch.
   (c) design the rest of the next batch (do not launch — concurrent
       batches OOM the shared GPUs).
   (d) analyse children that have already produced
       summary_submissions/<tag>/<stem>/latest.json; fill
       memory.md §2.
   Do not idle; do not poll without doing one of (a)-(d).

9. When wait returns: read all N summaries
   (summary_submissions/<tag>/batch_<x>_id<y>/latest.json),
   produce a comparison table (name | content_hash | mean@16 |
   pass@16 | throughput | e2e_time), append it to memory.md §2,
   remove the §1 row, and update §3 (hypotheses) / §4 (anti-
   patterns) / §5 (winners) as the evidence lands.

10. Loop back to step 4.

Hard rules (the framework will reject violations):
- No native torch ops inside create_cache / forward_cache /
  forward_indexer.
- Each op instance is for one call site — never share.
- Do not declare "k" or "v" in create_cache.
- forward_indexer must end in topK(score, o, ctx=ctx) or
  approxTopK(tolerate_ratio=...)(score, o, ctx=ctx); score must be
  RAGGED [S, 1, 1].
- Cache-side reductions support dim ∈ {1, 2} only.
- If a field is read+written across steps via Load/Save, zero it
  with CFill(0.0) in forward_cache.
- If forward_indexer uses Save(...), the engine JSON must set
  "disable_radix_cache": true.

Begin now with step 1: state your tag and confirm the bootstrap
reads in your first reply, then proceed straight to designing
batch 0 (or to step 8 if memory.md §1 shows an in-flight batch).
```

---

## What the agent will do

After you paste, the agent should reply with something like:

```
tag:                  claude_opus_4_7
bootstrap reads:      done (AGENTS.md + 6 tutorials + papers/guide.md + memory.md)
in-flight check:      memory.md §1 empty — launching fresh
batch 0 theme:        <one-paragraph theme>
knob matrix:          <table>
off-catalog variant:  <which id, hypothesis>
```

…and then proceed to write the files, pre-flight, and call
`/batch-benchmark`. From there the loop runs autonomously.

If you want to seed the agent with a specific theme (e.g. "explore
Prism dual-band centroid this batch"), append one extra line at the
end of the prompt: `First batch theme: <your theme>.` Otherwise it
picks from `papers/guide.md §16` and `memory.md §6` (open
questions / backlog).

## Stopping the loop

Either tell the agent "stop" between batches, or close the session.
The next session that pastes this prompt will resume cleanly: it
re-reads `memory.md`, sees the most recent completed batch in §2,
and designs the next one accordingly. State lives in `memory.md`
and the `submissions/<tag>/` and `summary_submissions/<tag>/` trees
— not in the conversation.
