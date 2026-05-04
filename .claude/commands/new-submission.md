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
2. Pick a starting template from the committed example library
   at `submissions/*.{py,json}` (top level — agent-tagged dirs
   are NOT examples). Available examples:
     - `example_block_sparse_attention` — minimal centroid flow
     - `gqa_quest_approx` — QUEST envelope + `approxTopK`
     - `kimi_v0` — tight throughput config (small topk, block_size=32)
     - `oai_v0` — wider budget + aggressive `vortex_layers_skip`
   Pick the one closest to the user's described idea (ask in one
   short message if ambiguous). Copy `submissions/<example>.py`
   and `submissions/<example>.json` to
   `submissions/<tag>/$1.py` and `submissions/<tag>/$1.json`.
3. Rename the class, the `@register("...")` string (must be
   globally unique — include `<tag>` and `$1`), and the JSON's
   `vortex_module_name` / `vortex_module_path` to match the new
   location.
4. If the user's idea differs from the picked example, customise
   `create_cache`, `forward_cache`, `forward_indexer` accordingly.
5. Run the cheap local pre-flight:
   `python -c "from vortex_torch.engine.sgl import check_engine_config; check_engine_config('submissions/<tag>/$1.json')"`
6. Report the result. Do NOT launch the benchmark automatically —
   that's `/benchmark $1` (debug, single variant) or
   `/batch-benchmark` (the sanctioned batch that fills every
   local GPU).

Follow every rule in [AI/AGENTS.md](AI/AGENTS.md). Never use native
torch ops inside the three vFlow methods.
