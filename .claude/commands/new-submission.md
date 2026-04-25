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
