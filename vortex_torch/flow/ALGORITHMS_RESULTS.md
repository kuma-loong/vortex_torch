# `flow/algorithms.py` — RULER results

RULER on **`examples/validation.jsonl`** (100 prompts, 64 new tokens each)
using **Qwen/Qwen3-1.7B** on **GPU 3**. All runs at identical settings to
isolate per-algorithm behaviour (knobs below match the existing
`submissions/_block_table_test/` baselines).

| # | Registered name | Class | Save / Load? | `disable_radix_cache` | RULER accuracy | Throughput (tok/s) | e2e (s) |
|---|---|---|---|---|---|---|---|
| 1 | `block_sparse_attention`          | `BlockSparseAttention`          | no  | (not set) | **0.98** | 2742 | 2.33 |
| 2 | `gqa_block_sparse_attention`      | `GQABlockSparseAttention`       | no  | (not set) | **1.00** | 2668 | 2.40 |
| 3 | `gqa_quest_sparse_attention`      | `GQAQuestSparseAttention`       | no  | (not set) | **1.00** | 2692 | 2.38 |
| 4 | `lserve_sparse_attention`         | `LServeSparseAttention`         | no  | (not set) | **1.00** | 2733 | 2.34 |
| 5 | `masked_quest_sparse_attention`   | `MaskedQuestSparseAttention`    | no  | (not set) | **1.00** | 2672 | 2.39 |
| 6 | `centered_block_sparse_attention` | `CenteredBlockSparseAttention`  | no  | (not set) | **0.98** | 2671 | 2.39 |
| 7 | `running_avg_block_sparse`        | `RunningAvgBlockSparse`         | **yes** | **true** (required) | **1.00** | 1984 | 3.22 |
| 8 | `venergy_gated_centroid`          | `VEnergyGatedCentroid`          | no  | (not set) | **1.00** | 2704 | 2.37 |

(Throughput is the warm decode rate RULER reports as `throughput`. `running_avg_block_sparse` is ~28% slower because `disable_radix_cache` disables sglang's prefix-sharing radix tree — required whenever a flow's `forward_indexer` writes per-request state via `Save(...)`.)

## Common knobs (identical across all 8 runs)

```json
{
  "vortex_attention_backend": "flashinfer",
  "vortex_block_size": 16,
  "vortex_topk_val": 29,
  "vortex_topk_ratio": 0.0625,
  "vortex_block_reserved_bos": 1,
  "vortex_block_reserved_eos": 2,
  "vortex_workload_chunk_size": 32,
  "vortex_layers_skip": [0],
  "kv_cache_dtype": "auto",
  "mem_fraction_static": 0.8
}
```

`disable_radix_cache: true` added only for `running_avg_block_sparse` (Save/Load required).

## Why one submission file per algorithm

`vortex_torch.engine.sgl._check_disable_radix_cache` does a **text scan** of
the module file passed via `vortex_module_path`. Pointing all 8 submissions
at `vortex_torch/flow/algorithms.py` (which contains the `Save(...)` site
inside `RunningAvgBlockSparse`) would force `disable_radix_cache: true` on
every algorithm — incorrect for the 7 that don't actually use Save/Load,
and a measurable throughput hit. The fix: extract each `@register` class
into its own file under `submissions/_flow_algorithms_test/<name>.py`, and
add a `_sub` suffix to the registered name (the sglang process auto-imports
`vortex_torch.flow.algorithms` at startup, so re-registering the same name
would collide).

## Files

  * Per-algorithm submission flow:
    `submissions/_flow_algorithms_test/<algo>.py`
  * Per-algorithm submission config:
    `submissions/_flow_algorithms_test/<algo>.json`
  * Per-algorithm RULER artefacts:
    `summary_ruler_submissions/_flow_algorithms_test/<algo>/latest.json`

## Reproduction

```bash
# Preflight
for cfg in submissions/_flow_algorithms_test/*.json; do
  python -c "from vortex_torch.engine.sgl import check_engine_config; check_engine_config('$cfg')"
done

# Sequential RULER (parallel >2 hits sglang server-startup collisions on
# this host — sequential is reliable and finishes in ~3-4 min total).
for cfg in submissions/_flow_algorithms_test/*.json; do
  CUDA_VISIBLE_DEVICES=<free_gpu> \
    python algorithm_scientist/run_ruler.py --config "$cfg"
done
```
