# `topk_output` — k-largest blocks per row, writing the sparse index buffer

## Semantics

For each (batch, kv_head) row, select the top-`k` blocks by score
(reading `x`) over the middle `[bos, num_blocks_this_seq - eos)` range
and write their indices into the sparse output. `k` is derived from
the difference `sparse_seqlens[row] - bos - eos` at the call site
(set by the planner per request).

## Calling signature

### flashinfer / CSR

```python
launch(
    x,                    # RAGGED [N_total_blocks, 1, 1]  scores
    dense_kv_indptr,      # [eff_bs + 1] int32  (input)
    sparse_kv_indptr,     # [eff_bs + 1] int32  (input — k per row)
    dense_kv_indices,     # [N_total_blocks] int32  (input)
    sparse_kv_indices,    # [N_sparse_blocks] int32  (OUTPUT)
    eff_batch_size,       # int
    reserved_bos,         # int
    reserved_eos,         # int
    max_num_pages,        # int  -- per-request block cap
)
```

### trtllm / block-table

```python
launch(
    x,                    # RAGGED [eff_bs * max_blocks_per_seq, 1, 1]  scores
    dense_seqlens,        # [eff_bs] int32, tokens
    sparse_seqlens,       # [eff_bs] int32, tokens (input — k * block_size)
    dense_block_tables,   # [eff_bs, max_blocks_per_seq] int32
    sparse_block_tables,  # [eff_bs, max_blocks_per_seq] int32  (OUTPUT)
    eff_batch_size,
    reserved_bos,
    reserved_eos,
    max_blocks_per_seq,
    block_size,
)
```

### Special: `approx=True` (flashinfer-only)

```python
launch_factory = find('topk_output', 'flashinfer', approx=True)
launch = launch_factory(tolerate_ratio=0.05)   # baked into ``__TOLERATE_RATIO__`` substitution
launch(x, ..., max_num_pages)
```

`tolerate_ratio` is a per-call knob — each distinct ratio yields a
separately-cached compiled module.

## I/O contract

  * `x` is a score tensor (`D0=1`, `D1=1`); shape and dtype set by the
    indexer's last reduce/scoring op.
  * `sparse_kv_indices` / `sparse_block_tables` are preallocated by
    the planner; the kernel writes top-`k` block indices in-place.
  * BOS/EOS blocks are reserved at fixed positions by the planner
    (not by this kernel) — this op only fills the middle slots.

## Constraints (matcher kwargs)

| kwarg | typical operator | meaning |
|---|---|---|
| `max_topk_val` | `le` | upper bound on `k` for this call; lets `k_96` / `k_128` / `k_256` opt in only when `k` fits |
| `approx` | `eq true` | exclusive — when `True`, only the `approx` leaf matches |

Future shape-specialised leaves: constrain on `dtype` (score precision)
and/or `block_size` (trtllm). Multi-key constraints win by specificity
— e.g. `{"max_topk_val": {"le": 96}, "dtype": "bfloat16"}` beats
`{"max_topk_val": {"le": 96}}` when both match.
