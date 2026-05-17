# `topk` — `TopK(k)` indexer op (trtllm-only)

## Semantics

For each (batch, kv_head) row, pick the top-`k` blocks by score from
the dense block-table's middle `[bos, num_blocks_this_seq - eos)`
range. Append `bos` head blocks and `eos` tail blocks back into the
sparse block-table at their reserved positions; write the per-row
sparse `seqlens` (in tokens, respecting last-block partial fill). When
fewer than `k + bos + eos` blocks exist, the entire dense row is
copied through (kernel-side fallback — no planner fill required).

## Calling signature

```python
launch(
    x,                    # RAGGED [eff_bs * max_blocks_per_seq, 1, 1]  scores
    dense_seqlens,        # [eff_bs] int32, tokens (input)
    sparse_seqlens,       # [eff_bs] int32, tokens (OUTPUT — written by kernel)
    dense_block_tables,   # [eff_bs, max_blocks_per_seq] int32 (input)
    sparse_block_tables,  # [eff_bs, max_blocks_per_seq] int32 (OUTPUT)
    eff_batch_size,
    reserved_bos,
    reserved_eos,
    max_blocks_per_seq,
    block_size,
    topk_val,             # int -- explicit k (excludes bos + eos)
)
```

## Differences vs `topk_output / trtllm`

  * `sparse_seqlens` is **OUTPUT** here (mutable arg). `topk_output`
    treats it as input.
  * `topk_val` is an explicit runtime arg (not derived from
    `sparse_seqlens`).
  * Row-too-small fallback is inside the kernel.

## I/O contract

Both `sparse_seqlens` and `sparse_block_tables` are preallocated as
auto-intermediate tensors by `generate_entry_point` (see the
multi-output op support in `vortex_torch/indexer/select.py`). The
caller does not pass these as graph inputs.

## Constraints (matcher kwargs)

Single `default` catch-all today (one leaf). Future leaves likely
specialise on:

| kwarg | typical operator | meaning |
|---|---|---|
| `topk_val_max` | `le` | upper bound on `k` |
| `block_size` | `eq` | constexpr block size |
| `dtype` | `in` | score dtype |
