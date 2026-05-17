# `union` — merge two `(block_table, seqlens)` pairs (trtllm-only)

## Semantics

For each (batch, kv_head) row, deduplicate the union of two sparse
block-tables `(bt_0, sl_0)` and `(bt_1, sl_1)` into the final
`(sparse_block_tables, sparse_seqlens)`. The dense path's last block
is placed at the tail with the correct partial-fill token count.
BOS/EOS slots are dedup-aware.

## Calling signature

```python
launch(
    dense_seqlens,         # [eff_bs] int32, tokens  (input)
    sparse_seqlens,        # [eff_bs] int32, tokens  (OUTPUT)
    dense_block_tables,    # [eff_bs, max_blocks_per_seq] int32  (input)
    block_tables_0,        # [eff_bs, max_blocks_per_seq] int32
    seqlens_0,             # [eff_bs] int32, tokens
    block_tables_1,        # [eff_bs, max_blocks_per_seq] int32
    seqlens_1,             # [eff_bs] int32, tokens
    sparse_block_tables,   # [eff_bs, max_blocks_per_seq] int32  (OUTPUT)
    eff_batch_size,
    max_blocks_per_seq,
    block_size,
)
```

## I/O contract

The two input `(block_table, seqlens)` pairs typically come from two
parallel `TopK(k)` ops; their `seqlens` are in tokens (multiples of
`block_size`, plus the optional partial fill on the last block).
`sparse_block_tables` + `sparse_seqlens` are framework-allocated.

## Constraints (matcher kwargs)

| kwarg | operator | meaning |
|---|---|---|
| `variant` | `eq` | algorithm: `"hash"` (default), `"sort"`, `"baseline"` |

The current default (`hash`) wins by 17-364× on the microbenchmark for
typical block counts. The sort and baseline leaves remain for
correctness audits.

Future leaves: constrain on `max_blocks_per_seq` (`le`) — the hash
shared-memory budget caps at `UNION_MAX_OUT=4096` today; an agent
specialising for very-large `max_blocks_per_seq` would land a wider
hash table or a different algorithm and constrain accordingly.
