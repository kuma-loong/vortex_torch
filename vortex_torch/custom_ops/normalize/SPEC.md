# `normalize` — per-row L2 normalize over RAGGED-packed blocks

## Semantics

Per (batch, kv_head) row, divide each block element by the L2 norm
(`sqrt(sum(x*x))` per `(D0, D1)` lane) across the middle
`[bos, num_blocks_this_seq - eos)` range. Tail/head reservation blocks
are passed through unchanged.

## Calling signature

### flashinfer / CSR

```python
launch(
    x,           # RAGGED [N_total_blocks, D0, D1]
    out,         # RAGGED [N_total_blocks, D0, D1]
    indptr,      # [eff_bs + 1] int32
    bos,         # int
    eos,         # int
    topk_val,    # int
    D0,          # int, x.shape[-2]
    D1,          # int, x.shape[-1]
    eff_batch_size,   # int = batch * num_kv_heads  -- grid size
)
```

### trtllm / block-table

```python
launch(
    x, out,
    seqlens,             # [eff_bs] int32, tokens per row
    bos, eos, topk_val,
    D0, D1,
    block_size,
    max_blocks_per_seq,
    eff_batch_size,
)
```

## I/O contract

Identical to `softmax` (RAGGED-packed `x` and `out`, same dtype). No
`scale` argument — normalize divides by the per-row L1 sum.

## Constraints (matcher kwargs)

Single `default` catch-all today. Future shape/dtype-specialised
leaves should mirror `softmax`'s constraint surface (`D0`, `D1`,
`dtype`, trtllm-only `block_size`).
