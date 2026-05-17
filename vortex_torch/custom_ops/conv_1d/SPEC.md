# `conv_1d` — per-row 1D causal conv across blocks

## Semantics

For each (batch, kv_head) row, slide a 1D causal kernel of `K` taps
across the per-block axis (axis 0) of the middle
`[bos, num_blocks_this_seq - eos)` range, computing
`out[p] = sum_k weight[k] * x[p - k]` per `(D0, D1)` lane.

## Calling signature

### flashinfer / CSR

```python
launch(
    x,           # RAGGED [N_total_blocks, D0, D1]
    out,         # RAGGED [N_total_blocks, D0, D1]
    weight,      # [K, D0, D1]  -- conv kernel
    indptr,      # [eff_bs + 1] int32
    bos, eos, topk_val,
    K,           # int, weight.shape[0]
    D0,          # int, x.shape[-2]
    D1,          # int, x.shape[-1]
    eff_batch_size,
)
```

### trtllm / block-table

```python
launch(
    x, out, weight,
    seqlens,             # [eff_bs] int32, tokens per row
    bos, eos, topk_val,
    K, D0, D1,
    block_size,
    max_blocks_per_seq,
    eff_batch_size,
)
```

## I/O contract

`weight.dtype` may differ from `x.dtype` — the kernel upcasts both to
fp32 for the multiply-accumulate, then casts the result back to
`x.dtype` on store. `K` is constexpr inside the kernel.

## Constraints (matcher kwargs)

Single `default` catch-all today. Future leaves likely specialise on:

| kwarg | typical operator | meaning |
|---|---|---|
| `K` | `eq` | exact tap count (e.g. `{"K": 3}` is a common short-conv case) |
| `D0`, `D1` | `eq` | head-dim |
| `dtype` | `in` | input/output dtype |
