# `reduce_dim0` — cross-row reduce over the packed RAGGED leading axis

## Semantics

Per (batch, kv_head) row, collapse the row's `[bos, num_blocks_this_seq
- eos)` block segment into a single `(D0, D1)` tile by the chosen
reduction (`sum` / `mean` / `max` / `min` / `l2norm`). RAGGED in →
BATCHED out.

## Calling signature

### flashinfer / CSR

```python
launch(
    x,           # RAGGED [N_total_blocks, D0, D1]
    out,         # BATCHED [eff_bs, D0, D1]
    indptr,      # [eff_bs + 1] int32
    bos, eos,
    D0,          # int, x.shape[-2]
    D1,          # int, x.shape[-1]
    eff_batch_size,
)
```

### trtllm / block-table

```python
launch(
    x, out,
    seqlens,             # [eff_bs] int32
    bos, eos,
    D0, D1,
    block_size,
    max_blocks_per_seq,
    eff_batch_size,
)
```

Note: no `topk_val` arg (unlike softmax/normalize/conv_1d).

## I/O contract

  * `x.dtype` may be bf16/fp16/fp32. `out.dtype` likewise; the kernel
    uses `out.dtype.element_ty` for the final cast.
  * fp8 outputs are **not** supported by the current leaves (they'd
    need range-clamping). Add a `<rt>_fp8/` bucket with a tighter
    `dtype` constraint when needed.

## Constraints (matcher kwargs)

The reduction kind is the **routing key**:

| kwarg | operator | meaning |
|---|---|---|
| `variant` | `eq` | one of `"sum"`, `"mean"`, `"max"`, `"min"`, `"l2norm"` |

Future shape-specialised leaves should add `eq` constraints on `D0` /
`D1` / `dtype` on top of `variant` — the matcher's specificity
tie-breaker then picks the more-constrained leaf when both apply.
