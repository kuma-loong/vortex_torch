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
    D0,          # int, x.shape[-2]   (real, used for strides)
    D1,          # int, x.shape[-1]   (real, used for strides)
    D0_PAD,      # int, x.padded_shape[-2]   (pow2, tile constexpr)
    D1_PAD,      # int, x.padded_shape[-1]   (pow2, tile constexpr)
    eff_batch_size,   # int = batch * num_kv_heads  -- grid size
)
```

### trtllm / block-table

```python
launch(
    x, out,
    seqlens,             # [eff_bs] int32, tokens per row
    bos, eos, topk_val,
    D0, D1, D0_PAD, D1_PAD,
    block_size,
    max_blocks_per_seq,
    eff_batch_size,
)
```

## `padded_shape` support

Same pattern as `softmax`: `D0`/`D1` are the real (stride) sizes;
`D0_PAD`/`D1_PAD` are the pow2 round-ups used by `tl.arange` and
tile-shape constexprs. Padded lanes load `0.0` so they contribute zero
to the squared sum, and the store mask suppresses writes to them. When
`shape == padded_shape` Triton specialises the unmasked branch.

## I/O contract

Identical to `softmax` (RAGGED-packed `x` and `out`, same dtype). No
`scale` argument — normalize divides by the per-row L1 sum.

## Constraints (matcher kwargs)

Single `default` catch-all today. Future shape/dtype-specialised
leaves should mirror `softmax`'s constraint surface (`D0`, `D1`,
`dtype`, trtllm-only `block_size`).
