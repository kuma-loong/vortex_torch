# `softmax` — per-row softmax over RAGGED-packed blocks

## Semantics

Per (batch, kv_head) row, compute `out = softmax(scale * x)` over the
middle `[bos, num_blocks_this_seq - eos)` of the row's RAGGED block
segment. The first `bos` and the last `eos` blocks are passed
through unchanged — they're typically the always-on (sink + recent)
tokens whose attention is already saturated.

## Calling signature (returned by `dispatch()`)

The leaf's `dispatch()` returns a plain Python callable. The trailing
positional arg is always `eff_batch_size` (the 1D grid size); the
preceding args are the kernel's positional parameters.

### flashinfer / CSR

```python
launch(
    x,           # RAGGED [N_total_blocks, D0, D1]  bf16/fp16/fp32
    out,         # RAGGED [N_total_blocks, D0, D1]  same dtype as x
    indptr,      # [eff_bs + 1] int32  -- ctx.metadata.dense_kv_indptr
    scale,       # float, multiplied into x before softmax
    bos,         # int, head-skip blocks
    eos,         # int, tail-skip blocks
    topk_val,    # int, used by ``if num_blocks <= bos+eos+topk_val: skip``
    D0,          # int, x.shape[-2]
    D1,          # int, x.shape[-1]
    eff_batch_size,   # int = batch * num_kv_heads  -- grid size
)
```

### trtllm / block-table

```python
launch(
    x,
    out,
    seqlens,             # [eff_bs] int32, tokens per row  -- dense_seqlens
    scale,
    bos,
    eos,
    topk_val,
    D0,
    D1,
    block_size,          # int, tokens per block
    max_blocks_per_seq,  # int, dense_block_tables.shape[1]
    eff_batch_size,
)
```

## I/O contract

  * `x`, `out`: RAGGED-packed blocks; `x.shape[0] == out.shape[0]`,
    `x.dtype == out.dtype`. The kernel must not allocate; `out` is
    pre-sized by the framework.
  * **Per-row range** lives in `indptr` (flashinfer) or `seqlens +
    block_size` (trtllm); see `block_size`/`max_blocks_per_seq` notes
    in `custom_ops/AGENTS.md`.
  * `bos`/`eos`/`topk_val` are the standard reservation/skip knobs
    used by every Schedule.S op.

## Constraints (matcher kwargs)

`find('softmax', '<backend>')` currently routes on no kwargs — every
backend has a single `default` catch-all leaf. Future
shape/dtype-specialised leaves should constrain on:

| kwarg | typical operator | meaning |
|---|---|---|
| `D0`, `D1` | `eq` | head-dim specialisation |
| `dtype` | `in` | input dtype (`"bfloat16"`, `"float16"`, `"float32"`) |
| `block_size` | `eq` | trtllm-only; constexpr block size |

## `padded_shape` support

The launcher passes both real (`shape[1]`, `shape[2]`) and pow2-padded
(`padded_shape[1]`, `padded_shape[2]`) inner sizes. Inside the kernel:

  * `tl.arange` / accumulator tile shapes use `x_D0_PAD` / `x_D1_PAD`
    (must be pow2).
  * Memory strides use the real `x_D0` / `x_D1`.
  * A `NEEDS_INNER_MASK: tl.constexpr` branch (set when
    `*_PAD != *`) decides whether the per-block `p_mask` is AND'd
    with the inner-dim validity mask. **When `shape == padded_shape`,
    Triton specializes the unmasked branch — no inner-dim mask is
    computed and `tl.load` / `tl.store` see only `p_mask`.**

All Triton leaves (`softmax`, `normalize`, `conv_1d`, `reduce_dim0`)
now follow this pattern. See `custom_ops/AGENTS.md § padded_shape
support` for the support matrix and per-op identity values.

Constraint precedence (most-specific wins; ties by `priority` desc)
is defined in `custom_ops/_dispatch_match.py`.
