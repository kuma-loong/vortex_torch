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

Constraint precedence (most-specific wins; ties by `priority` desc)
is defined in `custom_ops/_dispatch_match.py`.
