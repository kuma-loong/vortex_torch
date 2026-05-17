# Why `Qwen/Qwen3-30B-A3B` was broken under vortex_torch (and how it's fixed)

## TL;DR

The vortex KV cache stores K/V in a **block-interleaved** physical layout
(`(token//page)·page·num_kv_head + head·page + token%page`). sglang's
**fused-set-kv-buffer** path — which is silently invoked from inside fused
RoPE on every model whose attention layer routes K/V writes through
`apply_qk_norm_rope` — assumes the **standard token-major** layout and
writes through a `view(num_tokens, -1)` of the same buffer. Vortex never
opted out of that path, so on `Qwen3-30B-A3B` (which uses the fused
RoPE+set-KV path) every K and V got written to the **wrong physical
slot** in the vortex cache. Decode then read garbage (or stepped past
the allocation), producing `~0%` RULER accuracy or
`CUDA illegal memory access` — exactly the two symptoms you saw.

`Qwen3-4B` and `Qwen3-32B` use dense `Qwen3Attention.forward_prepare_native`,
which calls `self.rotary_emb(positions, q, k)` **without**
`fused_set_kv_buffer_arg`. The KV write therefore falls through to
`forward_batch.token_to_kv_pool.set_kv_buffer(...)`, which is vortex's
own block-aware launcher (`vortex_torch/cache/triton_kernels/set_kv.py`).
That path knows the layout, so the dense Qwen3 models worked fine.

## The exact code paths

### vortex's K/V layout (the source of truth)

`vortex_torch/cache/triton_kernels/set_kv.py:28-36` — note how the
"position" of a token in the physical buffer is **not** `token_position`
but a stride-rewritten index that interleaves `num_kv_head` and `page_size`:

```python
token_position = tl.load(loc + token_id)
position_trans = (token_position // PAGE_SIZE) * (PAGE_SIZE * NUM_KV_HEAD) \
                 + head_id * PAGE_SIZE \
                 + token_position %  PAGE_SIZE

dst_k_ptr = k_cache + position_trans * HEAD_DIM + dim
dst_v_ptr = v_cache + position_trans * HEAD_DIM + dim
```

This is the *only* correct way to write into a `VortexCachePool`
buffer. The buffer itself is allocated as

```
(num_pages * num_blocks_per_page, block_size, head_dim)
```

(see `vortex_torch/engine/sgl/memory_pool.py:_create_buffers`).

### sglang's fused-set-kv-buffer assumes the standard layout

`third_party/sglang/v0.5.9/sglang/python/sglang/srt/models/utils.py:117`:

```python
def create_fused_set_kv_buffer_arg(value, layer, forward_batch):
    ...
    k_buffer = token_to_kv_pool.get_key_buffer(layer_id)
    v_buffer = token_to_kv_pool.get_value_buffer(layer_id)
    return FusedSetKVBufferArg(
        value=value,
        k_buffer=k_buffer.view(k_buffer.shape[0], -1),   # <- token-major view
        v_buffer=v_buffer.view(v_buffer.shape[0], -1),
        ...
        cache_loc=forward_batch.out_cache_loc,
    )
```

The `FusedSetKVBufferArg` is consumed inside the fused RoPE kernel
(invoked from `sgl_kernel`), which writes
`k_buffer[cache_loc[t]] = rotated_k[t]` (and similarly for v). That
expects `k_buffer` to be `[num_tokens, num_kv_head*head_dim]` — i.e.
token-major. With vortex's pool the same `view(N, -1)` puts you on a
fundamentally different physical mapping: the indices addressable as
`k_buffer[cache_loc[t]]` are no longer the slots vortex's decode reads
from.

### What triggers it on Qwen3-MoE but not Qwen3 dense

`third_party/sglang/v0.5.9/sglang/python/sglang/srt/models/qwen3_moe.py:586-611` —
`Qwen3MoeAttention.apply_qk_norm_rope` builds the fused arg whenever
`enable_fused_set_kv_buffer(forward_batch)` is True and the layer is
"compatible":

```python
q, k = self.rotary_emb(
    positions, q, k,
    fused_set_kv_buffer_arg=(
        create_fused_set_kv_buffer_arg(value=v, layer=self.attn,
                                       forward_batch=forward_batch)
        if enable_fused_set_kv_buffer(forward_batch)
           and self.compatible_with_fused_kv_buffer
        else None
    ),
)
```

…and then `Qwen3MoeAttention.forward_core` deliberately **suppresses**
the regular `set_kv_buffer` call when the fused path took it:

```python
must_save_kv = self._used_fused_qk_norm_rope_last_call
save_kv_cache = must_save_kv or not (
    enable_fused_set_kv_buffer(forward_batch)
    and self.compatible_with_fused_kv_buffer
)
attn_output = self.attn(q, k, v, fb, save_kv_cache=save_kv_cache)
```

So on `Qwen3-30B-A3B` the fused kernel wrote into the buffer with the
wrong layout **and** vortex's own block-aware launcher never ran. Cache
corruption — full stop.

`Qwen3Attention.forward_prepare_native` in `qwen3.py:141-153` does
plain `self.rotary_emb(positions, q, k)` with no `fused_set_kv_buffer_arg`,
so `attn(q, k, v, forward_batch)` always reaches
`forward_batch.token_to_kv_pool.set_kv_buffer(...)` — vortex's launcher
— which is correct. Hence `Qwen3-4B` and `Qwen3-32B` work.

The gate that decides between the two paths is
`enable_fused_set_kv_buffer` in `models/utils.py:107`:

```python
def enable_fused_set_kv_buffer(forward_batch):
    return (
        _is_cuda
        and hasattr(forward_batch.token_to_kv_pool, "dtype")
        and forward_batch.token_to_kv_pool.dtype == torch.bfloat16
        and not isinstance(forward_batch.token_to_kv_pool, SWAKVPool)
    )
```

With `vortex_dtype="bfloat16"` (the default, and what `run_ruler.py`
sets) every condition is True for `VortexCachePool`, so the gate
returns True and the fused (incompatible) path is selected.

## The fix

Two tiny changes:

1. `third_party/sglang/v0.5.9/sglang/python/sglang/srt/models/utils.py`
   — extend the gate with a duck-typed opt-out so any pool with a
   non-token-major layout can disable fused-set-kv-buffer without
   sglang having to import vortex (which would create a dependency
   cycle):

   ```python
   and getattr(forward_batch.token_to_kv_pool,
               "supports_fused_set_kv_buffer", True)
   ```

   (Existing pools without the attribute default to `True`, i.e. no
   behaviour change for non-vortex deployments.)

2. `vortex_torch/engine/sgl/memory_pool.py` — declare the opt-out on
   `VortexCachePool`:

   ```python
   class VortexCachePool(KVCache):
       supports_fused_set_kv_buffer = False
       ...
   ```

After the fix, on `Qwen3-30B-A3B`:

- `enable_fused_set_kv_buffer(...)` returns False under vortex.
- `Qwen3MoeAttention.apply_qk_norm_rope` passes
  `fused_set_kv_buffer_arg=None`, so the fused kernel no longer writes
  K/V.
- `Qwen3MoeAttention.forward_core` therefore computes
  `save_kv_cache=True` (because `not (False and X) == True`), and the
  attention layer calls `forward_batch.token_to_kv_pool.set_kv_buffer(...)`
   — vortex's block-aware launcher — exactly as it does for the dense
  Qwen3 models that already worked.

The same fix transparently rescues every other sglang model that
follows the fused-RoPE+set-KV pattern (`gpt_oss`, `llada2`,
`bailing_moe`, …) when paired with a vortex KV pool.

## How to verify

```bash
conda activate vortex_v1
CUDA_VISIBLE_DEVICES=4 python examples/run_ruler.py
# Expect: Ruler Accuracy: >= ~97% (matches Qwen3-4B / Qwen3-32B baselines)
```

(`CUDA_VISIBLE_DEVICES=4` per the in-house note that GPU 0 has hardware
issues; any free non-zero GPU is fine.)

## Files touched

- `third_party/sglang/v0.5.9/sglang/python/sglang/srt/models/utils.py`
  — 1-line addition + comment to `enable_fused_set_kv_buffer`.
- `vortex_torch/engine/sgl/memory_pool.py` — class attribute
  `supports_fused_set_kv_buffer = False` on `VortexCachePool` + comment
  explaining the layout incompatibility.

No kernel changes, no model-side changes, no behaviour change for
non-vortex deployments.
