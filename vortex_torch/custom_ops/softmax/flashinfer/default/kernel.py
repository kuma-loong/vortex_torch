"""softmax @triton.jit kernel — flashinfer / CSR layout.

Per-row layout: the per-(req, kv_head) RAGGED slice
``x[indptr[pid] : indptr[pid+1])`` along the block axis. Online
two-pass softmax over the middle ``[bos, block_len-eos)`` slots; BOS /
EOS reserved blocks are skipped (the trtllm topk kernel re-uses them).
"""
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    x,
    out,
    indptr,
    scale,
    bos: tl.constexpr,
    eos: tl.constexpr,
    topk_val: tl.constexpr,
    x_D0: tl.constexpr,
    x_D1: tl.constexpr,
    x_D0_PAD: tl.constexpr,
    x_D1_PAD: tl.constexpr,
    BLOCK_P: tl.constexpr = 512,
):
    """Two-pass softmax over the per-row block axis.

    ``x_D0`` / ``x_D1`` are the **real** inner sizes used for memory
    strides; ``x_D0_PAD`` / ``x_D1_PAD`` are their next-pow2 round-ups
    used by ``tl.arange`` / tile-shape constexprs. When the inner dims
    are already pow2 (``*_PAD == *``) the constexpr ``NEEDS_INNER_MASK``
    is False and Triton specializes a branch that drops the inner-dim
    mask entirely — only the per-block ``p_mask`` is fused into
    load/store. Padded models keep the masked path: padded lanes carry
    ``-inf`` on load so they contribute zero mass to softmax, and the
    same mask suppresses the store.
    """
    pid = tl.program_id(0)

    start = tl.load(indptr + pid)
    _end = tl.load(indptr + pid + 1)
    num_blocks_this_seq = _end - start

    threshold: tl.constexpr = bos + eos + topk_val
    if num_blocks_this_seq <= threshold:
        return

    num_blocks_to_compute = num_blocks_this_seq - bos - eos
    if num_blocks_to_compute <= 0:
        return

    block_stride = x_D0 * x_D1

    x_base_ptr = x + (start + bos) * block_stride
    out_base_ptr = out + (start + bos) * block_stride

    d0_idx = tl.arange(0, x_D0_PAD)
    d1_idx = tl.arange(0, x_D1_PAD)
    p_idx = tl.arange(0, BLOCK_P)

    NEEDS_INNER_MASK: tl.constexpr = (x_D0_PAD != x_D0) or (x_D1_PAD != x_D1)

    neg_inf = -1e30
    m = tl.full((x_D0_PAD, x_D1_PAD), neg_inf, dtype=tl.float32)
    s = tl.zeros((x_D0_PAD, x_D1_PAD), dtype=tl.float32)

    for p in range(0, num_blocks_to_compute, BLOCK_P):
        kp = tl.minimum(BLOCK_P, num_blocks_to_compute - p)
        p_mask = p_idx < kp

        offs = (
            (p + p_idx)[:, None, None] * block_stride
            + d0_idx[None, :, None] * x_D1
            + d1_idx[None, None, :]
        ).to(tl.int32)

        if NEEDS_INNER_MASK:
            d0_valid = d0_idx < x_D0
            d1_valid = d1_idx < x_D1
            inner_valid = d0_valid[None, :, None] & d1_valid[None, None, :]
            mask = p_mask[:, None, None] & inner_valid
        else:
            mask = p_mask[:, None, None]
        slab = tl.load(x_base_ptr + offs, mask=mask, other=neg_inf).to(tl.float32)
        slab = slab * scale

        mc = tl.max(slab, axis=0)
        sc = tl.sum(tl.exp(slab - mc[None, :, :]), axis=0)

        m_new = tl.maximum(m, mc)
        s = s * tl.exp(m - m_new) + sc * tl.exp(mc - m_new)
        m = m_new

    for p in range(0, num_blocks_to_compute, BLOCK_P):
        kp = tl.minimum(BLOCK_P, num_blocks_to_compute - p)
        p_mask = p_idx < kp

        offs = (
            (p + p_idx)[:, None, None] * block_stride
            + d0_idx[None, :, None] * x_D1
            + d1_idx[None, None, :]
        ).to(tl.int32)

        if NEEDS_INNER_MASK:
            d0_valid = d0_idx < x_D0
            d1_valid = d1_idx < x_D1
            inner_valid = d0_valid[None, :, None] & d1_valid[None, None, :]
            mask = p_mask[:, None, None] & inner_valid
        else:
            mask = p_mask[:, None, None]
        slab = tl.load(x_base_ptr + offs, mask=mask, other=neg_inf).to(tl.float32)
        slab = slab * scale
        slab = tl.exp(slab - m[None, :, :]) / s[None, :, :]
        slab = slab.to(tl.bfloat16)
        tl.store(out_base_ptr + offs, slab, mask=mask)
