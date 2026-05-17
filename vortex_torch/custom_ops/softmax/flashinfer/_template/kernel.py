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
    BLOCK_P: tl.constexpr = 512,
):
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

    d0_idx = tl.arange(0, x_D0)
    d1_idx = tl.arange(0, x_D1)
    p_idx = tl.arange(0, BLOCK_P)

    neg_inf = -1e30
    m = tl.full((x_D0, x_D1), neg_inf, dtype=tl.float32)
    s = tl.zeros((x_D0, x_D1), dtype=tl.float32)

    for p in range(0, num_blocks_to_compute, BLOCK_P):
        kp = tl.minimum(BLOCK_P, num_blocks_to_compute - p)
        p_mask = p_idx < kp

        offs = (
            (p + p_idx)[:, None, None] * block_stride
            + d0_idx[None, :, None] * x_D1
            + d1_idx[None, None, :]
        ).to(tl.int32)

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

        mask = p_mask[:, None, None]
        slab = tl.load(x_base_ptr + offs, mask=mask, other=neg_inf).to(tl.float32)
        slab = slab * scale
        slab = tl.exp(slab - m[None, :, :]) / s[None, :, :]
        slab = slab.to(tl.bfloat16)
        tl.store(out_base_ptr + offs, slab, mask=mask)
