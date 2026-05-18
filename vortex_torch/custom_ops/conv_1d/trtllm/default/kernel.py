"""conv1d @triton.jit kernel — trtllm / block-table layout.

See the flashinfer leaf for the ``x_D0`` / ``x_D0_PAD`` split rationale
and the ``NEEDS_INNER_MASK`` constexpr fast-path.
"""
import triton
import triton.language as tl


@triton.jit
def conv1d_kernel(
    x,
    out,
    weight,
    seqlens,
    bos: tl.constexpr,
    eos: tl.constexpr,
    topk_val: tl.constexpr,
    K: tl.constexpr,
    x_D0: tl.constexpr,
    x_D1: tl.constexpr,
    x_D0_PAD: tl.constexpr,
    x_D1_PAD: tl.constexpr,
    block_size: tl.constexpr,
    max_blocks_per_seq: tl.constexpr,
    BLOCK_P: tl.constexpr = 128,
):
    pid = tl.program_id(0)

    _tokens = tl.load(seqlens + pid)
    num_blocks_this_seq = (_tokens + block_size - 1) // block_size
    start = pid * max_blocks_per_seq

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

    for p in range(0, num_blocks_to_compute, BLOCK_P):
        kp = tl.minimum(BLOCK_P, num_blocks_to_compute - p)
        p_mask = p_idx < kp

        acc = tl.zeros((BLOCK_P, x_D0_PAD, x_D1_PAD), dtype=tl.float32)

        for k in tl.static_range(K):
            in_pos = p + p_idx - k
            in_valid = (in_pos >= 0) & p_mask
            in_pos_clamped = tl.maximum(in_pos, 0)

            x_offs = (
                in_pos_clamped[:, None, None] * block_stride
                + d0_idx[None, :, None] * x_D1
                + d1_idx[None, None, :]
            ).to(tl.int32)

            if NEEDS_INNER_MASK:
                d0_valid = d0_idx < x_D0
                d1_valid = d1_idx < x_D1
                inner_valid = d0_valid[None, :, None] & d1_valid[None, None, :]
                x_mask = in_valid[:, None, None] & inner_valid
            else:
                x_mask = in_valid[:, None, None]
            x_val = tl.load(x_base_ptr + x_offs, mask=x_mask, other=0.0).to(tl.float32)

            w_offs = (
                k * block_stride
                + d0_idx[:, None] * x_D1
                + d1_idx[None, :]
            ).to(tl.int32)
            if NEEDS_INNER_MASK:
                w_d0_valid = d0_idx < x_D0
                w_d1_valid = d1_idx < x_D1
                w_mask = w_d0_valid[:, None] & w_d1_valid[None, :]
                w_val = tl.load(weight + w_offs, mask=w_mask, other=0.0).to(tl.float32)
            else:
                w_val = tl.load(weight + w_offs).to(tl.float32)

            acc = acc + x_val * w_val[None, :, :]

        out_offs = (
            (p + p_idx)[:, None, None] * block_stride
            + d0_idx[None, :, None] * x_D1
            + d1_idx[None, None, :]
        ).to(tl.int32)

        if NEEDS_INNER_MASK:
            d0_valid = d0_idx < x_D0
            d1_valid = d1_idx < x_D1
            inner_valid = d0_valid[None, :, None] & d1_valid[None, None, :]
            store_mask = p_mask[:, None, None] & inner_valid
        else:
            store_mask = p_mask[:, None, None]
        tl.store(out_base_ptr + out_offs, acc.to(tl.bfloat16), mask=store_mask)
