"""normalize @triton.jit kernel — flashinfer / CSR layout.

Two-pass L2 normalize over the middle ``[bos, block_len-eos)`` slots of
the per-(req, kv_head) RAGGED slice ``x[indptr[pid] : indptr[pid+1])``
along the block axis. BOS / EOS reserved blocks are skipped.
"""
import triton
import triton.language as tl


@triton.jit
def normalize_kernel(
    x,
    out,
    indptr,
    bos: tl.constexpr,
    eos: tl.constexpr,
    topk_val: tl.constexpr,
    x_D0: tl.constexpr,
    x_D1: tl.constexpr,
    BLOCK_P: tl.constexpr = 512,
    eps: tl.constexpr = 1e-12,
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

    square_norm = tl.zeros((x_D0, x_D1), dtype=tl.float32)
    eps_mat = tl.full((x_D0, x_D1), value=eps, dtype=tl.float32)

    for p in range(0, num_blocks_to_compute, BLOCK_P):
        kp = tl.minimum(BLOCK_P, num_blocks_to_compute - p)
        p_mask = p_idx < kp

        offs = (
            (p + p_idx)[:, None, None] * block_stride
            + d0_idx[None, :, None] * x_D1
            + d1_idx[None, None, :]
        ).to(tl.int32)

        mask = p_mask[:, None, None]
        slab = tl.load(x_base_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        square_norm_c = tl.sum(slab * slab, axis=0)
        square_norm = square_norm + square_norm_c

    norm = tl.maximum(tl.sqrt(square_norm), eps_mat)

    for p in range(0, num_blocks_to_compute, BLOCK_P):
        kp = tl.minimum(BLOCK_P, num_blocks_to_compute - p)
        p_mask = p_idx < kp

        offs = (
            (p + p_idx)[:, None, None] * block_stride
            + d0_idx[None, :, None] * x_D1
            + d1_idx[None, None, :]
        ).to(tl.int32)

        mask = p_mask[:, None, None]
        slab = tl.load(x_base_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        slab = slab / norm[None, :, :]
        slab = slab.to(tl.bfloat16)
        tl.store(out_base_ptr + offs, slab, mask=mask)
