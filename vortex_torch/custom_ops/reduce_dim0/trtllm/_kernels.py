"""Schedule.S reduce-dim0 @triton.jit kernels — trtllm / block-table layout.

One specialized kernel per reduce_type (sum / mean / max / min / l2norm).
Per-row segment is derived from ``seqlens`` + ``block_size`` +
``max_blocks_per_seq`` (in trtllm the RAGGED storage is allocated as a
flat ``[eff_bs, max_blocks_per_seq]`` block-table view; the row's first
block lives at ``pid * max_blocks_per_seq``). The trailing ``bos`` /
``eos`` blocks are trimmed (matching the softmax convention).

The store dtype is taken from ``out.dtype.element_ty`` at compile time,
so the same kernel handles bf16 / fp16 / fp32 outputs. fp8 outputs are
not supported here; add an fp8 bucket if needed.
"""
import triton
import triton.language as tl


@triton.jit
def reduce_dim0_sum_kernel(
    x,
    out,
    seqlens,
    bos: tl.constexpr,
    eos: tl.constexpr,
    x_D0: tl.constexpr,
    x_D1: tl.constexpr,
    block_size: tl.constexpr,
    max_blocks_per_seq: tl.constexpr,
    BLOCK_P: tl.constexpr = 512,
):
    pid = tl.program_id(0)
    _tokens = tl.load(seqlens + pid)
    num_blocks_this_seq = (_tokens + block_size - 1) // block_size
    _row_start = pid * max_blocks_per_seq
    start = _row_start + bos
    end = _row_start + num_blocks_this_seq - eos

    block_stride = x_D0 * x_D1
    d0_idx = tl.arange(0, x_D0)
    d1_idx = tl.arange(0, x_D1)
    out_ptr = out + pid * block_stride + d0_idx[:, None] * x_D1 + d1_idx[None, :]

    if end <= start:
        zero = tl.zeros((x_D0, x_D1), dtype=tl.float32)
        tl.store(out_ptr, zero.to(out.dtype.element_ty))
        return

    num_blocks = end - start
    acc = tl.zeros((x_D0, x_D1), dtype=tl.float32)
    p_idx = tl.arange(0, BLOCK_P)
    base_ptr = x + start * block_stride
    for p in range(0, num_blocks, BLOCK_P):
        kp = tl.minimum(BLOCK_P, num_blocks - p)
        p_mask = p_idx < kp
        offs = (
            (p + p_idx)[:, None, None] * block_stride
            + d0_idx[None, :, None] * x_D1
            + d1_idx[None, None, :]
        ).to(tl.int32)
        slab = tl.load(base_ptr + offs, mask=p_mask[:, None, None], other=0.0).to(tl.float32)
        acc += tl.sum(slab, axis=0)
    tl.store(out_ptr, acc.to(out.dtype.element_ty))


@triton.jit
def reduce_dim0_mean_kernel(
    x,
    out,
    seqlens,
    bos: tl.constexpr,
    eos: tl.constexpr,
    x_D0: tl.constexpr,
    x_D1: tl.constexpr,
    block_size: tl.constexpr,
    max_blocks_per_seq: tl.constexpr,
    BLOCK_P: tl.constexpr = 512,
):
    pid = tl.program_id(0)
    _tokens = tl.load(seqlens + pid)
    num_blocks_this_seq = (_tokens + block_size - 1) // block_size
    _row_start = pid * max_blocks_per_seq
    start = _row_start + bos
    end = _row_start + num_blocks_this_seq - eos

    block_stride = x_D0 * x_D1
    d0_idx = tl.arange(0, x_D0)
    d1_idx = tl.arange(0, x_D1)
    out_ptr = out + pid * block_stride + d0_idx[:, None] * x_D1 + d1_idx[None, :]

    if end <= start:
        zero = tl.zeros((x_D0, x_D1), dtype=tl.float32)
        tl.store(out_ptr, zero.to(out.dtype.element_ty))
        return

    num_blocks = end - start
    acc = tl.zeros((x_D0, x_D1), dtype=tl.float32)
    p_idx = tl.arange(0, BLOCK_P)
    base_ptr = x + start * block_stride
    for p in range(0, num_blocks, BLOCK_P):
        kp = tl.minimum(BLOCK_P, num_blocks - p)
        p_mask = p_idx < kp
        offs = (
            (p + p_idx)[:, None, None] * block_stride
            + d0_idx[None, :, None] * x_D1
            + d1_idx[None, None, :]
        ).to(tl.int32)
        slab = tl.load(base_ptr + offs, mask=p_mask[:, None, None], other=0.0).to(tl.float32)
        acc += tl.sum(slab, axis=0)
    acc = acc / num_blocks.to(tl.float32)
    tl.store(out_ptr, acc.to(out.dtype.element_ty))


@triton.jit
def reduce_dim0_max_kernel(
    x,
    out,
    seqlens,
    bos: tl.constexpr,
    eos: tl.constexpr,
    x_D0: tl.constexpr,
    x_D1: tl.constexpr,
    block_size: tl.constexpr,
    max_blocks_per_seq: tl.constexpr,
    BLOCK_P: tl.constexpr = 512,
):
    pid = tl.program_id(0)
    _tokens = tl.load(seqlens + pid)
    num_blocks_this_seq = (_tokens + block_size - 1) // block_size
    _row_start = pid * max_blocks_per_seq
    start = _row_start + bos
    end = _row_start + num_blocks_this_seq - eos

    block_stride = x_D0 * x_D1
    d0_idx = tl.arange(0, x_D0)
    d1_idx = tl.arange(0, x_D1)
    out_ptr = out + pid * block_stride + d0_idx[:, None] * x_D1 + d1_idx[None, :]

    if end <= start:
        zero = tl.zeros((x_D0, x_D1), dtype=tl.float32)
        tl.store(out_ptr, zero.to(out.dtype.element_ty))
        return

    num_blocks = end - start
    acc = tl.full((x_D0, x_D1), -1e30, dtype=tl.float32)
    p_idx = tl.arange(0, BLOCK_P)
    base_ptr = x + start * block_stride
    for p in range(0, num_blocks, BLOCK_P):
        kp = tl.minimum(BLOCK_P, num_blocks - p)
        p_mask = p_idx < kp
        offs = (
            (p + p_idx)[:, None, None] * block_stride
            + d0_idx[None, :, None] * x_D1
            + d1_idx[None, None, :]
        ).to(tl.int32)
        slab = tl.load(base_ptr + offs, mask=p_mask[:, None, None], other=0.0).to(tl.float32)
        slab = tl.where(p_mask[:, None, None], slab, -1e30)
        acc = tl.maximum(acc, tl.max(slab, axis=0))
    tl.store(out_ptr, acc.to(out.dtype.element_ty))


@triton.jit
def reduce_dim0_min_kernel(
    x,
    out,
    seqlens,
    bos: tl.constexpr,
    eos: tl.constexpr,
    x_D0: tl.constexpr,
    x_D1: tl.constexpr,
    block_size: tl.constexpr,
    max_blocks_per_seq: tl.constexpr,
    BLOCK_P: tl.constexpr = 512,
):
    pid = tl.program_id(0)
    _tokens = tl.load(seqlens + pid)
    num_blocks_this_seq = (_tokens + block_size - 1) // block_size
    _row_start = pid * max_blocks_per_seq
    start = _row_start + bos
    end = _row_start + num_blocks_this_seq - eos

    block_stride = x_D0 * x_D1
    d0_idx = tl.arange(0, x_D0)
    d1_idx = tl.arange(0, x_D1)
    out_ptr = out + pid * block_stride + d0_idx[:, None] * x_D1 + d1_idx[None, :]

    if end <= start:
        zero = tl.zeros((x_D0, x_D1), dtype=tl.float32)
        tl.store(out_ptr, zero.to(out.dtype.element_ty))
        return

    num_blocks = end - start
    acc = tl.full((x_D0, x_D1), 1e30, dtype=tl.float32)
    p_idx = tl.arange(0, BLOCK_P)
    base_ptr = x + start * block_stride
    for p in range(0, num_blocks, BLOCK_P):
        kp = tl.minimum(BLOCK_P, num_blocks - p)
        p_mask = p_idx < kp
        offs = (
            (p + p_idx)[:, None, None] * block_stride
            + d0_idx[None, :, None] * x_D1
            + d1_idx[None, None, :]
        ).to(tl.int32)
        slab = tl.load(base_ptr + offs, mask=p_mask[:, None, None], other=0.0).to(tl.float32)
        slab = tl.where(p_mask[:, None, None], slab, 1e30)
        acc = tl.minimum(acc, tl.min(slab, axis=0))
    tl.store(out_ptr, acc.to(out.dtype.element_ty))


@triton.jit
def reduce_dim0_l2norm_kernel(
    x,
    out,
    seqlens,
    bos: tl.constexpr,
    eos: tl.constexpr,
    x_D0: tl.constexpr,
    x_D1: tl.constexpr,
    block_size: tl.constexpr,
    max_blocks_per_seq: tl.constexpr,
    BLOCK_P: tl.constexpr = 512,
):
    pid = tl.program_id(0)
    _tokens = tl.load(seqlens + pid)
    num_blocks_this_seq = (_tokens + block_size - 1) // block_size
    _row_start = pid * max_blocks_per_seq
    start = _row_start + bos
    end = _row_start + num_blocks_this_seq - eos

    block_stride = x_D0 * x_D1
    d0_idx = tl.arange(0, x_D0)
    d1_idx = tl.arange(0, x_D1)
    out_ptr = out + pid * block_stride + d0_idx[:, None] * x_D1 + d1_idx[None, :]

    if end <= start:
        zero = tl.zeros((x_D0, x_D1), dtype=tl.float32)
        tl.store(out_ptr, zero.to(out.dtype.element_ty))
        return

    num_blocks = end - start
    acc = tl.zeros((x_D0, x_D1), dtype=tl.float32)
    p_idx = tl.arange(0, BLOCK_P)
    base_ptr = x + start * block_stride
    for p in range(0, num_blocks, BLOCK_P):
        kp = tl.minimum(BLOCK_P, num_blocks - p)
        p_mask = p_idx < kp
        offs = (
            (p + p_idx)[:, None, None] * block_stride
            + d0_idx[None, :, None] * x_D1
            + d1_idx[None, None, :]
        ).to(tl.int32)
        slab = tl.load(base_ptr + offs, mask=p_mask[:, None, None], other=0.0).to(tl.float32)
        acc += tl.sum(slab * slab, axis=0)
    acc = tl.sqrt(acc)
    tl.store(out_ptr, acc.to(out.dtype.element_ty))
