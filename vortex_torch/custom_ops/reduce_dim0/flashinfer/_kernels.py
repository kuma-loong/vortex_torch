"""Schedule.S reduce-dim0 @triton.jit kernels — flashinfer / CSR layout.

One specialized kernel per reduce_type (sum / mean / max / min / l2norm).
Each kernel is launched with grid ``(eff_batch_size,)``; one program per
(batch, kv_head) row. The row's segment in the RAGGED storage buffer is
``[start, start + num_blocks_this_seq)`` derived from
``dense_kv_indptr``. The trailing ``bos`` / ``eos`` blocks are trimmed
(matching the softmax convention).

The store dtype is taken from ``out.dtype.element_ty`` at compile time,
so the same kernel handles bf16 / fp16 / fp32 outputs. fp8 outputs are
not supported here (they'd need range-clamping); add an fp8 bucket
under ``custom_ops/reduce_dim0/flashinfer/fp8_<rt>/`` if needed.

``x_D0`` / ``x_D1`` are the **real** inner sizes used for memory strides
on the row-major RAGGED input and BATCHED output storage; ``x_D0_PAD``
/ ``x_D1_PAD`` are their next-pow2 round-ups used for ``tl.arange`` /
tile-shape constexprs. A ``NEEDS_INNER_MASK: tl.constexpr`` branch (set
when ``*_PAD != *``) gates whether load/store masks include the
inner-dim validity mask. Padded lanes load the reduction *identity*
(``0.0`` for sum/mean/l2norm, ``-inf`` for max, ``+inf`` for min) so
they don't contaminate the result; the store mask suppresses writes to
the padded lanes. When ``shape == padded_shape`` Triton specializes
the unmasked branch and emits no inner-dim mask arithmetic at all.
"""
import triton
import triton.language as tl


@triton.jit
def reduce_dim0_sum_kernel(
    x,
    out,
    indptr,
    bos: tl.constexpr,
    eos: tl.constexpr,
    x_D0: tl.constexpr,
    x_D1: tl.constexpr,
    x_D0_PAD: tl.constexpr,
    x_D1_PAD: tl.constexpr,
    BLOCK_P: tl.constexpr = 512,
):
    pid = tl.program_id(0)
    _row_start = tl.load(indptr + pid)
    _end = tl.load(indptr + pid + 1)
    num_blocks_this_seq = _end - _row_start
    start = _row_start + bos
    end = _row_start + num_blocks_this_seq - eos

    block_stride = x_D0 * x_D1
    d0_idx = tl.arange(0, x_D0_PAD)
    d1_idx = tl.arange(0, x_D1_PAD)
    out_ptr = out + pid * block_stride + d0_idx[:, None] * x_D1 + d1_idx[None, :]

    NEEDS_INNER_MASK: tl.constexpr = (x_D0_PAD != x_D0) or (x_D1_PAD != x_D1)
    if NEEDS_INNER_MASK:
        store_d0_valid = d0_idx < x_D0
        store_d1_valid = d1_idx < x_D1
        store_mask = store_d0_valid[:, None] & store_d1_valid[None, :]
    else:
        store_mask = None

    if end <= start:
        zero = tl.zeros((x_D0_PAD, x_D1_PAD), dtype=tl.float32)
        if NEEDS_INNER_MASK:
            tl.store(out_ptr, zero.to(out.dtype.element_ty), mask=store_mask)
        else:
            tl.store(out_ptr, zero.to(out.dtype.element_ty))
        return

    num_blocks = end - start
    acc = tl.zeros((x_D0_PAD, x_D1_PAD), dtype=tl.float32)
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
        if NEEDS_INNER_MASK:
            d0_valid = d0_idx < x_D0
            d1_valid = d1_idx < x_D1
            inner_valid = d0_valid[None, :, None] & d1_valid[None, None, :]
            mask = p_mask[:, None, None] & inner_valid
        else:
            mask = p_mask[:, None, None]
        slab = tl.load(base_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(slab, axis=0)
    if NEEDS_INNER_MASK:
        tl.store(out_ptr, acc.to(out.dtype.element_ty), mask=store_mask)
    else:
        tl.store(out_ptr, acc.to(out.dtype.element_ty))


@triton.jit
def reduce_dim0_mean_kernel(
    x,
    out,
    indptr,
    bos: tl.constexpr,
    eos: tl.constexpr,
    x_D0: tl.constexpr,
    x_D1: tl.constexpr,
    x_D0_PAD: tl.constexpr,
    x_D1_PAD: tl.constexpr,
    BLOCK_P: tl.constexpr = 512,
):
    pid = tl.program_id(0)
    _row_start = tl.load(indptr + pid)
    _end = tl.load(indptr + pid + 1)
    num_blocks_this_seq = _end - _row_start
    start = _row_start + bos
    end = _row_start + num_blocks_this_seq - eos

    block_stride = x_D0 * x_D1
    d0_idx = tl.arange(0, x_D0_PAD)
    d1_idx = tl.arange(0, x_D1_PAD)
    out_ptr = out + pid * block_stride + d0_idx[:, None] * x_D1 + d1_idx[None, :]

    NEEDS_INNER_MASK: tl.constexpr = (x_D0_PAD != x_D0) or (x_D1_PAD != x_D1)
    if NEEDS_INNER_MASK:
        store_d0_valid = d0_idx < x_D0
        store_d1_valid = d1_idx < x_D1
        store_mask = store_d0_valid[:, None] & store_d1_valid[None, :]
    else:
        store_mask = None

    if end <= start:
        zero = tl.zeros((x_D0_PAD, x_D1_PAD), dtype=tl.float32)
        if NEEDS_INNER_MASK:
            tl.store(out_ptr, zero.to(out.dtype.element_ty), mask=store_mask)
        else:
            tl.store(out_ptr, zero.to(out.dtype.element_ty))
        return

    num_blocks = end - start
    acc = tl.zeros((x_D0_PAD, x_D1_PAD), dtype=tl.float32)
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
        if NEEDS_INNER_MASK:
            d0_valid = d0_idx < x_D0
            d1_valid = d1_idx < x_D1
            inner_valid = d0_valid[None, :, None] & d1_valid[None, None, :]
            mask = p_mask[:, None, None] & inner_valid
        else:
            mask = p_mask[:, None, None]
        slab = tl.load(base_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(slab, axis=0)
    acc = acc / num_blocks.to(tl.float32)
    if NEEDS_INNER_MASK:
        tl.store(out_ptr, acc.to(out.dtype.element_ty), mask=store_mask)
    else:
        tl.store(out_ptr, acc.to(out.dtype.element_ty))


@triton.jit
def reduce_dim0_max_kernel(
    x,
    out,
    indptr,
    bos: tl.constexpr,
    eos: tl.constexpr,
    x_D0: tl.constexpr,
    x_D1: tl.constexpr,
    x_D0_PAD: tl.constexpr,
    x_D1_PAD: tl.constexpr,
    BLOCK_P: tl.constexpr = 512,
):
    pid = tl.program_id(0)
    _row_start = tl.load(indptr + pid)
    _end = tl.load(indptr + pid + 1)
    num_blocks_this_seq = _end - _row_start
    start = _row_start + bos
    end = _row_start + num_blocks_this_seq - eos

    block_stride = x_D0 * x_D1
    d0_idx = tl.arange(0, x_D0_PAD)
    d1_idx = tl.arange(0, x_D1_PAD)
    out_ptr = out + pid * block_stride + d0_idx[:, None] * x_D1 + d1_idx[None, :]

    NEEDS_INNER_MASK: tl.constexpr = (x_D0_PAD != x_D0) or (x_D1_PAD != x_D1)
    if NEEDS_INNER_MASK:
        store_d0_valid = d0_idx < x_D0
        store_d1_valid = d1_idx < x_D1
        store_mask = store_d0_valid[:, None] & store_d1_valid[None, :]
    else:
        store_mask = None

    if end <= start:
        zero = tl.zeros((x_D0_PAD, x_D1_PAD), dtype=tl.float32)
        if NEEDS_INNER_MASK:
            tl.store(out_ptr, zero.to(out.dtype.element_ty), mask=store_mask)
        else:
            tl.store(out_ptr, zero.to(out.dtype.element_ty))
        return

    num_blocks = end - start
    acc = tl.full((x_D0_PAD, x_D1_PAD), -1e30, dtype=tl.float32)
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
        if NEEDS_INNER_MASK:
            d0_valid = d0_idx < x_D0
            d1_valid = d1_idx < x_D1
            inner_valid = d0_valid[None, :, None] & d1_valid[None, None, :]
            mask = p_mask[:, None, None] & inner_valid
        else:
            mask = p_mask[:, None, None]
        # ``-inf`` for both p-tail (kp..BLOCK_P) and padded inner lanes.
        slab = tl.load(base_ptr + offs, mask=mask, other=-1e30).to(tl.float32)
        acc = tl.maximum(acc, tl.max(slab, axis=0))
    if NEEDS_INNER_MASK:
        tl.store(out_ptr, acc.to(out.dtype.element_ty), mask=store_mask)
    else:
        tl.store(out_ptr, acc.to(out.dtype.element_ty))


@triton.jit
def reduce_dim0_min_kernel(
    x,
    out,
    indptr,
    bos: tl.constexpr,
    eos: tl.constexpr,
    x_D0: tl.constexpr,
    x_D1: tl.constexpr,
    x_D0_PAD: tl.constexpr,
    x_D1_PAD: tl.constexpr,
    BLOCK_P: tl.constexpr = 512,
):
    pid = tl.program_id(0)
    _row_start = tl.load(indptr + pid)
    _end = tl.load(indptr + pid + 1)
    num_blocks_this_seq = _end - _row_start
    start = _row_start + bos
    end = _row_start + num_blocks_this_seq - eos

    block_stride = x_D0 * x_D1
    d0_idx = tl.arange(0, x_D0_PAD)
    d1_idx = tl.arange(0, x_D1_PAD)
    out_ptr = out + pid * block_stride + d0_idx[:, None] * x_D1 + d1_idx[None, :]

    NEEDS_INNER_MASK: tl.constexpr = (x_D0_PAD != x_D0) or (x_D1_PAD != x_D1)
    if NEEDS_INNER_MASK:
        store_d0_valid = d0_idx < x_D0
        store_d1_valid = d1_idx < x_D1
        store_mask = store_d0_valid[:, None] & store_d1_valid[None, :]
    else:
        store_mask = None

    if end <= start:
        zero = tl.zeros((x_D0_PAD, x_D1_PAD), dtype=tl.float32)
        if NEEDS_INNER_MASK:
            tl.store(out_ptr, zero.to(out.dtype.element_ty), mask=store_mask)
        else:
            tl.store(out_ptr, zero.to(out.dtype.element_ty))
        return

    num_blocks = end - start
    acc = tl.full((x_D0_PAD, x_D1_PAD), 1e30, dtype=tl.float32)
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
        if NEEDS_INNER_MASK:
            d0_valid = d0_idx < x_D0
            d1_valid = d1_idx < x_D1
            inner_valid = d0_valid[None, :, None] & d1_valid[None, None, :]
            mask = p_mask[:, None, None] & inner_valid
        else:
            mask = p_mask[:, None, None]
        # ``+inf`` for both p-tail and padded inner lanes.
        slab = tl.load(base_ptr + offs, mask=mask, other=1e30).to(tl.float32)
        acc = tl.minimum(acc, tl.min(slab, axis=0))
    if NEEDS_INNER_MASK:
        tl.store(out_ptr, acc.to(out.dtype.element_ty), mask=store_mask)
    else:
        tl.store(out_ptr, acc.to(out.dtype.element_ty))


@triton.jit
def reduce_dim0_l2norm_kernel(
    x,
    out,
    indptr,
    bos: tl.constexpr,
    eos: tl.constexpr,
    x_D0: tl.constexpr,
    x_D1: tl.constexpr,
    x_D0_PAD: tl.constexpr,
    x_D1_PAD: tl.constexpr,
    BLOCK_P: tl.constexpr = 512,
):
    pid = tl.program_id(0)
    _row_start = tl.load(indptr + pid)
    _end = tl.load(indptr + pid + 1)
    num_blocks_this_seq = _end - _row_start
    start = _row_start + bos
    end = _row_start + num_blocks_this_seq - eos

    block_stride = x_D0 * x_D1
    d0_idx = tl.arange(0, x_D0_PAD)
    d1_idx = tl.arange(0, x_D1_PAD)
    out_ptr = out + pid * block_stride + d0_idx[:, None] * x_D1 + d1_idx[None, :]

    NEEDS_INNER_MASK: tl.constexpr = (x_D0_PAD != x_D0) or (x_D1_PAD != x_D1)
    if NEEDS_INNER_MASK:
        store_d0_valid = d0_idx < x_D0
        store_d1_valid = d1_idx < x_D1
        store_mask = store_d0_valid[:, None] & store_d1_valid[None, :]
    else:
        store_mask = None

    if end <= start:
        zero = tl.zeros((x_D0_PAD, x_D1_PAD), dtype=tl.float32)
        if NEEDS_INNER_MASK:
            tl.store(out_ptr, zero.to(out.dtype.element_ty), mask=store_mask)
        else:
            tl.store(out_ptr, zero.to(out.dtype.element_ty))
        return

    num_blocks = end - start
    acc = tl.zeros((x_D0_PAD, x_D1_PAD), dtype=tl.float32)
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
        if NEEDS_INNER_MASK:
            d0_valid = d0_idx < x_D0
            d1_valid = d1_idx < x_D1
            inner_valid = d0_valid[None, :, None] & d1_valid[None, None, :]
            mask = p_mask[:, None, None] & inner_valid
        else:
            mask = p_mask[:, None, None]
        slab = tl.load(base_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(slab * slab, axis=0)
    acc = tl.sqrt(acc)
    if NEEDS_INNER_MASK:
        tl.store(out_ptr, acc.to(out.dtype.element_ty), mask=store_mask)
    else:
        tl.store(out_ptr, acc.to(out.dtype.element_ty))
