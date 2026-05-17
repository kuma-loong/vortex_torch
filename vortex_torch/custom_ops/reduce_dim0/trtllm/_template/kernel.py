"""reduce_dim0 / trtllm / _template — standalone kernel.py.

Copied from sibling ``mean`` variant under ``../_kernels.py`` and
renamed. Specialise this kernel for the constraints you declare in
``meta.json`` (e.g. exact ``D0``/``D1``, fixed ``block_size``,
fused with a downstream op, etc.).
"""
import triton
import triton.language as tl


@triton.jit
def reduce_dim0_template_kernel(
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

