import torch
import triton
import triton.language as tl
from ..context import Context


@triton.jit
def conv1d_r_kernel(
    x,
    out,
    weight,
    indptr,
    bos: tl.constexpr,
    eos: tl.constexpr,
    topk_val: tl.constexpr,
    K: tl.constexpr,
    x_D0: tl.constexpr,
    x_D1: tl.constexpr,
    BLOCK_P: tl.constexpr = 128,
):
    pid = tl.program_id(0)

    start = tl.load(indptr + pid)
    end   = tl.load(indptr + pid + 1)
    num_pages_this_seq = end - start

    threshold: tl.constexpr = bos + eos + topk_val
    if num_pages_this_seq <= threshold:
        return

    num_pages_to_compute = num_pages_this_seq - bos - eos
    if num_pages_to_compute <= 0:
        return

    page_stride = x_D0 * x_D1
    x_base_ptr = x + (start + bos) * page_stride
    out_base_ptr = out + (start + bos) * page_stride

    d0_idx = tl.arange(0, x_D0)
    d1_idx = tl.arange(0, x_D1)
    p_idx  = tl.arange(0, BLOCK_P)

    for p in range(0, num_pages_to_compute, BLOCK_P):
        kp = tl.minimum(BLOCK_P, num_pages_to_compute - p)
        p_mask = p_idx < kp

        acc = tl.zeros((BLOCK_P, x_D0, x_D1), dtype=tl.float32)

        for k in tl.static_range(K):
            in_pos = p + p_idx - k
            in_valid = (in_pos >= 0) & p_mask
            in_pos_clamped = tl.maximum(in_pos, 0)

            x_offs = (
                in_pos_clamped[:, None, None] * page_stride
                + d0_idx[None, :, None] * x_D1
                + d1_idx[None, None, :]
            ).to(tl.int32)

            x_val = tl.load(
                x_base_ptr + x_offs,
                mask=in_valid[:, None, None],
                other=0.0,
            ).to(tl.float32)

            w_offs = (
                k * page_stride
                + d0_idx[:, None] * x_D1
                + d1_idx[None, :]
            ).to(tl.int32)
            w_val = tl.load(weight + w_offs).to(tl.float32)

            acc = acc + x_val * w_val[None, :, :]

        out_offs = (
            (p + p_idx)[:, None, None] * page_stride
            + d0_idx[None, :, None] * x_D1
            + d1_idx[None, None, :]
        ).to(tl.int32)

        tl.store(
            out_base_ptr + out_offs,
            acc.to(tl.bfloat16),
            mask=p_mask[:, None, None],
        )


def conv1d_r(
    x: torch.Tensor,
    out: torch.Tensor,
    weight: torch.Tensor,
    ctx: Context,
):
    eff_batch_size = ctx.batch_size * ctx.num_kv_heads

    conv1d_r_kernel[(eff_batch_size,)](
        x,
        out,
        weight,
        ctx.dense_kv_indptr,
        ctx.block_reserved_bos,
        ctx.block_reserved_eos,
        ctx.topk_val,
        weight.shape[0],
        x.shape[-2],
        x.shape[-1],
        num_warps=4,
        num_stages=1,
    )


def _conv1d_r(
    x: torch.Tensor,
    out: torch.Tensor,
    weight: torch.Tensor,
    indptr: torch.Tensor,
    block_reserved_bos: int,
    block_reserved_eos: int,
    topk_val: int,
    batch_size: int,
):
    conv1d_r_kernel[(batch_size,)](
        x,
        out,
        weight,
        indptr,
        block_reserved_bos,
        block_reserved_eos,
        topk_val,
        weight.shape[0],
        x.shape[-2],
        x.shape[-1],
        num_warps=4,
        num_stages=1,
    )
