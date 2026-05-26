"""Diagnostic: isolate the cost of (load-mask, where-mask, loop-peel) in the
single-pass MLA kernel, in the benchmark regime (seqlen % BLOCK_N == 0, so the
fully-unmasked path is numerically correct). Run under CUDA_VISIBLE_DEVICES.
"""
import statistics
import torch
import triton
import triton.language as tl
from bench_triton_mla import make_inputs, KV_DIM, KV_LORA


@triton.jit
def _diag_kernel(
    Q, K_Buffer, V_Buffer, sm_scale, Seqlens, Block_Table, O,
    stride_qbs, stride_qh, stride_buf_kbs, stride_buf_vbs,
    stride_obs, stride_oh, stride_bt_b,
    q_head_num: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr, BLOCK_DPE: tl.constexpr, BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_H: tl.constexpr,
    USE_LOAD_MASK: tl.constexpr, USE_WHERE: tl.constexpr, PEEL: tl.constexpr,
    WITH_TAIL: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_head = cur_head_id * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < q_head_num

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
    seqlen = tl.load(Seqlens + cur_batch)

    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]
    q = tl.load(Q + offs_q, mask=mask_h[:, None], other=0.0)
    off_qpe = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :]
    qpe = tl.load(Q + off_qpe, mask=mask_h[:, None], other=0.0)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)
    bt_base = Block_Table + cur_batch * stride_bt_b

    if PEEL:
        loop_end = (seqlen // BLOCK_N) * BLOCK_N
    else:
        loop_end = seqlen

    for start_n in range(0, loop_end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        valid = offs_n < seqlen
        if USE_LOAD_MASK:
            page = tl.load(bt_base + (offs_n // BLOCK_SIZE), mask=valid, other=0)
        else:
            page = tl.load(bt_base + (offs_n // BLOCK_SIZE))
        kv_loc = page * BLOCK_SIZE + (offs_n % BLOCK_SIZE)
        if USE_LOAD_MASK:
            k = tl.load(K_Buffer + kv_loc[None, :] * stride_buf_kbs + offs_d[:, None],
                        mask=valid[None, :], other=0.0)
            kpe = tl.load(K_Buffer + kv_loc[None, :] * stride_buf_kbs + offs_dpe[:, None],
                          mask=valid[None, :], other=0.0)
            v = tl.load(V_Buffer + kv_loc[:, None] * stride_buf_vbs + offs_dv[None, :],
                        mask=valid[:, None], other=0.0)
        else:
            k = tl.load(K_Buffer + kv_loc[None, :] * stride_buf_kbs + offs_d[:, None])
            kpe = tl.load(K_Buffer + kv_loc[None, :] * stride_buf_kbs + offs_dpe[:, None])
            v = tl.load(V_Buffer + kv_loc[:, None] * stride_buf_vbs + offs_dv[None, :])
        qk = tl.dot(q, k.to(q.dtype))
        qk += tl.dot(qpe, kpe.to(qpe.dtype))
        qk *= sm_scale
        if USE_WHERE:
            qk = tl.where(valid[None, :], qk, float("-inf"))
        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        acc *= re_scale[:, None]
        acc += tl.dot(p.to(v.dtype), v)
        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max

    if WITH_TAIL:
        # separate duplicated masked tail block (mirrors _fwd_blocktable_mla_kernel_opt)
        if loop_end < seqlen:
            offs_n = loop_end + tl.arange(0, BLOCK_N)
            valid = offs_n < seqlen
            page = tl.load(bt_base + (offs_n // BLOCK_SIZE), mask=valid, other=0)
            kv_loc = page * BLOCK_SIZE + (offs_n % BLOCK_SIZE)
            k = tl.load(K_Buffer + kv_loc[None, :] * stride_buf_kbs + offs_d[:, None],
                        mask=valid[None, :], other=0.0)
            qk = tl.dot(q, k.to(q.dtype))
            kpe = tl.load(K_Buffer + kv_loc[None, :] * stride_buf_kbs + offs_dpe[:, None],
                          mask=valid[None, :], other=0.0)
            qk += tl.dot(qpe, kpe.to(qpe.dtype))
            qk *= sm_scale
            qk = tl.where(valid[None, :], qk, float("-inf"))
            v = tl.load(V_Buffer + kv_loc[:, None] * stride_buf_vbs + offs_dv[None, :],
                        mask=valid[:, None], other=0.0)
            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(v.dtype), v)
            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

    offs_o = cur_batch * stride_obs + cur_head[:, None] * stride_oh + offs_dv[None, :]
    tl.store(O + offs_o, acc / e_sum[:, None], mask=mask_h[:, None])


def run(q, latent, bt, sl, sm, blk, o, BLOCK_H, lm, wh, peel, tail=False, num_warps=8):
    bs, H, _ = q.shape
    grid = (bs, triton.cdiv(H, BLOCK_H))
    _diag_kernel[grid](
        q, latent, latent, sm, sl, bt, o,
        q.stride(0), q.stride(1), latent.stride(0), latent.stride(0),
        o.stride(0), o.stride(1), bt.stride(0),
        q_head_num=H, BLOCK_SIZE=blk, BLOCK_DMODEL=512, BLOCK_DPE=64,
        BLOCK_DV=512, BLOCK_N=blk, BLOCK_H=BLOCK_H,
        USE_LOAD_MASK=lm, USE_WHERE=wh, PEEL=peel, WITH_TAIL=tail, num_warps=num_warps,
    )
    return o


def bench(fn, q, latent, bt, sl, o, blk, iters=50, warmup=20):
    sm = 1.0 / (KV_DIM ** 0.5)
    for _ in range(warmup):
        fn(q, latent, bt, sl, sm, blk, o)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn(q, latent, bt, sl, sm, blk, o)
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def gbps(q, latent, bt, sl, o, blk, BLOCK_H, lm, wh, peel, tail=False):
    bs, H, _ = q.shape
    tok = int(sl[0])
    best = []
    for _ in range(3):
        ms = bench(lambda *a: run(*a, BLOCK_H, lm, wh, peel, tail), q, latent, bt, sl, o, blk)
        best.append(bs * tok * KV_DIM * 2 / (ms * 1e-3) / 1e9)
    return statistics.median(best)


if __name__ == "__main__":
    dev, dt = "cuda", torch.bfloat16
    bs = 128
    configs = [
        # name, load_mask, where, peel, with_tail
        ("baseline  single-loop lm+wh ", True,  True,  False, False),
        ("unmasked  single-loop       ", False, False, False, False),
        ("peel, NO tail block         ", False, False, True,  False),
        ("peel + SEPARATE tail block  ", False, False, True,  True),
    ]
    for H, BLOCK_H in [(16, 16), (20, 16), (20, 32)]:
        print(f"\n=== H={H} BLOCK_H={BLOCK_H} bs={bs} (median of 3 GB/s) ===")
        hdr = f"{'config':<30}" + "".join(f"{f'b{b}/t{t}':>11}" for b in (16, 64) for t in (1024, 4096))
        print(hdr)
        for name, lm, wh, peel, tail in configs:
            row = f"{name:<30}"
            for blk in (16, 64):
                for tok in (1024, 4096):
                    q, latent, bt, sl, o = make_inputs(bs, H, blk, tok, dev, dt)
                    g = gbps(q, latent, bt, sl, o, blk, BLOCK_H, lm, wh, peel, tail)
                    row += f"{g:>11.0f}"
            print(row)
