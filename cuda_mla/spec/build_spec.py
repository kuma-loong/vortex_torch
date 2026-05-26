"""Build + bench one specialized (bs,block) MLA kernel at a time (H=20 hardcoded).

    SGLANG_ENABLE_TORCH_COMPILE=0 PYTHONPATH=. CUDA_VISIBLE_DEVICES=<g> \
        TORCH_CUDA_ARCH_LIST=10.0 python cuda_mla/spec/build_spec.py <stem> <bs> <blk>
e.g. ... build_spec.py k_bs128_blk64 128 64
"""
import os, statistics, sys, torch
from torch.utils.cpp_extension import load

_HERE = os.path.dirname(os.path.abspath(__file__))
KV_DIM, KV_LORA, H = 576, 512, 20
sm = 1.0 / (KV_DIM ** 0.5)


def build(stem):
    bd = os.path.join(_HERE, "build_" + stem); os.makedirs(bd, exist_ok=True)
    return load(name="vortex_" + stem, sources=[os.path.join(_HERE, stem + ".cu")],
                extra_cuda_cflags=["-O3", "-arch=sm_100a", "--use_fast_math", "-lineinfo"],
                extra_include_paths=[_HERE], build_directory=bd, verbose=False)


def mk(bs, blk, tok):
    nb = (tok + blk - 1) // blk; npg = bs * nb
    latent = torch.randn(npg * blk, KV_DIM, device='cuda', dtype=torch.bfloat16)
    bt = torch.randperm(npg, device='cuda', dtype=torch.int32).view(bs, nb).contiguous()
    sl = torch.full((bs,), tok, device='cuda', dtype=torch.int32)
    q = torch.randn(bs, H, KV_DIM, device='cuda', dtype=torch.bfloat16)
    return q, latent, bt, sl


def ref(q, latent, bt, sl, blk):
    bs, h, _ = q.shape; out = torch.empty(bs, h, KV_LORA, device='cuda', dtype=torch.float32)
    qf, lf = q.float(), latent.float()
    for b in range(bs):
        s = int(sl[b]); nb = (s + blk - 1) // blk
        rows = [torch.arange(int(bt[b, j]) * blk, int(bt[b, j]) * blk + blk, device='cuda') for j in range(nb)]
        slots = torch.cat(rows)[:s]; k = lf[slots]
        out[b] = torch.softmax((qf[b] @ k.t()) * sm, -1) @ k[:, :KV_LORA]
    return out


def bench(mod, bs, blk, tok, reps=4):
    vals = []
    for _ in range(reps):
        q, latent, bt, sl = mk(bs, blk, tok)
        o = torch.empty(bs, H, KV_LORA, device='cuda', dtype=torch.bfloat16)
        f = lambda: mod.run(q, latent, bt, sl, o, sm)
        for _ in range(20): f()
        torch.cuda.synchronize(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(50): f()
        e.record(); torch.cuda.synchronize()
        vals.append(bs * tok * KV_DIM * 2 / ((s.elapsed_time(e) / 50) * 1e-3) / 1e9)
    return statistics.median(vals)


if __name__ == "__main__":
    stem = sys.argv[1] if len(sys.argv) > 1 else "k_bs128_blk64"
    bs = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    blk = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    mod = build(stem)
    # correctness
    errs = []
    for tok in (2048, 2000, 512):
        q, latent, bt, sl = mk(4, blk, tok)
        o = torch.empty(4, H, KV_LORA, device='cuda', dtype=torch.bfloat16)
        mod.run(q, latent, bt, sl, o, sm)
        errs.append((o.float() - ref(q, latent, bt, sl, blk)).abs().max().item())
    print(f"[{stem}] H={H} correctness max={max(errs):.2e} ({'OK' if max(errs)<3e-2 else 'FAIL'})")
    for tok in (1024, 2048, 4096):
        g = bench(mod, bs, blk, tok)
        print(f"  bs={bs} blk={blk} sel={tok}: {g:.0f} GB/s ({g/8000*100:.1f}% peak)")
