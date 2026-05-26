"""Rebuild the custom CUDA kernel + correctness-check + benchmark vs best Triton.
    SGLANG_ENABLE_TORCH_COMPILE=0 CUDA_VISIBLE_DEVICES=<g> TORCH_CUDA_ARCH_LIST=10.0 \
        python cuda_mla/bench_cuda.py
"""
import statistics, sys, torch
from bench_triton_mla import make_inputs, KV_DIM, KV_LORA
import cuda_mla.build as cb
KERNELS = cb.register()
sm = 1.0 / (KV_DIM ** 0.5)


def mk(bs, H, blk, tok):
    nb = (tok + blk - 1) // blk; npg = bs * nb
    latent = torch.randn(npg * blk, KV_DIM, device='cuda', dtype=torch.bfloat16)
    bt = torch.randperm(npg, device='cuda', dtype=torch.int32).view(bs, nb).contiguous()
    sl = torch.full((bs,), tok, device='cuda', dtype=torch.int32)
    q = torch.randn(bs, H, KV_DIM, device='cuda', dtype=torch.bfloat16)
    return q, latent, bt, sl


def ref(q, latent, bt, sl, blk):
    bs, H, _ = q.shape; out = torch.empty(bs, H, KV_LORA, device='cuda', dtype=torch.float32)
    qf, lf = q.float(), latent.float()
    for b in range(bs):
        s = int(sl[b]); nb = (s + blk - 1) // blk
        rows = [torch.arange(int(bt[b, j]) * blk, int(bt[b, j]) * blk + blk, device='cuda') for j in range(nb)]
        slots = torch.cat(rows)[:s]; k = lf[slots]
        out[b] = torch.softmax((qf[b] @ k.t()) * sm, -1) @ k[:, :KV_LORA]
    return out


def gbps(name, bs, H, blk, tok, reps=3):
    fn = KERNELS[name]; vals = []
    for _ in range(reps):
        q, latent, bt, sl, o = make_inputs(bs, H, blk, tok, 'cuda', torch.bfloat16)
        for _ in range(15): fn(q, latent, bt, sl, sm, blk, KV_LORA, o)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(40): fn(q, latent, bt, sl, sm, blk, KV_LORA, o)
        e.record(); torch.cuda.synchronize()
        vals.append(bs * tok * KV_DIM * 2 / ((s.elapsed_time(e) / 40) * 1e-3) / 1e9)
    return statistics.median(vals)


if __name__ == "__main__":
    # correctness
    errs = []
    for H in (16, 20):
        for blk in (16, 32, 64):
            for tok in (2048, 2000):
                q, latent, bt, sl = mk(4, H, blk, tok)
                for spl in (1, 4):
                    o = cb.decode_blocktable_mla_cuda(q, latent, bt, sl, sm, blk, KV_LORA, None, splits=spl)
                    errs.append((o.float() - ref(q, latent, bt, sl, blk)).abs().max().item())
    print(f"correctness: max over all configs = {max(errs):.2e}  ({'OK' if max(errs)<3e-2 else 'FAIL'})")
    cands = ["cuda", "kv_split_opt", "single_pass_opt"]
    bss = [int(x) for x in (sys.argv[1].split(',') if len(sys.argv) > 1 else "8,32,64,128".split(','))]
    for H in (16, 20):
        print(f"\n=== H={H} sel=2048 (median-of-3 GB/s; *=best) ===")
        print(f"{'bs':>4}{'blk':>4} | " + "".join(f"{c:>15}" for c in cands) + "   best")
        for blk in (32, 64):
            for bs in bss:
                r = {c: gbps(c, bs, H, blk, 2048) for c in cands}; best = max(r, key=r.get)
                print(f"{bs:>4}{blk:>4} | " + "".join(f"{(('*' if c == best else ' ') + f'{r[c]:.0f}'):>15}" for c in cands) + f"   {best}")
