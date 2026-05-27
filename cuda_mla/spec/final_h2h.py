"""FINAL apples-to-apples on one empty GPU: the delivered kernel (k_h20_bs128_blk64
run() = bf16-O + MINB3 + sp3 + vectorized Q-load) vs Triton best."""
import os, statistics, torch
from torch.utils.cpp_extension import load
from vortex_torch.engine.sgl.attention_backend.triton_mla_kernel import (
    decode_blocktable_mla_opt, decode_blocktable_mla_split)
HERE = "cuda_mla/spec"; KV_DIM, KV_LORA, H = 576, 512, 20; sm = 1.0 / (KV_DIM ** 0.5)
bd = HERE + "/build_k_h20_bs128_blk64"; os.makedirs(bd, exist_ok=True)
mod = load(name="vortex_k_h20_bs128_blk64", sources=[HERE + "/k_h20_bs128_blk64.cu"],
           extra_cuda_cflags=["-O3", "-arch=sm_100a", "--use_fast_math", "-lineinfo"],
           extra_include_paths=[HERE], build_directory=bd, verbose=False)

def mk(bs, blk, tok, ragged=False):
    nb = (tok + blk - 1) // blk; npg = bs * nb
    latent = torch.randn(npg * blk, KV_DIM, device='cuda', dtype=torch.bfloat16)
    bt = torch.randperm(npg, device='cuda', dtype=torch.int32).view(bs, nb).contiguous()
    sl = (torch.randint(tok // 2, tok + 1, (bs,), device='cuda', dtype=torch.int32) if ragged
          else torch.full((bs,), tok, device='cuda', dtype=torch.int32))
    q = torch.randn(bs, H, KV_DIM, device='cuda', dtype=torch.bfloat16)
    return q, latent, bt, sl

def bench(call, bs, blk, tok, reps=12, ragged=False):
    vals = []
    for _ in range(reps):
        q, latent, bt, sl = mk(bs, blk, tok, ragged)
        o = torch.empty(bs, H, KV_LORA, device='cuda', dtype=torch.bfloat16)
        f = lambda: call(q, latent, bt, sl, o)
        for _ in range(20): f()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True); s.record()
        for _ in range(50): f()
        e.record(); torch.cuda.synchronize()
        vals.append(bs * tok * KV_DIM * 2 / ((s.elapsed_time(e) / 50) * 1e-3) / 1e9)
    return statistics.median(vals)

mine = lambda q,l,bt,sl,o: mod.run(q,l,bt,sl,o,sm)
trsp = lambda q,l,bt,sl,o: decode_blocktable_mla_opt(q,l,bt,sl,sm,64,KV_LORA,o)
trks = lambda q,l,bt,sl,o: decode_blocktable_mla_split(q,l,bt,sl,sm,64,KV_LORA,o)

print("=== FINAL h20 bs=128 blk=64 (one empty GPU, GB/s) ===")
print(f"{'sel':>8} {'mine':>7} {'tri_sp':>7} {'tri_ks':>7}  mine/best")
for tok in (1024, 2048, 3072, 4096):
    m = bench(mine, 128, 64, tok); a = bench(trsp, 128, 64, tok); k = bench(trks, 128, 64, tok)
    print(f"{tok:>8} {m:>7.0f} {a:>7.0f} {k:>7.0f}  {m/max(a,k):.3f}")
m = bench(mine, 128, 64, 2048, ragged=True); a = bench(trsp, 128, 64, 2048, ragged=True)
print(f"{'ragged':>8} {m:>7.0f} {a:>7.0f} {'':>7}  {m/a:.3f}")
