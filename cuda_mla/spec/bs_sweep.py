"""Validate MLADecoder is general across bs and block_size (vs Triton).
Correctness + throughput on uniform and ragged, bs in {8,32,64,128,256}, blk in {32,64}."""
import os, statistics, random, torch
from torch.utils.cpp_extension import load
from vortex_torch.engine.sgl.attention_backend.triton_mla_kernel import (
    decode_blocktable_mla_opt, decode_blocktable_mla_split)
HERE = "cuda_mla/spec"; KV_DIM, KV_LORA, H = 576, 512, 20; sm = 1.0 / (KV_DIM ** 0.5)
bd = HERE + "/build_decoder"; os.makedirs(bd, exist_ok=True)
mod = load(name="vortex_mla_decoder", sources=[HERE + "/mla_decoder.cu"],
           extra_cuda_cflags=["-O3", "-arch=sm_100a", "--use_fast_math", "-lineinfo"],
           extra_include_paths=[HERE], build_directory=bd, verbose=False)

def mk(bs, blk, sls):
    maxtok = int(max(sls)); nb = (maxtok + blk - 1) // blk; npg = bs * nb
    latent = torch.randn(npg * blk, KV_DIM, device='cuda', dtype=torch.bfloat16)
    bt = torch.randperm(npg, device='cuda', dtype=torch.int32).view(bs, nb).contiguous()
    sl = torch.tensor(sls, device='cuda', dtype=torch.int32)
    q = torch.randn(bs, H, KV_DIM, device='cuda', dtype=torch.bfloat16)
    return q, latent, bt, sl, nb

def ref(q, latent, bt, sl, blk):
    bs = q.size(0); out = torch.empty(bs, H, KV_LORA, device='cuda', dtype=torch.float32)
    qf, lf = q.float(), latent.float()
    for b in range(bs):
        s = int(sl[b]); nb = (s + blk - 1) // blk
        rows = [torch.arange(int(bt[b, j]) * blk, int(bt[b, j]) * blk + blk, device='cuda') for j in range(nb)]
        slots = torch.cat(rows)[:s]; k = lf[slots]
        out[b] = torch.softmax((qf[b] @ k.t()) * sm, -1) @ k[:, :KV_LORA]
    return out

def bench(call, q, latent, bt, sl, reps=8):
    o = torch.empty(q.size(0), H, KV_LORA, device='cuda', dtype=torch.bfloat16); vals = []
    for _ in range(reps):
        for _ in range(15): call(q, latent, bt, sl, o)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True); s.record()
        for _ in range(40): call(q, latent, bt, sl, o)
        e.record(); torch.cuda.synchronize()
        vals.append(int(sl.sum()) * KV_DIM * 2 / ((s.elapsed_time(e) / 40) * 1e-3) / 1e9)
    return statistics.median(vals)

random.seed(0)
for blk in (32, 64):
    print(f"\n========== block_size={blk}  (GB/s; mine/triton_best) ==========")
    print(f"{'bs':>5} {'pattern':>10} | {'splits':>6} {'mine':>6} {'tri_sp':>6} {'tri_ks':>6} | ratio  err")
    for bs in (8, 32, 64, 128, 256):
        for pat in ("uniform", "ragged"):
            if pat == "uniform": sls = [2048] * bs
            else: sls = [random.choice([256, 512, 1024, 2048, 4096]) for _ in range(bs)]
            nb = (max(sls) + blk - 1) // blk
            q, latent, bt, sl, _ = mk(bs, blk, sls)
            dec = mod.MLADecoder(bs, H, blk, nb)
            o = torch.empty(bs, H, KV_LORA, device='cuda', dtype=torch.bfloat16)
            dec.plan(sl); dec.run(q, latent, bt, o, sm); torch.cuda.synchronize()
            err = (o.float() - ref(q, latent, bt, sl, blk)).abs().max().item()
            me = bench(lambda q,l,bt,sl,o: (dec.plan(sl), dec.run(q,l,bt,o,sm)), q, latent, bt, sl)
            tsp = bench(lambda q,l,bt,sl,o: decode_blocktable_mla_opt(q,l,bt,sl,sm,blk,KV_LORA,o), q, latent, bt, sl)
            tks = bench(lambda q,l,bt,sl,o: decode_blocktable_mla_split(q,l,bt,sl,sm,blk,KV_LORA,o), q, latent, bt, sl)
            tb = max(tsp, tks)
            tag = "OK" if err < 3e-2 else "FAIL<<"
            print(f"{bs:>5} {pat:>10} | {dec.target_ctas:>6} {me:>6.0f} {tsp:>6.0f} {tks:>6.0f} | {me/tb:.2f}x  {err:.1e} {tag}")
