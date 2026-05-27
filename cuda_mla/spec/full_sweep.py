"""Full bs x blk table: MLADecoder (plan+run, bs-general work-queue) vs Triton best.
bs in {1..8} U {8*i, 1<=i<=16}; blk in {16,32,64}; sel=2048 uniform. Writes CSV + grid."""
import os, statistics, json, torch
from torch.utils.cpp_extension import load
from vortex_torch.engine.sgl.attention_backend.triton_mla_kernel import (
    decode_blocktable_mla_opt, decode_blocktable_mla_split)
HERE = "cuda_mla/spec"; KV_DIM, KV_LORA, H = 576, 512, 20; sm = 1.0 / (KV_DIM ** 0.5)
bd = HERE + "/build_decoder"; os.makedirs(bd, exist_ok=True)
mod = load(name="vortex_mla_decoder", sources=[HERE + "/mla_decoder.cu"],
           extra_cuda_cflags=["-O3", "-arch=sm_100a", "--use_fast_math", "-lineinfo"],
           extra_include_paths=[HERE], build_directory=bd, verbose=False)
SEL = 2048

def mk(bs, blk, tok):
    nb = (tok + blk - 1) // blk; npg = bs * nb
    latent = torch.randn(npg * blk, KV_DIM, device='cuda', dtype=torch.bfloat16)
    bt = torch.randperm(npg, device='cuda', dtype=torch.int32).view(bs, nb).contiguous()
    sl = torch.full((bs,), tok, device='cuda', dtype=torch.int32)
    q = torch.randn(bs, H, KV_DIM, device='cuda', dtype=torch.bfloat16)
    return q, latent, bt, sl, nb

def bench(call, q, latent, bt, sl, reps=6):
    o = torch.empty(q.size(0), H, KV_LORA, device='cuda', dtype=torch.bfloat16); vals = []
    for _ in range(reps):
        for _ in range(12): call(q, latent, bt, sl, o)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True); s.record()
        for _ in range(40): call(q, latent, bt, sl, o)
        e.record(); torch.cuda.synchronize()
        vals.append(int(sl.sum()) * KV_DIM * 2 / ((s.elapsed_time(e) / 40) * 1e-3) / 1e9)
    return statistics.median(vals)

bss = sorted(set(list(range(1, 9)) + [8 * i for i in range(1, 17)]))
rows = []
for blk in (16, 32, 64):
    for bs in bss:
        nb = (SEL + blk - 1) // blk
        q, latent, bt, sl, _ = mk(bs, blk, SEL)
        dec = mod.MLADecoder(bs, H, blk, nb)
        me = bench(lambda q,l,bt,sl,o: (dec.plan(sl), dec.run(q,l,bt,o,sm)), q, latent, bt, sl)
        tsp = bench(lambda q,l,bt,sl,o: decode_blocktable_mla_opt(q,l,bt,sl,sm,blk,KV_LORA,o), q, latent, bt, sl)
        tks = bench(lambda q,l,bt,sl,o: decode_blocktable_mla_split(q,l,bt,sl,sm,blk,KV_LORA,o), q, latent, bt, sl)
        tb = max(tsp, tks); best = 'sp' if tsp >= tks else 'ks'
        rows.append(dict(blk=blk, bs=bs, mine=me, tri_sp=tsp, tri_ks=tks, ratio=me/tb, tbest=best))
        print(f"blk={blk:2d} bs={bs:3d}: mine={me:5.0f}  tri_sp={tsp:5.0f}  tri_ks={tks:5.0f}  ratio={me/tb:.2f}x")

json.dump(rows, open(HERE + "/full_sweep.json", "w"), indent=0)
# pretty grid per blk
print("\n\n##### TABLE (GB/s mine | ratio vs Triton-best), sel=2048 uniform #####")
for blk in (16, 32, 64):
    print(f"\n### block_size = {blk} ###")
    print(f"{'bs':>4} | {'mine':>6} {'tri_sp':>6} {'tri_ks':>6} | {'ratio':>6}")
    for r in rows:
        if r['blk'] == blk:
            print(f"{r['bs']:>4} | {r['mine']:>6.0f} {r['tri_sp']:>6.0f} {r['tri_ks']:>6.0f} | {r['ratio']:>5.2f}x")
