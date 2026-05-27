"""MLADecoder init/plan/run test: correctness, ragged-batch balance, CUDA-graph capture.
  init  -> allocate metadata+scratch (once)
  plan  -> schedule kernel populates the load-balanced work queue from seqlens
  run   -> decode over the work queue (graph-capturable)"""
import os, statistics, random, torch
from torch.utils.cpp_extension import load
from vortex_torch.engine.sgl.attention_backend.triton_mla_kernel import decode_blocktable_mla_opt
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
    bs, h, _ = q.shape; out = torch.empty(bs, h, KV_LORA, device='cuda', dtype=torch.float32)
    qf, lf = q.float(), latent.float()
    for b in range(bs):
        s = int(sl[b]); nb = (s + blk - 1) // blk
        rows = [torch.arange(int(bt[b, j]) * blk, int(bt[b, j]) * blk + blk, device='cuda') for j in range(nb)]
        slots = torch.cat(rows)[:s]; k = lf[slots]
        out[b] = torch.softmax((qf[b] @ k.t()) * sm, -1) @ k[:, :KV_LORA]
    return out

print("=== 1) correctness (init, plan, run) ===")
for nm, sls in [("uniform 2048", [2048] * 16), ("ragged", [2048,128,512,4096,1,777,2000,64]*2),
                ("non-mult 2000", [2000] * 16), ("all-1-token", [1]*16)]:
    blk = 64; bs = len(sls); q, latent, bt, sl, nb = mk(bs, blk, sls)
    dec = mod.MLADecoder(bs, H, blk, nb, 16)
    o = torch.empty(bs, H, KV_LORA, device='cuda', dtype=torch.bfloat16)
    dec.plan(sl); dec.run(q, latent, bt, o, sm); torch.cuda.synchronize()
    e = (o.float() - ref(q, latent, bt, sl, blk)).abs().max().item()
    print(f"  {nm:16s} max_err={e:.2e}  {'OK' if e < 3e-2 else 'FAIL <<<'}")

print("\n=== 1b) ONE plan() drives MANY run()s (simulated layers, distinct q/latent) ===")
blk = 64; bs = 16; sls = [2048,128,512,4096,1,777,2000,64]*2
q0, _, bt, sl, nb = mk(bs, blk, sls)
dec = mod.MLADecoder(bs, H, blk, nb, 16)
dec.plan(sl)                                   # schedule ONCE for the whole step
maxerr = 0.0
for layer in range(8):                          # each "layer": fresh q + latent, SAME plan
    q = torch.randn(bs, H, KV_DIM, device='cuda', dtype=torch.bfloat16)
    nbt = bt.size(1); latent = torch.randn(bs * nbt * blk, KV_DIM, device='cuda', dtype=torch.bfloat16)
    o = torch.empty(bs, H, KV_LORA, device='cuda', dtype=torch.bfloat16)
    dec.run(q, latent, bt, o, sm); torch.cuda.synchronize()
    maxerr = max(maxerr, (o.float() - ref(q, latent, bt, sl, blk)).abs().max().item())
print(f"  8 layers, 1 plan: max_err={maxerr:.2e}  {'OK' if maxerr < 3e-2 else 'FAIL <<<'}")

print("\n=== 2) CUDA-graph capture [plan(); N x run()] + replay (seqlens change in place) ===")
blk = 64; bs = 32; NLAYERS = 4
q, latent, bt, sl, nb = mk(bs, blk, [4096] + [2048] * (bs - 1))  # bt sized for max 4096
sl.copy_(torch.full((bs,), 2048, device='cuda', dtype=torch.int32))
dec = mod.MLADecoder(bs, H, blk, nb, 16)
qs = [torch.randn(bs, H, KV_DIM, device='cuda', dtype=torch.bfloat16) for _ in range(NLAYERS)]
os_ = [torch.empty(bs, H, KV_LORA, device='cuda', dtype=torch.bfloat16) for _ in range(NLAYERS)]
def step():
    dec.plan(sl)
    for i in range(NLAYERS): dec.run(qs[i], latent, bt, os_[i], sm)
for _ in range(3): step()
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g): step()
sl.copy_(torch.tensor([1024, 4096, 333, 2000] * (bs // 4), device='cuda', dtype=torch.int32))
g.replay(); torch.cuda.synchronize()
e = max((os_[i].float() - ref(qs[i], latent, bt, sl, blk)).abs().max().item() for i in range(NLAYERS))
print(f"  graph: 1 plan + {NLAYERS} layers, replay after seqlen change: max_err={e:.2e}  {'OK' if e < 3e-2 else 'FAIL <<<'}")

print("\n=== 3) ragged-batch throughput: work-queue plan/run vs Triton ===")
def bench(call, q, latent, bt, sl, reps=10):
    o = torch.empty(q.size(0), H, KV_LORA, device='cuda', dtype=torch.bfloat16); vals = []
    for _ in range(reps):
        for _ in range(15): call(q, latent, bt, sl, o)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True); s.record()
        for _ in range(40): call(q, latent, bt, sl, o)
        e.record(); torch.cuda.synchronize()
        vals.append(int(sl.sum()) * KV_DIM * 2 / ((s.elapsed_time(e) / 40) * 1e-3) / 1e9)
    return statistics.median(vals)
blk = 64; bs = 128; random.seed(0)
for label, sls in [("uniform 2048", [2048]*bs),
                   ("ragged 2x (512-4096)", [random.choice([512,1024,2048,3072,4096]) for _ in range(bs)]),
                   ("heavy skew (16 long)", [4096 if i<16 else 256 for i in range(bs)])]:
    nb = (max(sls)+blk-1)//blk; q, latent, bt, sl, _ = mk(bs, blk, sls)
    dec = mod.MLADecoder(bs, H, blk, nb, 16)
    def mine(q,l,bt,sl,o): dec.plan(sl); dec.run(q,l,bt,o,sm)
    me = bench(mine, q, latent, bt, sl)
    tr = bench(lambda q,l,bt,sl,o: decode_blocktable_mla_opt(q,l,bt,sl,sm,blk,KV_LORA,o), q, latent, bt, sl)
    print(f"  {label:24s} mine={me:5.0f}  triton={tr:5.0f}  ratio={me/tr:.3f}")
