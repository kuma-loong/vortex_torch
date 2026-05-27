"""Drive Triton single_pass_opt (the 1944 GB/s competitor) for ncu profiling."""
import torch
from vortex_torch.engine.sgl.attention_backend.triton_mla_kernel import decode_blocktable_mla_opt

KV_DIM, KV_LORA, H = 576, 512, 20
sm = 1.0 / (KV_DIM ** 0.5)
bs, blk, tok = 128, 64, 2048
nb = tok // blk; npg = bs * nb
latent = torch.randn(npg * blk, KV_DIM, device='cuda', dtype=torch.bfloat16)
bt = torch.randperm(npg, device='cuda', dtype=torch.int32).view(bs, nb).contiguous()
sl = torch.full((bs,), tok, device='cuda', dtype=torch.int32)
q = torch.randn(bs, H, KV_DIM, device='cuda', dtype=torch.bfloat16)
o = torch.empty(bs, H, KV_LORA, device='cuda', dtype=torch.bfloat16)
f = lambda: decode_blocktable_mla_opt(q, latent, bt, sl, sm, blk, KV_LORA, o)
for _ in range(20): f()
torch.cuda.synchronize()
for _ in range(3): f()
torch.cuda.synchronize()
