import torch
from torch.utils.cpp_extension import load
HERE="cuda_mla/spec"; KV_DIM,KV_LORA,H=576,512,20; sm=1.0/(KV_DIM**0.5)
mod=load(name="vk1prof",sources=[HERE+"/k_h20_bs128_blk64.cu"],
  extra_cuda_cflags=["-O3","-arch=sm_100a","--use_fast_math","-lineinfo"],
  extra_include_paths=[HERE],build_directory=HERE+"/build_prof",verbose=False)
bs,blk,tok=128,64,2048; nb=tok//blk; npg=bs*nb
latent=torch.randn(npg*blk,KV_DIM,device='cuda',dtype=torch.bfloat16)
btab=torch.randperm(npg,device='cuda',dtype=torch.int32).view(bs,nb).contiguous()
sl=torch.full((bs,),tok,device='cuda',dtype=torch.int32)
q=torch.randn(bs,H,KV_DIM,device='cuda',dtype=torch.bfloat16)
o=torch.empty(bs,H,KV_LORA,device='cuda',dtype=torch.bfloat16)
for _ in range(10): mod.run(q,latent,btab,sl,o,sm)
torch.cuda.synchronize()
for _ in range(3): mod.run(q,latent,btab,sl,o,sm)
torch.cuda.synchronize()
