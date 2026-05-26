"""JIT-build + wrapper for the mma.sync register-O MLA decode kernel (mla_mma.cu)."""
import os
import torch
from torch.utils.cpp_extension import load

_HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(_HERE, "build_mma"), exist_ok=True)
_MOD = None


def _mod():
    global _MOD
    if _MOD is None:
        _MOD = load(name="vortex_mla_mma", sources=[os.path.join(_HERE, "mla_mma.cu")],
                    extra_cuda_cflags=["-O3", "-arch=sm_100a", "--use_fast_math", "-lineinfo"],
                    build_directory=os.path.join(_HERE, "build_mma"), verbose=False)
    return _MOD


_SM = {}


def _sm_count(dev):
    i = dev.index if hasattr(dev, "index") else torch.cuda.current_device()
    if i not in _SM:
        _SM[i] = torch.cuda.get_device_properties(dev).multi_processor_count
    return _SM[i]


def decode_mma(q, latent, block_table, seqlens, sm_scale, block_size, kv_lora_rank, o=None,
               splits=None):
    bs, H, Lk = q.shape
    if o is None:
        o = q.new_empty((bs, H, kv_lora_rank))
    if splits is None:
        # measured sweet spot ~ splits*bs in [128,256] (B200, 148 SMs, 4 warps/CTA):
        # bs8->16, bs32->8, bs64->4, bs128->2. clamp to [1,32].
        splits = max(1, min(32, 256 // max(1, bs)))
    _mod().mla_decode_mma(q, latent, block_table.to(torch.int32), seqlens.to(torch.int32),
                          o, float(sm_scale), int(block_size), int(splits))
    return o


def register():
    from vortex_torch.engine.sgl.attention_backend import triton_mla_kernel as T
    T.KERNELS["cuda_mma"] = decode_mma
    return T.KERNELS


if __name__ == "__main__":
    register(); print("built + registered cuda_mma:", _mod())
