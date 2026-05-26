"""JIT-build + wrapper for the tensor-core MLA decode kernel (mla_tc.cu)."""
import os
import torch
from torch.utils.cpp_extension import load

_HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(_HERE, "build_tc"), exist_ok=True)
_MOD = None


def _mod():
    global _MOD
    if _MOD is None:
        _MOD = load(
            name="vortex_mla_tc",
            sources=[os.path.join(_HERE, "mla_tc.cu")],
            extra_cuda_cflags=["-O3", "-arch=sm_100a", "--use_fast_math", "-lineinfo"],
            build_directory=os.path.join(_HERE, "build_tc"),
            verbose=False,
        )
    return _MOD


_SM = {}


def _sm_count(dev):
    i = dev.index if hasattr(dev, "index") else torch.cuda.current_device()
    if i not in _SM:
        _SM[i] = torch.cuda.get_device_properties(dev).multi_processor_count
    return _SM[i]


def decode_tc(q, latent, block_table, seqlens, sm_scale, block_size, kv_lora_rank, o=None,
              splits=None):
    bs, H, Lk = q.shape
    if o is None:
        o = q.new_empty((bs, H, kv_lora_rank))
    if splits is None:
        # one CTA per (request, split); target ~3 waves over the SMs
        splits = max(1, min(32, (3 * _sm_count(q.device)) // max(1, bs)))
    _mod().mla_decode_tc(q, latent, block_table.to(torch.int32), seqlens.to(torch.int32),
                         o, float(sm_scale), int(block_size), int(splits))
    return o


def register():
    from vortex_torch.engine.sgl.attention_backend import triton_mla_kernel as T
    T.KERNELS["cuda_tc"] = decode_tc
    return T.KERNELS


if __name__ == "__main__":
    register(); print("built + registered cuda_tc:", _mod())
