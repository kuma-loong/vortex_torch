"""JIT-build + Python wrapper for the custom CUDA block-table MLA decode kernel.

Importing this module compiles cuda_mla/mla_decode.cu (once, cached) and exposes
`decode_blocktable_mla_cuda(...)` with the vortex KERNELS signature. It also
registers itself into the triton_mla_kernel.KERNELS registry as "cuda" so the
microbenchmark and plan can see it.
"""
import os
import torch
from torch.utils.cpp_extension import load

_HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(_HERE, "build"), exist_ok=True)

_MOD = None


def _mod():
    global _MOD
    if _MOD is None:
        _MOD = load(
            name="vortex_cuda_mla",
            sources=[os.path.join(_HERE, "mla_decode.cu")],
            extra_cuda_cflags=["-O3", "-arch=sm_100a", "--use_fast_math",
                               "-lineinfo"],
            build_directory=os.path.join(_HERE, "build"),
            verbose=False,
        )
    return _MOD


_SM = {}


def _sm_count(dev):
    i = dev.index if hasattr(dev, "index") else torch.cuda.current_device()
    if i not in _SM:
        _SM[i] = torch.cuda.get_device_properties(dev).multi_processor_count
    return _SM[i]


_MAX_SPLITS = 16


def _pick_splits(bs, sm_count):
    # one head-block (all H share the CTA), so program count ~ bs * splits.
    # target ~4 waves over the SMs; high bs collapses to splits=1 (no mid-buffer).
    return max(1, min(_MAX_SPLITS, (4 * sm_count) // max(1, bs)))


def decode_blocktable_mla_cuda(
    q, latent, block_table, seqlens, sm_scale, block_size, kv_lora_rank, o=None,
    splits=None,
):
    bs, H, Lk = q.shape
    Lv = kv_lora_rank
    assert Lk == 576 and Lv == 512, f"MLA latent must be 512+64 (got {Lk},{Lv})"
    if o is None:
        o = q.new_empty((bs, H, Lv))
    if splits is None:
        splits = _pick_splits(bs, _sm_count(q.device))
    _mod().mla_decode(q, latent, block_table.to(torch.int32), seqlens.to(torch.int32),
                      o, float(sm_scale), int(block_size), int(splits))
    return o


def register():
    """Add the CUDA kernel to triton_mla_kernel.KERNELS (idempotent)."""
    from vortex_torch.engine.sgl.attention_backend import triton_mla_kernel as T
    T.KERNELS["cuda"] = decode_blocktable_mla_cuda
    return T.KERNELS


if __name__ == "__main__":
    register()
    print("built + registered 'cuda' kernel; module:", _mod())
