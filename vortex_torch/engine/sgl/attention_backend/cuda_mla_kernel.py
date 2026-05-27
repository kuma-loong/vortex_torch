"""JIT loader + Python wrapper for the hand-written CUDA block-table MLA decode.

Mirrors the closure/cache pattern of ``vortex_torch/indexer/utils_sglang.py``:
``get_cuda_mla_module()`` JIT-compiles ``csrc/mla_cuda.cu`` once (process-cached)
and returns the extension module; ``decode_blocktable_mla_cuda`` is a drop-in for
``triton_mla_kernel.decode_blocktable_mla`` (identical signature) so the backend
swap is a one-line change.

The kernel (``csrc/mla_ldm.cuh``) is a from-scratch CUDA flash-decode: ldmatrix +
``mma.sync.m16n8k16`` GEMMs, bf16-packed register-O, register-resident online
softmax, a 576->584 smem bank-conflict pad, ``__launch_bounds__`` occupancy
forcing, and split-KV. See ``cuda_mla/REPORT.md`` for the design + benchmarks.
"""
import os

import torch
from torch.utils.cpp_extension import load

_HERE = os.path.dirname(os.path.abspath(__file__))
_CSRC = os.path.join(_HERE, "csrc")
_MODULE = None
_SM_COUNT: dict = {}
# Optional override of the split-KV count (>=1). In a full model the MLA decode is
# a small fraction of the forward, so the split path's extra stage-2 + MidO traffic
# can cost more than its parallelism buys; setting this to 1 forces single-pass.
_SPLITS_ENV = os.environ.get("VORTEX_CUDA_MLA_SPLITS")


def get_cuda_mla_module():
    """JIT-compile (once, process-cached) and return the CUDA MLA extension."""
    global _MODULE
    if _MODULE is None:
        build_dir = os.path.join(_CSRC, "build")
        os.makedirs(build_dir, exist_ok=True)
        _MODULE = load(
            name="vortex_cuda_mla",
            sources=[os.path.join(_CSRC, "mla_cuda.cu")],
            extra_cuda_cflags=["-O3", "-arch=sm_100a", "--use_fast_math",
                               "--expt-relaxed-constexpr", "-lineinfo"],
            extra_include_paths=[_CSRC],
            build_directory=build_dir,
            verbose=False,
        )
    return _MODULE


def _sm_count(device) -> int:
    idx = device.index if hasattr(device, "index") else torch.cuda.current_device()
    if idx not in _SM_COUNT:
        _SM_COUNT[idx] = torch.cuda.get_device_properties(device).multi_processor_count
    return _SM_COUNT[idx]


def make_mla_decoder(bs: int, H: int, block_size: int, max_blocks: int,
                     max_split_cap: int = -1, chunk_min: int = -1, minb: int = -1):
    """flashinfer-style plan/run decoder. Allocate ONCE per (bs) for a fixed launch
    geometry; ``plan(seqlens)`` builds the load-balanced work queue (once per decode
    step, shared by all layers), ``run(q, latent, block_table, o, sm_scale)`` does the
    per-layer decode. Both are fixed-grid => cuda-graph-capturable."""
    return get_cuda_mla_module().MLADecoder(
        bs, H, block_size, max_blocks, max_split_cap, chunk_min, minb)


def decode_blocktable_mla_cuda(
    q: torch.Tensor,            # [bs, H, Lk]   fused [q_nope_out | q_pe], Lk=576
    latent: torch.Tensor,       # [num_slots, Lk]  fused [kv_c | k_pe]
    block_table: torch.Tensor,  # [bs, max_blocks] selected page ids (int32)
    seqlens: torch.Tensor,      # [bs] sparse token count (int32)
    sm_scale: float,
    block_size: int,
    kv_lora_rank: int,          # Lv = 512
    o: torch.Tensor = None,     # [bs, H, Lv] (allocated if None)
    splits: int = None,
) -> torch.Tensor:
    """Drop-in replacement for ``decode_blocktable_mla`` backed by the CUDA kernel."""
    bs, H, Lk = q.shape
    Lv = kv_lora_rank
    assert Lk == 576 and Lv == 512, f"MLA latent must be 512+64=576 (got Lk={Lk}, Lv={Lv})"
    if o is None:
        o = q.new_empty((bs, H, Lv))
    if splits is None:
        if _SPLITS_ENV is not None:
            splits = max(1, int(_SPLITS_ENV))
        else:
            # split-KV count. In a FULL model the MLA decode is a small fraction of
            # the forward, so the split path's extra stage-2 + MidO traffic (which
            # scales with `splits`) easily outweighs its parallelism — measured on
            # DeepSeek-V2-Lite RULER: splits=2 matched Triton (378 vs 376 tok/s),
            # while the standalone-microbench-optimal ~7 splits was ~30% slower.
            # So target a MODEST fill (~1 CTA/SM, total CTAs ~ bs*splits ~ SM),
            # capped at 4. Pure host arithmetic (bs + SM count) => fixed launch
            # geometry => cuda-graph-capture safe. Override with VORTEX_CUDA_MLA_SPLITS.
            splits = max(1, min(4, _sm_count(q.device) // max(1, bs)))
    get_cuda_mla_module().decode(
        q, latent, block_table.to(torch.int32), seqlens.to(torch.int32), o,
        float(sm_scale), int(block_size), int(splits),
    )
    return o
