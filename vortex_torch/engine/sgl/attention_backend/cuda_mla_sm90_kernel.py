"""SM90 copy of the hand-written CUDA block-table MLA decode loader.

This module intentionally remains separate from ``cuda_mla_kernel.py``. The
kernel sources are an arch-only copy so SM90 changes and tuning cannot alter the
existing SM100 backend.

Mirrors the closure/cache pattern of ``vortex_torch/indexer/utils_sglang.py``:
``get_cuda_mla_module()`` JIT-compiles ``csrc/sm90/mla_cuda_sm90.cu`` once
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
_CSRC = os.path.join(_HERE, "csrc", "sm90")
_MODULE = None
_SM_COUNT: dict = {}
# Optional override of the split-KV count (>=1). In a full model the MLA decode is
# a small fraction of the forward, so the split path's extra stage-2 + MidO traffic
# can cost more than its parallelism buys; setting this to 1 forces single-pass.
_SPLITS_ENV = os.environ.get("VORTEX_CUDA_MLA_SPLITS")


def _geometry_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    if raw.lower() == "auto":
        # Reproduce the arch-only copy: the C++ geometry derives this value from
        # the H100 properties / sparse budget when Python passes a non-positive
        # sentinel.  Keeping this selectable avoids editing or checking out the
        # tuned source when the formal matrix switches between stages.
        return -1
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be positive or 'auto', got {raw!r}.")
    return value


# H100 sweep defaults. These apply only to the independent SM90 backend; the
# original SM100 backend and sources remain untouched. Environment overrides
# keep the arch-only geometry reproducible and allow follow-up tuning without a
# source edit.
SM90_MAX_SPLIT_CAP = _geometry_env("VORTEX_CUDA_MLA_SM90_MAX_SPLIT_CAP", 32)
SM90_CHUNK_MIN = _geometry_env("VORTEX_CUDA_MLA_SM90_CHUNK_MIN", 16)
SM90_MINB = _geometry_env("VORTEX_CUDA_MLA_SM90_MINB", 3)


def get_cuda_mla_module():
    """JIT-compile (once, process-cached) and return the CUDA MLA extension."""
    global _MODULE
    if _MODULE is None:
        build_dir = os.environ.get(
            "VORTEX_CUDA_MLA_SM90_BUILD_DIR", os.path.join(_CSRC, "build")
        )
        os.makedirs(build_dir, exist_ok=True)
        _MODULE = load(
            name="vortex_cuda_mla_sm90",
            sources=[os.path.join(_CSRC, "mla_cuda_sm90.cu")],
            extra_cuda_cflags=["-O3", "-arch=sm_90a", "--use_fast_math",
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


# kv_lora_rank — the latent width the kernel reduces over (ldm::CKV). Constant for
# MLA; mid_o is [target_ctas, M, CKV]. The decode path asserts Lk==576, Lv==512.
CKV = 512


def mla_decoder_geometry(bs: int, H: int, block_size: int, max_blocks: int,
                         max_split_cap: int = SM90_MAX_SPLIT_CAP,
                         chunk_min: int = SM90_CHUNK_MIN,
                         minb: int = SM90_MINB):
    """Host-only launch geometry for a decode batch size — NO allocation. Returns a
    dict with ``target_ctas`` (run_wq grid / work-queue length) and ``M`` (=ceil(H/16)*16,
    the mid-buffer head-pad). Used to size the shared scratch at the max decode bs."""
    tc, M, minb_, target, cap, cmin = get_cuda_mla_module().mla_decoder_geometry(
        bs, H, block_size, max_blocks, max_split_cap, chunk_min, minb)
    return {"target_ctas": tc, "M": M, "minb": minb_,
            "target": target, "max_split_cap": cap, "chunk_min": cmin}


def allocate_mla_buffers(max_bs: int, H: int, block_size: int, max_blocks: int,
                         device, max_split_cap: int = SM90_MAX_SPLIT_CAP,
                         chunk_min: int = SM90_CHUNK_MIN,
                         minb: int = SM90_MINB) -> dict:
    """Allocate the work-queue + split-reduction scratch ONCE, sized for the maximum
    decode batch size. ``target_ctas`` grows monotonically with bs, so these buffers
    cover every smaller bs; all per-bs ``MLADecoder``s slice into them. This replaces
    the old one-scratch-allocation-per-cuda-graph-bs behaviour, which duplicated the
    (largest) ``target_ctas*M*512`` fp32 ``mid_o`` for every captured batch size."""
    g = mla_decoder_geometry(max_bs, H, block_size, max_blocks,
                             max_split_cap, chunk_min, minb)
    tc, M = g["target_ctas"], g["M"]
    i32 = {"dtype": torch.int32, "device": device}
    f32 = {"dtype": torch.float32, "device": device}
    # Uninitialised (torch.empty) is correct: the contract is plan -> run. Every decode
    # step, plan() (schedule_wq) repopulates the whole work queue (work_*), and run()'s
    # stage1 writes the mid_* scratch for exactly the active work items
    # [0, sum(nsplits)) that stage2_wq then reduces. The schedule keeps
    # sum(nsplits) <= target_ctas (the +bs margin), and every slot a reduction reads,
    # [WorkOffset[b], WorkOffset[b+1]), is an active item stage1 wrote — so nothing here
    # is ever read before being written, even with one scratch shared across batch
    # sizes / steps. (Verified: full-scratch poison does not leak into the output.)
    return {
        "work_batch": torch.empty(tc, **i32),
        "work_kv_start": torch.empty(tc, **i32),
        "work_kv_end": torch.empty(tc, **i32),
        "work_offset": torch.empty(max_bs + 1, **i32),
        "mid_o": torch.empty((tc, M, CKV), **f32),
        "mid_m": torch.empty((tc, M), **f32),
        "mid_l": torch.empty((tc, M), **f32),
    }


def make_mla_decoder(bs: int, H: int, block_size: int, max_blocks: int, buffers: dict,
                     max_split_cap: int = SM90_MAX_SPLIT_CAP,
                     chunk_min: int = SM90_CHUNK_MIN,
                     minb: int = SM90_MINB):
    """flashinfer-style plan/run decoder. The launch geometry is fixed per (bs), but
    the work-queue / split scratch is NOT allocated here — pass ``buffers`` from
    ``allocate_mla_buffers(max_bs, ...)`` (sized for the max decode bs) and the decoder
    slices them to this bs's ``target_ctas``. ``plan(seqlens)`` builds the load-balanced
    work queue (once per decode step, shared by all layers); ``run(q, latent,
    block_table, o, sm_scale)`` does the per-layer decode. Both are fixed-grid =>
    cuda-graph-capturable."""
    return get_cuda_mla_module().MLADecoder(
        bs, H, block_size, max_blocks,
        buffers["work_batch"], buffers["work_kv_start"], buffers["work_kv_end"],
        buffers["work_offset"], buffers["mid_o"], buffers["mid_m"], buffers["mid_l"],
        max_split_cap, chunk_min, minb)


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
