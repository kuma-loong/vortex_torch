"""Generic ``load_inline`` dispatcher for top-k kernels.

Reads a ``.cu`` file from disk, applies optional sentinel substitutions
(compile-time knobs), and JIT-compiles it via
``torch.utils.cpp_extension.load_inline``. Each unique
``(file, substitutions, extra_cuda_cflags)`` triple is compiled once and
cached process-wide.

All top-k sources expose the same C entry point::

    void topk(
        const at::Tensor& x,
        const at::Tensor& dense_kv_indptr,
        const at::Tensor& sparse_kv_indptr,
        const at::Tensor& dense_kv_indices,
        at::Tensor&       sparse_kv_indices,
        const int64_t     eff_batch_size,
        const int64_t     reserved_bos,
        const int64_t     reserved_eos,
        const int64_t     max_num_pages);
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Optional

from torch.utils.cpp_extension import load_inline


_CPP_SOURCE = r"""
#include <torch/extension.h>

void topk(
    const at::Tensor& x,
    const at::Tensor& dense_kv_indptr,
    const at::Tensor& sparse_kv_indptr,
    const at::Tensor& dense_kv_indices,
    at::Tensor&       sparse_kv_indices,
    const int64_t     eff_batch_size,
    const int64_t     reserved_bos,
    const int64_t     reserved_eos,
    const int64_t     max_num_pages);
"""

_DEFAULT_CFLAGS: tuple[str, ...] = ("-O3",)
_DEFAULT_CUDA_CFLAGS: tuple[str, ...] = (
    "-O3",
    "--use_fast_math",
    "--expt-relaxed-constexpr",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
)

_HERE = Path(__file__).resolve().parent

_module_cache: dict[tuple, object] = {}
_module_lock = threading.Lock()


def _resolve_source(filename: str | os.PathLike) -> Path:
    p = Path(filename)
    if not p.is_absolute():
        p = _HERE / p
    p = p.resolve()
    if not p.exists():
        raise FileNotFoundError(f"CUDA source not found: {p}")
    return p


def get_kernel(
    filename: str | os.PathLike,
    substitutions: dict[str, object] | None = None,
    extra_cuda_cflags: tuple[str, ...] | None = None,
    verbose: bool = True,
    build_dir_env: str = "VORTEX_TOPK_BUILD_DIR",
):
    """Compile and cache a top-k CUDA source.

    Arguments:
        filename: path to a ``.cu`` file. Relative paths resolve against
            this module's directory.
        substitutions: ``{sentinel: value}`` map of compile-time knobs.
            Each sentinel string in the source is replaced by ``str(value)``.
        extra_cuda_cflags: additional nvcc flags appended to the defaults.
        verbose: pass-through to ``load_inline``.
        build_dir_env: env var consulted for an override of the build dir.

    Returns the loaded extension module. Call ``module.topk(...)``.
    """
    src_path = _resolve_source(filename)
    subs = dict(substitutions or {})
    extra_cuda = tuple(extra_cuda_cflags or ())

    key = (
        str(src_path),
        tuple(sorted((k, str(v)) for k, v in subs.items())),
        extra_cuda,
    )

    cached = _module_cache.get(key)
    if cached is not None:
        return cached

    with _module_lock:
        cached = _module_cache.get(key)
        if cached is not None:
            return cached

        cuda_source = src_path.read_text()
        for sentinel, value in subs.items():
            cuda_source = cuda_source.replace(sentinel, str(value))

        digest = hashlib.sha256(repr(key).encode("utf-8")).hexdigest()[:12]
        module_name = f"vortex_topk_{src_path.stem}_{digest}"

        cu_flags = list(_DEFAULT_CUDA_CFLAGS) + list(extra_cuda)

        build_dir = os.environ.get(build_dir_env)
        kwargs = {}
        if build_dir:
            os.makedirs(build_dir, exist_ok=True)
            kwargs["build_directory"] = build_dir

        module = load_inline(
            name=module_name,
            cpp_sources=_CPP_SOURCE,
            cuda_sources=cuda_source,
            functions=["topk"],
            extra_cflags=list(_DEFAULT_CFLAGS),
            extra_cuda_cflags=cu_flags,
            with_cuda=True,
            verbose=verbose,
            **kwargs,
        )
        _module_cache[key] = module
        return module


def load_submission(
    json_path: str | os.PathLike,
    verbose: bool = True,
    build_dir_env: str = "VORTEX_TOPK_BUILD_DIR",
):
    """Compile a submission described by a JSON config.

    Schema::

        {
            "file": "radix_topk.cu",              // required (str)
            "substitutions": {                    // optional ({str: scalar})
                "__THREADS_PER_BLOCK__": 1024,
                "__VORTEX_MAX_TOPK__": 2048,
                "__SMEM_BYTES__": 32768
            },
            "extra_cuda_cflags": ["-DFOO=1"]      // optional (list[str])
        }

    The ``file`` field is resolved relative to the JSON's parent directory
    (so configs can sit next to their ``.cu`` sources). Absolute paths are
    used as-is.
    """
    cfg_path = Path(json_path).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"submission config not found: {cfg_path}")

    with cfg_path.open("r") as f:
        cfg = json.load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"{cfg_path}: top-level JSON must be an object")

    if "file" not in cfg:
        raise ValueError(f"{cfg_path}: missing required field 'file'")
    file_field = cfg["file"]
    if not isinstance(file_field, str):
        raise ValueError(f"{cfg_path}: 'file' must be a string, got {type(file_field).__name__}")

    src_path = Path(file_field)
    if not src_path.is_absolute():
        src_path = (cfg_path.parent / src_path).resolve()

    substitutions = cfg.get("substitutions", {})
    if not isinstance(substitutions, dict):
        raise ValueError(f"{cfg_path}: 'substitutions' must be an object")

    extra_cuda_cflags = cfg.get("extra_cuda_cflags", [])
    if not isinstance(extra_cuda_cflags, list) or not all(
        isinstance(x, str) for x in extra_cuda_cflags
    ):
        raise ValueError(f"{cfg_path}: 'extra_cuda_cflags' must be a list of strings")

    return get_kernel(
        src_path,
        substitutions=substitutions,
        extra_cuda_cflags=tuple(extra_cuda_cflags),
        verbose=verbose,
        build_dir_env=build_dir_env,
    )


_K96_WINNER_CONFIG = _HERE / "k_96" / "configs" / "claude_opus_4_7" / "batch_16_id3.json"
_K128_WINNER_CONFIG = _HERE / "k_128" / "configs" / "claude_opus_4_7" / "batch_10_id2.json"
_RADIX_WINNER_CONFIG = _HERE / "k_256" / "configs" / "claude_opus_4_7" / "batch_24_id1.json"
_SORT_BASELINE_CONFIG = _HERE / "configs" / "sort_default.json"
_K96_WINNER_MAX_TOPK = 96
_K128_WINNER_MAX_TOPK = 128
_RADIX_WINNER_MAX_TOPK = 256


def dispatch(max_topk: Optional[int] = None, *, verbose: bool = True):
    """Pick the best-known top-k kernel for ``max_topk`` and return its callable.

      - ``max_topk <= 96`` → ``k_96/configs/claude_opus_4_7/batch_16_id3.json``
        (radix winner: T=512, smem=4KB, x³ transform, adaptive_approx_skip
        with TOLERATE=10). Geomean 1.6227x vs CUB baseline with
        non-clustered R@96 ≥ 0.97.
      - ``96 < max_topk <= 128`` → ``k_128/configs/claude_opus_4_7/batch_10_id2.json``
        (radix winner: T=512, smem=3KB, signed x⁴ transform,
        adaptive_approx_skip with TOLERATE=14, MAX_ITERS=3). Geomean
        1.6468x vs CUB baseline with non-clustered R@128 ≥ 0.97.
      - ``128 < max_topk <= 256`` → ``k_256/configs/claude_opus_4_7/batch_24_id1.json``
        (radix winner: T=512, smem=8KB, x*|x| transform).
      - ``max_topk > 256`` or ``max_topk is None`` → ``configs/sort_default.json``
        (CUB ``BlockRadixSort`` baseline; supports any k).

    The returned callable has the entry-point signature documented at
    the top of this module. Modules are JIT-compiled on first hit and
    cached, so repeated ``dispatch`` calls are free of recompile overhead.
    """
    if max_topk is not None and max_topk <= _K96_WINNER_MAX_TOPK:
        cfg = _K96_WINNER_CONFIG
    elif max_topk is not None and max_topk <= _K128_WINNER_MAX_TOPK:
        cfg = _K128_WINNER_CONFIG
    elif max_topk is not None and max_topk <= _RADIX_WINNER_MAX_TOPK:
        cfg = _RADIX_WINNER_CONFIG
    else:
        cfg = _SORT_BASELINE_CONFIG
    return load_submission(cfg, verbose=verbose).topk
