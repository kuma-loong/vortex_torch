"""JIT-compilation helpers for CUDA-backed custom_ops leaves.

Each leaf describes its kernel via:

  * a ``kernel.cu`` source file living next to the leaf's ``dispatch.py``,
  * a ``config.json`` with ``{"file": <kernel.cu>, "substitutions": {...},
    "extra_cuda_cflags": [...]}``,
  * the matching C ABI string (a ``cpp_source`` parameter to
    :func:`load_submission`).

The compiled module is cached process-wide keyed on
``(source path, substitutions, extra_cflags, cpp_source hash)`` so
repeated calls from different leaves resolve to the same extension if
they share all four.

This module is the *single* JIT path under ``custom_ops/``; no leaf
should ``torch.utils.cpp_extension.load_inline`` directly.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

from torch.utils.cpp_extension import load_inline


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

_module_cache: dict[tuple, object] = {}
_module_lock = threading.Lock()


def _resolve_source(filename: str | os.PathLike) -> Path:
    p = Path(filename)
    if not p.is_absolute():
        raise ValueError(
            f"custom_ops._jit: kernel source path must be absolute; got {filename!r}"
        )
    p = p.resolve()
    if not p.exists():
        raise FileNotFoundError(f"CUDA source not found: {p}")
    return p


def get_kernel(
    filename: str | os.PathLike,
    cpp_source: str,
    *,
    substitutions: dict[str, object] | None = None,
    extra_cuda_cflags: tuple[str, ...] | None = None,
    verbose: bool = True,
    build_dir_env: str = "VORTEX_CUSTOM_OPS_BUILD_DIR",
):
    """Compile and cache a CUDA source against ``cpp_source``.

    ``substitutions`` are sentinel-replacements applied to the kernel
    source text before compilation. ``extra_cuda_cflags`` extend the
    default nvcc flag set. The compiled module is returned; the leaf
    typically reads ``module.topk`` (the function declared in the ABI).
    """
    src_path = _resolve_source(filename)
    subs = dict(substitutions or {})
    extra_cuda = tuple(extra_cuda_cflags or ())

    key = (
        str(src_path),
        tuple(sorted((k, str(v)) for k, v in subs.items())),
        extra_cuda,
        hashlib.sha256(cpp_source.encode("utf-8")).hexdigest()[:8],
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
        module_name = f"vortex_custom_op_{src_path.stem}_{digest}"

        cu_flags = list(_DEFAULT_CUDA_CFLAGS) + list(extra_cuda)

        build_dir = os.environ.get(build_dir_env)
        kwargs = {}
        if build_dir:
            os.makedirs(build_dir, exist_ok=True)
            kwargs["build_directory"] = build_dir

        module = load_inline(
            name=module_name,
            cpp_sources=cpp_source,
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
    cpp_source: str,
    *,
    verbose: bool = True,
    build_dir_env: str = "VORTEX_CUSTOM_OPS_BUILD_DIR",
):
    """Compile a kernel described by ``json_path``.

    JSON schema::

        {
            "file": "kernel.cu",              // required (str), resolved
                                              // against the JSON's parent dir
            "substitutions": {...},           // optional ({str: scalar})
            "extra_cuda_cflags": ["-DFOO=1"]  // optional (list[str])
        }
    """
    cfg_path = Path(json_path).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"custom_ops config not found: {cfg_path}")

    with cfg_path.open("r") as f:
        cfg = json.load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"{cfg_path}: top-level JSON must be an object")
    if "file" not in cfg:
        raise ValueError(f"{cfg_path}: missing required field 'file'")
    file_field = cfg["file"]
    if not isinstance(file_field, str):
        raise ValueError(
            f"{cfg_path}: 'file' must be a string, got {type(file_field).__name__}"
        )

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
        cpp_source,
        substitutions=substitutions,
        extra_cuda_cflags=tuple(extra_cuda_cflags),
        verbose=verbose,
        build_dir_env=build_dir_env,
    )


__all__ = ["get_kernel", "load_submission"]
