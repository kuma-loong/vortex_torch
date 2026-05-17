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

# Trtllm/block-table variants take an *indptr-free* ABI: per-row token
# counts via ``dense_seqlens`` / ``sparse_seqlens`` (length ``eff_bs``)
# and the block stride via ``block_size`` (so the kernel can derive
# ``block_count = ceil(tokens / block_size)`` per row). The output index
# buffer is a 2D ``[eff_bs, max_blocks_per_seq]`` block-table.
_CPP_SOURCE_TRTLLM = r"""
#include <torch/extension.h>

void topk(
    const at::Tensor& x,
    const at::Tensor& dense_seqlens,
    const at::Tensor& sparse_seqlens,
    const at::Tensor& dense_block_tables,
    at::Tensor&       sparse_block_tables,
    const int64_t     eff_batch_size,
    const int64_t     reserved_bos,
    const int64_t     reserved_eos,
    const int64_t     max_blocks_per_seq,
    const int64_t     block_size);
"""

# Trtllm "Union" ABI — merges two (block_table, seqlens) pairs into a
# fresh (sparse_block_tables, sparse_seqlens) pair. See
# ``union_trtllm.cu`` for the per-row algorithm and tail-placement
# semantics.
_CPP_SOURCE_TRTLLM_UNION = r"""
#include <torch/extension.h>

void topk(
    const at::Tensor& dense_seqlens,
    at::Tensor&       sparse_seqlens,
    const at::Tensor& dense_block_tables,
    const at::Tensor& block_tables_0,
    const at::Tensor& seqlens_0,
    const at::Tensor& block_tables_1,
    const at::Tensor& seqlens_1,
    at::Tensor&       sparse_block_tables,
    const int64_t     eff_batch_size,
    const int64_t     max_blocks_per_seq,
    const int64_t     block_size);
"""

# Trtllm "block-table TopK(k)" ABI — like ``_CPP_SOURCE_TRTLLM`` but
#   * ``sparse_seqlens`` is **mutable** (the kernel writes it as one of
#     its outputs);
#   * ``topk_val`` is an explicit runtime argument (not derived from
#     ``sparse_seqlens`` like the regular trtllm topk does);
#   * row-too-small fallback (copy entire dense row) lives inside the
#     kernel — no planner pre-fill required.
# Powers the new ``TopK(k)`` indexer op (see ``triton_impl/topk.py``).
_CPP_SOURCE_TRTLLM_BLOCK_TABLE_TOPK = r"""
#include <torch/extension.h>

void topk(
    const at::Tensor& x,
    const at::Tensor& dense_seqlens,
    at::Tensor&       sparse_seqlens,
    const at::Tensor& dense_block_tables,
    at::Tensor&       sparse_block_tables,
    const int64_t     eff_batch_size,
    const int64_t     reserved_bos,
    const int64_t     reserved_eos,
    const int64_t     max_blocks_per_seq,
    const int64_t     block_size,
    const int64_t     topk_val);
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
    cpp_source: str = _CPP_SOURCE,
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

    # Key on cpp_source too: the same .cu file compiled against the
    # flashinfer ABI vs the trtllm ABI is a different module.
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
        module_name = f"vortex_topk_{src_path.stem}_{digest}"

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
    verbose: bool = True,
    build_dir_env: str = "VORTEX_TOPK_BUILD_DIR",
    cpp_source: str = _CPP_SOURCE,
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
        cpp_source=cpp_source,
    )


_K96_WINNER_CONFIG = _HERE / "k_96" / "configs" / "claude_opus_4_7" / "batch_16_id3.json"
_K128_WINNER_CONFIG = _HERE / "k_128" / "configs" / "claude_opus_4_7" / "batch_10_id2.json"
_RADIX_WINNER_CONFIG = _HERE / "k_256" / "configs" / "claude_opus_4_7" / "batch_24_id1.json"
_SORT_BASELINE_CONFIG = _HERE / "configs" / "sort_default.json"

# trtllm variants (2D block_tables instead of CSR indices). Only the radix
# winner and the CUB-sort baseline have trtllm builds — k_96 / k_128 fall
# back to the radix winner in trtllm mode.
_RADIX_WINNER_CONFIG_TRTLLM = _HERE / "k_256" / "configs" / "claude_opus_4_7" / "batch_24_id1_trtllm.json"
_SORT_BASELINE_CONFIG_TRTLLM = _HERE / "configs" / "sort_default_trtllm.json"

# Powers the ``TopK(k)`` indexer op (block-table + seqlens output).
_BLOCK_TABLE_TOPK_CONFIG_TRTLLM = _HERE / "configs" / "block_table_topk_default_trtllm.json"

# Powers the ``Union()`` output op (merges two (block_table, seqlens) pairs).
_UNION_CONFIG_TRTLLM = _HERE / "configs" / "union_default_trtllm.json"

_K96_WINNER_MAX_TOPK = 96
_K128_WINNER_MAX_TOPK = 128
_RADIX_WINNER_MAX_TOPK = 256


def dispatch(
    max_topk: Optional[int] = None,
    *,
    attention_backend: str = "flashinfer",
    verbose: bool = True,
):
    """Pick the best-known top-k kernel for ``(max_topk, attention_backend)``
    and return its callable.

    ``attention_backend == "flashinfer"`` (default; CSR ``kv_indices``):
      - ``max_topk <= 96`` → k_96 winner.
      - ``96 < max_topk <= 128`` → k_128 winner.
      - ``128 < max_topk <= 256`` → k_256 radix winner.
      - else / ``None`` → CUB sort baseline.

    ``attention_backend == "trtllm"`` (2D ``block_tables``):
      - ``max_topk <= 256`` → k_256 radix winner *trtllm* build.
      - else / ``None`` → CUB sort baseline *trtllm* build.

    The returned callable's C signature **differs across backends**:
      - flashinfer (``_CPP_SOURCE``): ``(x, dense_kv_indptr, sparse_kv_indptr,
        dense_kv_indices, sparse_kv_indices, eff_bs, bos, eos, max_num_pages)``
      - trtllm (``_CPP_SOURCE_TRTLLM``): ``(x, dense_seqlens, sparse_seqlens,
        dense_block_tables, sparse_block_tables, eff_bs, bos, eos,
        max_blocks_per_seq, block_size)`` — indptr-free; the kernel derives
        ``block_count = ceil(tokens / block_size)`` per row.
    The indexer codegen (``triton_impl/topk.py``) passes the right tensors
    per backend.

    Modules are JIT-compiled on first hit and cached.
    """
    backend = (attention_backend or "flashinfer").lower()

    if backend == "trtllm":
        if max_topk is not None and max_topk <= _RADIX_WINNER_MAX_TOPK:
            cfg = _RADIX_WINNER_CONFIG_TRTLLM
        else:
            cfg = _SORT_BASELINE_CONFIG_TRTLLM
        return load_submission(cfg, verbose=verbose, cpp_source=_CPP_SOURCE_TRTLLM).topk
    elif backend == "flashinfer":
        if max_topk is not None and max_topk <= _K96_WINNER_MAX_TOPK:
            cfg = _K96_WINNER_CONFIG
        elif max_topk is not None and max_topk <= _K128_WINNER_MAX_TOPK:
            cfg = _K128_WINNER_CONFIG
        elif max_topk is not None and max_topk <= _RADIX_WINNER_MAX_TOPK:
            cfg = _RADIX_WINNER_CONFIG
        else:
            cfg = _SORT_BASELINE_CONFIG
    else:
        raise ValueError(
            f"topk dispatch: unknown attention_backend {attention_backend!r}; "
            f"expected 'flashinfer' or 'trtllm'"
        )

    return load_submission(cfg, verbose=verbose).topk


def dispatch_block_table_topk(
    *,
    attention_backend: str = "trtllm",
    verbose: bool = True,
):
    """Pick the ``TopK(k)`` (block-table + seqlens output) callable.

    Currently only the trtllm backend has an implementation; raises
    :class:`NotImplementedError` for flashinfer (the existing flashinfer
    ``topK`` op uses planner-driven CSR layout and has no need for a
    block-table-shaped variant).

    The returned callable's C signature is:

        topk(
            x,                              // RAGGED scores [eff_bs * max_blocks_per_seq, 1, 1]
            dense_seqlens,                  // [eff_bs] int32, tokens
            sparse_seqlens,                 // [eff_bs] int32, tokens  (OUTPUT)
            dense_block_tables,             // [eff_bs, max_blocks_per_seq] int32
            sparse_block_tables,            // [eff_bs, max_blocks_per_seq] int32  (OUTPUT)
            eff_batch_size,
            reserved_bos, reserved_eos,
            max_blocks_per_seq,
            block_size,
            topk_val,                       // explicit k (excludes bos+eos)
        )
    """
    backend = (attention_backend or "trtllm").lower()
    if backend != "trtllm":
        raise NotImplementedError(
            f"dispatch_block_table_topk: only the 'trtllm' backend is supported; "
            f"got {attention_backend!r}"
        )
    return load_submission(
        _BLOCK_TABLE_TOPK_CONFIG_TRTLLM,
        verbose=verbose,
        cpp_source=_CPP_SOURCE_TRTLLM_BLOCK_TABLE_TOPK,
    ).topk


def dispatch_union(
    *,
    attention_backend: str = "trtllm",
    verbose: bool = True,
):
    """Pick the ``Union()`` (merge two (block_table, seqlens) pairs) callable.

    trtllm-only — the flashinfer/CSR backend's index buffers are flat and
    have no in-row dedup semantics to mirror.

    The returned callable's C signature is::

        topk(
            dense_seqlens,                 // [eff_bs] int32, tokens
            sparse_seqlens,                // [eff_bs] int32, tokens  (OUTPUT)
            dense_block_tables,            // [eff_bs, max_blocks_per_seq] int32
            block_tables_0,                // [eff_bs, max_blocks_per_seq] int32
            seqlens_0,                     // [eff_bs] int32, tokens
            block_tables_1,                // [eff_bs, max_blocks_per_seq] int32
            seqlens_1,                     // [eff_bs] int32, tokens
            sparse_block_tables,           // [eff_bs, max_blocks_per_seq] int32  (OUTPUT)
            eff_batch_size,
            max_blocks_per_seq,
            block_size,
        )
    """
    backend = (attention_backend or "trtllm").lower()
    if backend != "trtllm":
        raise NotImplementedError(
            f"dispatch_union: only the 'trtllm' backend is supported; "
            f"got {attention_backend!r}"
        )
    return load_submission(
        _UNION_CONFIG_TRTLLM,
        verbose=verbose,
        cpp_source=_CPP_SOURCE_TRTLLM_UNION,
    ).topk
