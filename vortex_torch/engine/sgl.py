import json
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Union

import sglang as sgl


DEFAULT_SCHEDULE_POLICY = r"""
const int static_kv_budget = topk_val + block_reserved_bos + block_reserved_eos;
const int dynamic_kv_budget = int(cached_block_len * topk_ratio);
return max(static_kv_budget, dynamic_kv_budget);
"""


MODEL_PATH = "/data/zhuoming/hf_models/Qwen3-4B"


def get_engine(
    *,
    vortex_block_size: int = 16,
    vortex_topk_val: int = 29,
    vortex_topk_ratio: float | None = None,
    vortex_block_reserved_bos: int = 1,
    vortex_block_reserved_eos: int = 2,
    vortex_workload_chunk_size: int = 32,
    vortex_layers_skip: list[int] | None = None,
    vortex_module_name: str = "example_block_sparse_attention_cls",
    vortex_module_path: str = "submissions/example_block_sparse_attention.py",
    vortex_schedule_policy: str | None = None,
    kv_cache_dtype: str = "auto",
    **kwargs,
):
    layers_skip = list(range(1)) if vortex_layers_skip is None else vortex_layers_skip
    policy = DEFAULT_SCHEDULE_POLICY if vortex_schedule_policy is None else vortex_schedule_policy

    engine_kwargs = dict(
        model_path=MODEL_PATH,
        page_size=vortex_block_size,
        vortex_block_size=vortex_block_size,
        vortex_topk_val=vortex_topk_val,
        vortex_max_seq_lens=20480,
        vortex_block_reserved_bos=vortex_block_reserved_bos,
        vortex_block_reserved_eos=vortex_block_reserved_eos,
        vortex_workload_chunk_size=vortex_workload_chunk_size,
        vortex_layers_skip=layers_skip,
        vortex_module_name=vortex_module_name,
        vortex_module_path=vortex_module_path,
        vortex_schedule_policy=policy,
        vortex_dtype="bfloat16",
        vortex_compilation_cache_dir="~/.vortex_compilation_cache",
        enable_vortex_sparsity=True,
        kv_cache_dtype=kv_cache_dtype,
        attention_backend="flashinfer",
        mem_fraction_static=0.8,
        disable_cuda_graph=False,
        disable_overlap_schedule=True,
        tp_size=1,
    )
    if vortex_topk_ratio is not None:
        engine_kwargs["vortex_topk_ratio"] = vortex_topk_ratio

    engine_kwargs.update(kwargs)
    return sgl.Engine(**engine_kwargs)


def get_engine_from_json(config_path: str | Path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Engine config at {config_path} must be a JSON object")
    return get_engine(**config)


# ---------------------------------------------------------------------------
# Pre-flight checks for an engine JSON
# ---------------------------------------------------------------------------

class EngineConfigError(ValueError):
    """Raised when an engine JSON fails any pre-flight validation step."""


def _coerce_int(x: Any, label: str) -> int:
    """Coerce a JSON-loaded number to an ``int``.

    Accepts ints and other numerics that round-trip exactly (e.g. ``2.0``
    is treated as ``2``); rejects ``bool``, strings, ``None``, and floats
    that would silently truncate (e.g. ``1.5``).
    """
    if isinstance(x, bool) or x is None or isinstance(x, str):
        raise EngineConfigError(f"{label} must be an int, got {x!r}")
    try:
        i = int(x)
    except (TypeError, ValueError):
        raise EngineConfigError(f"{label} must be an int, got {x!r}")
    if i != x:
        raise EngineConfigError(f"{label} must be a whole number, got {x!r}")
    return i


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _resolve_module_path(json_path: Path, raw: str) -> Path:
    """``vortex_module_path`` may be absolute, CWD-relative, or relative to
    the JSON file. Try each in that order and return the first that exists."""
    p = Path(raw)
    candidates = [p] if p.is_absolute() else [Path.cwd() / p, json_path.parent / p, p]
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise EngineConfigError(
        f"vortex_module_path={raw!r} not found "
        f"(tried: {[str(c) for c in candidates]})"
    )


def _check_registers_class(module_path: Path, module_name: str) -> None:
    """Heuristic source-level check that the file declares
    ``@register("<module_name>")`` somewhere — catches typos before we run
    the file."""
    src = module_path.read_text(encoding="utf-8")
    pattern = rf"@register\(\s*[\"']{re.escape(module_name)}[\"']\s*\)"
    if re.search(pattern, src) is None:
        raise EngineConfigError(
            f"{module_path} does not contain @register({module_name!r})"
        )


_INDEXER_SAVE_PATTERN = re.compile(r"\bSave\s*\(")


def _flow_uses_indexer_save(module_path: Path) -> bool:
    """Return True iff the submission file contains an indexer-side ``Save(``
    call (the persistent state pattern paired with ``Load`` and
    ``CFill(0.0)``). Cache side has no ``Save`` op, so this textual match
    is unambiguous."""
    try:
        src = module_path.read_text(encoding="utf-8")
    except OSError:
        return False
    return bool(_INDEXER_SAVE_PATTERN.search(src))


def _check_disable_radix_cache(module_path: Path, config: Dict[str, Any]) -> None:
    """If the flow uses ``Save(...)`` in the indexer, the engine config
    must set ``disable_radix_cache: true``. Otherwise sglang's prefix
    radix cache will share the Save'd per-request persistent state
    across requests with matching prompt prefixes — silently corrupting
    Save/Load values across decode batches."""
    if not _flow_uses_indexer_save(module_path):
        return
    if config.get("disable_radix_cache", False) is not True:
        raise EngineConfigError(
            f"{module_path.name} uses `Save(...)` in the indexer "
            f"(persistent per-request state via Save/Load), so the "
            f"engine config must set `\"disable_radix_cache\": true`. "
            f"Without it, sglang's radix cache will share Save'd state "
            f"across requests with matching prompt prefixes and corrupt "
            f"the values. Add `\"disable_radix_cache\": true` to the JSON."
        )


def _check_compilable(module_path: Path, module_name: str) -> None:
    """Load the user file, build the registered vFlow, and run a tiny
    compile sweep. Uses CPU-side metadata only — no CUDA required."""
    # Imports are local: avoid pulling vortex_torch.flow at module-import
    # time of vortex_torch.engine, since that drags in the whole compiler
    # surface for callers that only want ``get_engine``.
    from ..flow.loader import build_vflow
    from ..flow.verify import verify_flow_compilable
    from ..flow.registry import has as _is_registered

    # Re-executing the file would re-run its top-level ``@register`` and
    # trip the "already exists" guard. Once the name is in the registry we
    # can fetch the class directly — step 7 already verified the file is
    # the source of truth for the @register declaration.
    user_file = None if _is_registered(module_name) else str(module_path)

    try:
        flow = build_vflow(module_name, user_file=user_file)
    except Exception as e:
        raise EngineConfigError(
            f"failed to build vFlow {module_name!r} from {module_path}: {e}"
        ) from e

    # Sweep over a small grid of GQA shapes — enough to catch shape /
    # dispatch issues that only show up at certain ``(G, num_kv_heads)``
    # combinations. ``verify_flow_compilable`` only sweeps ``G``
    # internally, so we wrap kvh in an outer loop.
    with tempfile.TemporaryDirectory(prefix="vortex_check_") as cache_dir:
        for kvh in (1, 2, 4):
            report = verify_flow_compilable(
                flow,
                B=2, num_kv_heads=kvh,
                G_values=(1, 2, 4), D_values=(64,),
                block_sizes=(16,), page_block_ratios=(1,),
                pages_per_workload_values=(1,),
                max_num_pages_per_request=16,
                max_new_tokens_per_batch=64,
                cache_dir=cache_dir,
                verify_indexer=True, verify_cache=True,
            )
            if not report.ok:
                first = report.failed[0]
                raise EngineConfigError(
                    f"vFlow {module_name!r} failed to compile "
                    f"(num_kv_heads={kvh}, {first.phase}, cfg {first.cfg.label()}):\n"
                    f"{first.traceback}"
                )


def check_engine_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Validate an engine JSON config end-to-end.

    Steps:
      1. JSON path exists and parses to a dict.
      2. ``vortex_block_size`` is a positive power of 2.
      3. ``vortex_workload_chunk_size`` is a positive power of 2.
      4. ``vortex_block_reserved_bos`` and ``vortex_block_reserved_eos``
         are ints ``>= 1``.
      5. ``vortex_layers_skip`` is missing/None/empty or a list of ints.
      6. ``vortex_module_path`` resolves to an existing file.
      7. That file declares ``@register("<vortex_module_name>")``.
      8. The registered vFlow compiles on a 1-config smoke sweep.
      9. If the flow uses ``Save(...)`` in the indexer, the JSON sets
         ``"disable_radix_cache": true`` (otherwise sglang's prefix
         cache would share per-request persistent state across
         requests with matching prompt prefixes).

    Returns the parsed config dict on success. Raises
    :class:`EngineConfigError` with a descriptive message on the first
    failing step.
    """
    # 1. JSON path exists
    json_path = Path(config_path).expanduser()
    if not json_path.is_file():
        raise EngineConfigError(f"config not found: {json_path}")
    try:
        config = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise EngineConfigError(f"{json_path} is not valid JSON: {e}") from e
    if not isinstance(config, dict):
        raise EngineConfigError(f"{json_path} must be a JSON object")

    # 2. vortex_block_size = 2**n
    block_size = _coerce_int(config.get("vortex_block_size"), "vortex_block_size")
    if not _is_pow2(block_size):
        raise EngineConfigError(
            f"vortex_block_size must be a positive power of 2, got {block_size!r}"
        )

    # 3. vortex_workload_chunk_size = 2**k
    chunk = _coerce_int(
        config.get("vortex_workload_chunk_size"), "vortex_workload_chunk_size"
    )
    if not _is_pow2(chunk):
        raise EngineConfigError(
            f"vortex_workload_chunk_size must be a positive power of 2, got {chunk!r}"
        )

    # 4. vortex_block_reserved_{bos,eos} >= 1, ints
    for key in ("vortex_block_reserved_bos", "vortex_block_reserved_eos"):
        v = _coerce_int(config.get(key), key)
        if v < 1:
            raise EngineConfigError(f"{key} must be an int >= 1, got {v!r}")

    # 5. vortex_layers_skip absent/None/empty or list of ints
    layers_skip = config.get("vortex_layers_skip", None)
    if layers_skip not in (None, []):
        if not isinstance(layers_skip, list):
            raise EngineConfigError(
                f"vortex_layers_skip must be empty or a list of ints, got {layers_skip!r}"
            )
        # Validate each element via _coerce_int so 0.0 / "0" / etc. give a
        # clean per-element error.
        for idx, x in enumerate(layers_skip):
            _coerce_int(x, f"vortex_layers_skip[{idx}]")

    # 6. vortex_module_path exists
    raw_module_path = config.get("vortex_module_path")
    if not isinstance(raw_module_path, str) or not raw_module_path:
        raise EngineConfigError(
            f"vortex_module_path must be a non-empty string, got {raw_module_path!r}"
        )
    module_path = _resolve_module_path(json_path, raw_module_path)

    # 7. file declares @register("<name>")
    module_name = config.get("vortex_module_name")
    if not isinstance(module_name, str) or not module_name:
        raise EngineConfigError(
            f"vortex_module_name must be a non-empty string, got {module_name!r}"
        )
    _check_registers_class(module_path, module_name)

    # 8. registered class compiles
    _check_compilable(module_path, module_name)

    # 9. Save() in indexer ⇒ disable_radix_cache must be true
    _check_disable_radix_cache(module_path, config)

    return config
