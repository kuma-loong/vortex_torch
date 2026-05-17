"""Union() (merge two ``(block_table, seqlens)`` pairs) — op-level
dispatch (trtllm-only today)."""
import importlib

_OP = "union"
_SUPPORTED = ("trtllm",)


def find(backend, **kwargs):
    backend = (backend or "trtllm").lower()
    if backend not in _SUPPORTED:
        raise NotImplementedError(
            f"custom_ops/{_OP}: backend {backend!r} not supported; "
            f"expected one of {list(_SUPPORTED)} "
            f"(Union() deduplicates 2D block_tables; only the trtllm "
            f"backend has that layout)"
        )
    mod = importlib.import_module(f"vortex_torch.custom_ops.{_OP}.{backend}.dispatch")
    return mod.find(**kwargs)
