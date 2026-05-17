"""TopK(k) (block-table + seqlens output) — op-level dispatch
(trtllm-only today)."""
import importlib

_OP = "topk"
_SUPPORTED = ("trtllm",)


def find(backend, **kwargs):
    backend = (backend or "trtllm").lower()
    if backend not in _SUPPORTED:
        raise NotImplementedError(
            f"custom_ops/{_OP}: backend {backend!r} not supported; "
            f"expected one of {list(_SUPPORTED)} "
            f"(TopK(k) writes 2D block_tables; only the trtllm backend "
            f"consumes that layout)"
        )
    mod = importlib.import_module(f"vortex_torch.custom_ops.{_OP}.{backend}.dispatch")
    return mod.find(**kwargs)
