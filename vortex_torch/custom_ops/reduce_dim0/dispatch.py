"""reduce_dim0 — op-level dispatch (route on ``backend``)."""
import importlib

_OP = "reduce_dim0"
_SUPPORTED = ("flashinfer", "trtllm")


def find(backend, **kwargs):
    backend = (backend or "flashinfer").lower()
    if backend not in _SUPPORTED:
        raise NotImplementedError(
            f"custom_ops/{_OP}: backend {backend!r} not supported; "
            f"expected one of {list(_SUPPORTED)}"
        )
    mod = importlib.import_module(f"vortex_torch.custom_ops.{_OP}.{backend}.dispatch")
    return mod.find(**kwargs)
