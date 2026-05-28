"""vortex_torch engine package. Lazy (PEP 562) re-export of the sgl entry
points so importing a light submodule (e.g. ``engine.sgl.config``) does not
pull in ``api``/sglang until a name below is actually used."""

__all__ = ["get_engine", "get_engine_from_json"]


def __getattr__(name):
    if name in __all__:
        from vortex_torch.engine import sgl
        return getattr(sgl, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
