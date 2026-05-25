# registry.py
from typing import Dict, Type, Union
from .flow import vFlow
from .flow_mla import vFlowMLA

# Global registry: key -> subclass of vFlow (MHA) or vFlowMLA (latent attention)
_FlowBase = (vFlow, vFlowMLA)
_REGISTRY: Dict[str, Type[Union[vFlow, vFlowMLA]]] = {}

class RegistryError(Exception):
    """Errors related to class registration and lookup."""
    ...

def register(name: str):
    """
    Decorator used by users to register their vFlow subclasses.
    Example:
        @register("cls_a")
        class MyFlow(vFlow): ...
    """
    def deco(cls):
        if not issubclass(cls, _FlowBase):
            raise RegistryError(f"{cls.__name__} must inherit from vFlow or vFlowMLA")
        if name in _REGISTRY:
            raise RegistryError(f"Registration name '{name}' already exists")
        _REGISTRY[name] = cls
        return cls
    return deco

def get(name: str) -> Type[vFlow]:
    """Return the registered class for a given name, or raise if not found."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise RegistryError(f"Registration name '{name}' not found")

def has(name: str) -> bool:
    """Check whether a name is registered."""
    return name in _REGISTRY

def list_keys():
    """List all registered names."""
    return list(_REGISTRY.keys())
