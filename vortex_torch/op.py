from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional, Literal
from .context_base import ContextBase
from .utils import Mode
# Mode can only be "profile" or "execute"

class vOp(ABC):
    """Base class for defining virtual operators that support profiling and execution modes."""

    def __init__(self) -> None:
        super().__init__()

    # ------------------------------------------------------------
    # Abstract method: profile
    # Called during the profiling / preparation phase.
    # Typical use cases:
    #   - Allocate persistent output buffers.
    #   - Compute static shapes.
    #   - Collect performance statistics.
    # Subclasses must implement this method.
    # ------------------------------------------------------------
    @abstractmethod
    def profile(self, *args: Any, ctx: ContextBase = None, **kwargs: Any) -> Any:
        raise NotImplementedError

    # ------------------------------------------------------------
    # Abstract method: execute
    # Called during the normal execution phase.
    # Implements the actual operator logic.
    # Subclasses must implement this method.
    # ------------------------------------------------------------
    @abstractmethod
    def execute(self, *args: Any, ctx: ContextBase = None, **kwargs: Any) -> Any:
        raise NotImplementedError

    # ------------------------------------------------------------
    # Callable entry point.
    # `ctx.mode` is a keyword-only argument:
    #   - ctx.mode="profile" → call self.profile(*args, **kwargs)
    #   - ctx.mode="execute" or None → call self.execute(*args, **kwargs)
    # Any other mode will raise ValueError.
    # ------------------------------------------------------------
    def __call__(self, *args: Any, ctx: ContextBase = None, **kwargs: Any) -> Any:
        if ctx.mode is None or ctx.mode == Mode.execute:
            return self.execute(*args, ctx=ctx, **kwargs)
        if ctx.mode == Mode.profile:
            return self.profile(*args, ctx=ctx, **kwargs)
        raise ValueError(f"Unknown mode: {ctx.mode!r}, expected 'profile' or 'execute'")