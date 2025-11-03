from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, List, Union
import torch
from .utils import UNSET, Mode

class ContextBase(ABC):
    """
    Abstract base for context objects with ONLY two allowed attributes:
      - mode: str        # operating mode, e.g. "profile"/"execute"
      - created: bool    # whether the context has been populated
    All behaviors are abstract and must be implemented by subclasses.
    """
    __slots__ = ("name", "mode", "_created")

    @property
    def created(self) -> bool:
        return self._created
    
    # --- abstract lifecycle ---
    @abstractmethod
    def create(self, *args: Any, **kwargs: Any) -> "ContextBase":
        """Populate the context (idempotency/overwrite rules are up to the subclass)."""
        raise NotImplementedError

    
    def profile(self) -> None:
        
        self.mode = Mode.profile
    
    def execute(self) -> None:
        
        self.mode = Mode.execute

    # Utilities
    def assert_created(self) -> None:
        if not self._created:
            raise RuntimeError("Context is not created yet.")

    def missing(self) -> list[str]:
        return [n for n in self.__slots__ if n != "_created" and getattr(self, n) is UNSET]
    
    def _tensor_nbytes(self, t: torch.Tensor) -> int:

            return int(t.element_size() * t.nelement())

    def add_aux_memory(self, obj: Union[int, torch.Tensor]) -> int:
        """
        Accumulate auxiliary memory.
        - If obj is an int: treated as bytes.
        - If obj is a torch.Tensor: converts to bytes and adds.
        Returns the added bytes.
        NOTE: This is a simple accumulator. Calling multiple times on tensors that share
              the same storage (or the same tensor) will add multiple times.
        """
        if isinstance(obj, int):
            nbytes = obj
        elif isinstance(obj, torch.Tensor):
            nbytes = self._tensor_nbytes(obj)
        else:
            raise TypeError("add_aux_memory expects an int (bytes) or a torch.Tensor")

        if nbytes < 0:
            raise ValueError("nbytes must be non-negative")

        self._aux_total_bytes += nbytes
        return nbytes


    def clear_aux_memory(self) -> None:
        """Reset the total auxiliary memory to zero."""
        self._aux_total_bytes = 0
        
    
    def summary(self) -> None:
        """Print fields; tensor fields show shape/dtype/device, and append memory totals incl. auxiliary."""
        def _fmt_bytes(n: int) -> str:
            units = ("B", "KB", "MB", "GB", "TB", "PB")
            f = float(n)
            for u in units:
                if f < 1024 or u == units[-1]:
                    return f"{f:.2f} {u}"
                f /= 1024.0

        print("=== Context Summary ===")

        # internal tensors memory (dedup by storage)
        seen: set[tuple[str, int]] = set()
        by_device_internal: dict[str, int] = {}
        internal_cpu = internal_gpu = 0

        def _acc_tensor(t: torch.Tensor) -> int:
            dev = str(t.device)
            try:
                st = t.untyped_storage()
                nbytes = int(st.nbytes())
                key = (dev, int(st.data_ptr()))
            except Exception:
                nbytes = int(t.element_size() * t.nelement())
                try:
                    key = (dev, int(t.data_ptr()))
                except Exception:
                    key = (dev, id(t))
            if key in seen:
                return 0
            seen.add(key)
            by_device_internal[dev] = by_device_internal.get(dev, 0) + nbytes
            return nbytes

        for name in self.__slots__:
            if name.startswith("_"):
                continue
            val = getattr(self, name)
            if isinstance(val, torch.Tensor):
                print(f"{name:<25}: Tensor(shape={tuple(val.shape)}, dtype={val.dtype}, device={val.device})")
                n = _acc_tensor(val)
                if val.device.type == "cpu":
                    internal_cpu += n
                else:
                    internal_gpu += n
            else:
                print(f"{name:<25}: {val}")

        # auxiliary memory (single counter, no device split)
        aux_total = int(self._aux_total_bytes)

        # totals
        print("--- Memory Totals ---")
        print(f"Internal CPU : {internal_cpu} bytes ({_fmt_bytes(internal_cpu)})")
        print(f"Internal GPU : {internal_gpu} bytes ({_fmt_bytes(internal_gpu)})")
        print(f"Auxiliary ALL: {aux_total} bytes ({_fmt_bytes(aux_total)})")
        grand_total = internal_cpu + internal_gpu + aux_total
        print(f"GRAND TOTAL  : {grand_total} bytes ({_fmt_bytes(grand_total)})")


        print("====================================")