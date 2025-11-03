from enum import Enum

class _UNSET_T:
    def __repr__(self) -> str: return "UNSET"
UNSET = _UNSET_T()


class Mode(Enum):
    profile = 0
    execute = 1
    