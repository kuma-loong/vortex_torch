"""flashinfer/CSR CUB-sort fallback — leaf dispatch (used when
``max_topk_val`` is unknown or exceeds every radix winner's cap)."""
from pathlib import Path

from vortex_torch.custom_ops._jit import load_submission
from .._abi import CPP_SOURCE

_CONFIG = Path(__file__).resolve().parent / "config.json"


def dispatch(verbose: bool = True):
    """Return the JIT-compiled C ``topk`` callable for this leaf."""
    return load_submission(_CONFIG, cpp_source=CPP_SOURCE, verbose=verbose).topk
