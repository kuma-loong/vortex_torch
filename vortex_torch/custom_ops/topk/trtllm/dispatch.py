"""topk / trtllm — backend-level dispatch (constraint-matched).

Every bucket under this directory declares its applicability in a
``meta.json`` file; :func:`vortex_torch.custom_ops._dispatch_match.pick_leaf`
selects the most-specific match. Adding a new bucket is a pure drop-in
(``mkdir <bucket>``; drop ``meta.json`` + leaf files); no edit to this
file. See ``custom_ops/AGENTS.md`` for the meta.json schema.
"""
import importlib
from pathlib import Path

from vortex_torch.custom_ops._dispatch_match import pick_leaf

_BACKEND_DIR = Path(__file__).resolve().parent
_MODULE_PREFIX = "vortex_torch.custom_ops.topk.trtllm"


def find(**kwargs):
    bucket = pick_leaf(_BACKEND_DIR, kwargs)
    leaf = importlib.import_module(f"{_MODULE_PREFIX}.{bucket}.dispatch")
    return leaf.dispatch
