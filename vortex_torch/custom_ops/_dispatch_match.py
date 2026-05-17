"""Constraint-matched leaf picker for custom_ops backends.

Each leaf under ``<op>/<backend>/<bucket>/`` carries a ``meta.json``
declaring the kwarg shape it is applicable to. The backend-level
``dispatch.py`` calls :func:`pick_leaf` with the routing kwargs and gets
back the leaf's bucket name; it then imports the leaf's ``dispatch``.

This module is the *single* place that interprets ``meta.json``.
Adding a constraint operator (e.g. a new ``"between"``) or changing
the specificity tie-breaker happens here once, not in every backend.

──────────────────────────────────────────────────────────────────────
meta.json schema
──────────────────────────────────────────────────────────────────────

    {
        "name": "<bucket directory name>",         // required, must match dirname
        "description": "<one-line summary>",       // optional, agent-readable
        "constraints": {                            // optional, {} == catch-all
            "<kwarg_name>": <constraint-value>,
            ...
        },
        "priority": <int>,                          // optional, default 0
        "default": <bool>                           // optional, default False
    }

A ``<constraint-value>`` is either:

  * a JSON scalar — implicit equality (``"approx": false`` ⇔
    ``{"eq": false}``);
  * an object ``{"<op>": <value>}`` where ``<op>`` ∈ ``OP_REGISTRY``
    (``eq`` / ``ne`` / ``le`` / ``lt`` / ``ge`` / ``gt`` / ``in`` /
    ``nin``).

All constraints inside ``constraints`` are ANDed. A missing kwarg in the
dispatch call is treated as *not satisfying* the constraint (i.e. the
leaf is filtered out) — declare the leaf with no constraint on that
kwarg if you want it to apply regardless.

──────────────────────────────────────────────────────────────────────
Matching algorithm
──────────────────────────────────────────────────────────────────────

1. Filter to leaves whose constraints are *all* satisfied by ``kwargs``.
2. Among the surviving leaves, pick the most-specific one — defined as:
     a. higher ``len(constraints)`` wins (more specific);
     b. ties broken by higher ``priority``;
     c. remaining ties broken alphabetically by ``name`` (stable).
3. If no leaf survives step 1: pick the ``default: true`` leaf if one
   exists; otherwise raise ``LookupError``.

This guarantees that adding a new shape-specialised leaf (e.g.
``{"max_topk_val": {"le": 96}, "D0": {"eq": 128}}``) always beats a
sibling with the same ``max_topk_val`` constraint but no ``D0``
constraint — agents can land hyper-specific kernels without touching
``dispatch.py``.
"""
from __future__ import annotations

import json
import operator
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constraint operators
# ---------------------------------------------------------------------------

# Each operator is ``(lhs_from_kwargs, rhs_from_meta) -> bool``. ``lhs`` may
# be ``_MISSING`` when the kwarg was not supplied — we surface that to the
# operator so e.g. an ``"eq": null`` constraint can be satisfied by an
# absent kwarg if the leaf author opts in (current behaviour: missing
# always fails — keep it strict so agents don't accidentally match).
_MISSING = object()


def _eq(lhs, rhs):
    return lhs is not _MISSING and lhs == rhs


def _ne(lhs, rhs):
    return lhs is not _MISSING and lhs != rhs


def _ord(op: Callable[[Any, Any], bool]):
    def _check(lhs, rhs):
        if lhs is _MISSING or lhs is None:
            return False
        return op(lhs, rhs)
    return _check


def _in(lhs, rhs):
    if lhs is _MISSING:
        return False
    if not isinstance(rhs, (list, tuple, set)):
        raise ValueError(f"'in' constraint requires a list/tuple/set rhs; got {rhs!r}")
    return lhs in rhs


def _nin(lhs, rhs):
    if lhs is _MISSING:
        return False
    if not isinstance(rhs, (list, tuple, set)):
        raise ValueError(f"'nin' constraint requires a list/tuple/set rhs; got {rhs!r}")
    return lhs not in rhs


OP_REGISTRY: Dict[str, Callable[[Any, Any], bool]] = {
    "eq": _eq,
    "ne": _ne,
    "le": _ord(operator.le),
    "lt": _ord(operator.lt),
    "ge": _ord(operator.ge),
    "gt": _ord(operator.gt),
    "in": _in,
    "nin": _nin,
}


# ---------------------------------------------------------------------------
# Leaf discovery + matching
# ---------------------------------------------------------------------------

class _LeafMeta:
    __slots__ = ("name", "description", "constraints", "priority", "default", "path")

    def __init__(self, path: Path, raw: dict):
        self.path = path
        self.name = raw["name"]
        self.description = raw.get("description", "")
        self.constraints: Dict[str, Tuple[str, Any]] = {}
        for key, spec in (raw.get("constraints") or {}).items():
            if isinstance(spec, dict):
                if len(spec) != 1:
                    raise ValueError(
                        f"{path}: constraint on {key!r} must have exactly one "
                        f"operator, got {spec!r}"
                    )
                op_name, rhs = next(iter(spec.items()))
                if op_name not in OP_REGISTRY:
                    raise ValueError(
                        f"{path}: unknown constraint operator {op_name!r}; "
                        f"expected one of {sorted(OP_REGISTRY)}"
                    )
                self.constraints[key] = (op_name, rhs)
            else:
                # scalar: implicit equality
                self.constraints[key] = ("eq", spec)
        self.priority = int(raw.get("priority", 0))
        self.default = bool(raw.get("default", False))

    def matches(self, kwargs: dict) -> bool:
        for key, (op_name, rhs) in self.constraints.items():
            lhs = kwargs.get(key, _MISSING)
            if not OP_REGISTRY[op_name](lhs, rhs):
                return False
        return True

def _load_leaves(backend_dir: Path) -> List[_LeafMeta]:
    leaves: List[_LeafMeta] = []
    for child in sorted(backend_dir.iterdir()):
        if not child.is_dir() or child.name.startswith("_"):
            continue
        meta_path = child / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{child}: leaf is missing meta.json — every bucket under "
                f"<op>/<backend>/ must declare its applicability "
                f"(see custom_ops/AGENTS.md for the schema)."
            )
        raw = json.loads(meta_path.read_text())
        if raw.get("name") != child.name:
            raise ValueError(
                f"{meta_path}: meta.json 'name' {raw.get('name')!r} does not "
                f"match the bucket directory name {child.name!r}."
            )
        leaves.append(_LeafMeta(child, raw))
    return leaves


def pick_leaf(backend_dir: str | Path, kwargs: dict) -> str:
    """Pick the most-specific leaf under ``backend_dir`` matching ``kwargs``.

    Returns the bucket *name* (directory name); the caller is responsible
    for importing ``<backend_dir>/<name>/dispatch.py``.

    Raises :class:`LookupError` when no leaf matches and no ``default``
    leaf exists.
    """
    backend_dir = Path(backend_dir)
    leaves = _load_leaves(backend_dir)
    if not leaves:
        raise LookupError(f"{backend_dir}: no leaves found")

    matches = [m for m in leaves if m.matches(kwargs)]
    if matches:
        # Sort key encodes preference: ``(-len, -priority, name)`` ascending
        # ⇒ pick the leaf with most constraints first, then highest priority,
        # final tie-break by name (asc, stable).
        matches.sort(key=lambda m: (-len(m.constraints), -m.priority, m.name))
        return matches[0].name

    defaults = [m for m in leaves if m.default]
    if defaults:
        defaults.sort(key=lambda m: m.priority, reverse=True)
        return defaults[0].name

    raise LookupError(
        f"{backend_dir}: no leaf matches kwargs {kwargs!r} and no leaf is "
        f"marked 'default: true' in meta.json"
    )


__all__ = ["pick_leaf"]
