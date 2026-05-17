# `custom_ops/` — agent contract

This tree is the **single** place where Schedule.S indexer kernels live.
The compiler emits `find('<op>', '<backend>', **routing)(*args)` calls;
this module resolves them through a 4-level dispatch tree, returning a
plain Python callable regardless of impl (Triton or CUDA).

## Goal

Let agents land new, possibly hyper-specialised kernels (e.g. tuned
for one specific `D0` × `dtype` × `topk_val` combination) **without
ever editing dispatch code**. Drop a directory, drop a `meta.json`,
the matcher picks it up.

## Dispatch tree

```
custom_ops/
    __init__.py             # level-0: find(op_name, backend, **kwargs)
    _jit.py                 # CUDA JIT helper (load_inline cache)
    _triton_launcher.py     # Triton @triton.jit → plain-callable wrapper
    _dispatch_match.py      # meta.json constraint matcher
    AGENTS.md               # this file
    <op_name>/              # softmax / normalize / conv_1d / reduce_dim0 /
                            # topk_output / topk / union
        SPEC.md             # op semantics + calling signature + I/O contract
        dispatch.py         # level-1: route on backend
        <backend>/          # flashinfer / trtllm
            _abi.py         # C ABI string (CUDA leaves only)
            dispatch.py     # level-2: constraint-matched bucket picker
                            #          (identical 5-liner across backends)
            <bucket>/       # level-3 — the actual implementation
                meta.json   # applicability constraints + priority
                dispatch.py # leaf entry point — returns a plain callable
                kernel.py   # Triton leaves: @triton.jit body
                kernel.cu   # CUDA leaves: source
                config.json # CUDA leaves: substitutions + cflags
```

## Unified callable contract

`find(...)()` returns a **plain Python callable**. Triton kernels are
wrapped via `_triton_launcher.make_launcher`; CUDA `load_inline`'d
callables are returned directly. Callers invoke either as
`op(*positional_args)` — no `[grid]` indexing, no `num_warps` kwargs.

Per-op argument order lives in each op's `SPEC.md`. Convention for
Triton-wrapped callables: the **trailing positional arg is
`eff_batch_size`** (the 1D grid size); preceding args map 1-to-1 onto
the underlying kernel's positional parameters.

## `meta.json` schema

Every leaf carries one:

```json
{
    "name": "k_96",
    "description": "Adaptive-approx-skip radix winner for max_topk_val <= 96.",
    "constraints": {
        "max_topk_val": {"le": 96},
        "D0":           {"eq": 128},
        "dtype":        {"in": ["bfloat16"]}
    },
    "priority": 100,
    "default":  false
}
```

  * `name` — must match the bucket directory name.
  * `description` — one line, agent-readable.
  * `constraints` — key = routing kwarg, value = scalar (implicit `eq`)
    or `{"<op>": <rhs>}` with `<op>` ∈
    `eq` / `ne` / `le` / `lt` / `ge` / `gt` / `in` / `nin`. All
    constraints are ANDed; a missing kwarg never satisfies a
    constraint (i.e. omit the constraint if you want the leaf to
    apply regardless).
  * `priority` — int, used to break specificity ties. Higher wins.
  * `default` — if `true`, this leaf is selected when no leaf's
    `constraints` are satisfied. Exactly zero or one per backend.

### Matching algorithm

1. Filter to leaves whose constraints are all satisfied by the
   dispatch kwargs.
2. Among survivors: pick the most-specific, defined as
   `(more constraints) > (higher priority) > (name asc)`.
3. If no survivor: fall back to the `default: true` leaf, else
   `LookupError`.

Defined in `_dispatch_match.py`.

## Adding a new bucket

```
mkdir custom_ops/<op>/<backend>/<my_bucket>
```

Drop:

  * `meta.json` — declare constraints that select your leaf. The
    matcher picks the most-specific leaf when several apply, so the
    more keys you constrain on, the tighter your specialisation. Don't
    forget to bump `priority` above the catch-all when you want to
    actually win.
  * `dispatch.py` — implements `def dispatch(**leaf_kwargs):` and
    returns a plain Python callable (see `_triton_launcher.make_launcher`
    for Triton, `_jit.load_submission` for CUDA).
  * `kernel.py` (Triton) or `kernel.cu` + `config.json` (CUDA) — the
    implementation. CUDA leaves additionally import the per-backend
    `_abi.py` for the ABI string.

No edit to `<backend>/dispatch.py` is required — `pick_leaf` scans
the directory.

## Adding a new backend (under an existing op)

```
mkdir custom_ops/<op>/<new_backend>
```

Drop a `dispatch.py` identical to the existing siblings (5-line
constraint-matched wrapper), plus an `_abi.py` if CUDA. Then add
`<new_backend>` to the `_SUPPORTED` tuple in `custom_ops/<op>/dispatch.py`.

## Adding a new op

```
mkdir custom_ops/<my_op>
```

Drop:

  * `SPEC.md` — semantics + calling signature + I/O contract +
    constraint kwargs (one of the existing ops is a good template).
  * `dispatch.py` — op-level router, identical pattern to existing
    siblings (route on backend, list of supported backends).
  * `<backend>/dispatch.py`, `<backend>/<bucket>/...` — as above.

Register the op in the compiler's launcher emitter as needed (see
`vortex_torch/indexer/compiler/triton_impl/`).

## Where to look first

  * **What does this op do?** → `<op>/SPEC.md`
  * **How do I call it?** → `<op>/SPEC.md` § "Calling signature"
  * **Which leaves exist + when does each apply?** → `ls <op>/<backend>/`
    and read each `meta.json`
  * **How does dispatch pick a leaf?** → `_dispatch_match.py`
  * **How is the kernel wrapped?** → `_triton_launcher.py` (Triton)
    or `_jit.py` (CUDA)
