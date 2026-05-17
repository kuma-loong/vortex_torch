# `_template/` — copy-and-edit scaffold for a new `topk` / `trtllm` bucket

Drop this whole directory under a new bucket name, then edit:

  1. **`meta.json`**
     * Set `name` to your new bucket directory name (must match).
     * Set `description` to one line.
     * Replace the `constraints` placeholder with real applicability
       keys — `max_topk_val` / `D0` / `dtype` / `block_size` / ...
       See `custom_ops/topk/SPEC.md` for the kwargs this op routes on.
     * Bump `priority` if you want to beat siblings at the same
       specificity level.

  2. **`kernel.cu` / `config.json`** (CUDA) or **`kernel.py`** (Triton)
     * The starting body is a clone of the current default leaf — a
       working kernel you can iteratively modify. Specialise for the
       constraints you declared in `meta.json`.

  3. **`dispatch.py`**
     * Usually no edit needed. The boilerplate wires `kernel.cu`
       via the standard JIT helper (CUDA: `_jit.load_submission`,
       Triton: `_triton_launcher.make_launcher`).

Quick correctness check:

```python
from vortex_torch.custom_ops import find
from vortex_torch.custom_ops.topk._reference import reference_trtllm

fn = find("topk", "trtllm", <your routing kwargs>)()  # picks your leaf
out_kernel = ...
out_ref    = ...
fn(*args)                              # writes out_kernel
reference_trtllm(*ref_args)         # writes out_ref
assert (out_kernel.float() - out_ref.float()).abs().max() < 1e-2
```

The matcher in `custom_ops/_dispatch_match.py` skips directories
whose name starts with `_`, so `_template/` is never picked at
runtime — it lives here for copying only.
