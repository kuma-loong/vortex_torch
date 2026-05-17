# `normalize` — trtllm default

Triton-codegen op (no standalone `.cu` file). The kernel string is
emitted at compile time by the codegen function in
[`vortex_torch/indexer/compiler/triton_impl/normalize.py`](../../../../vortex_torch/indexer/compiler/triton_impl/normalize.py), which calls into
`compiler.triton_impl.backend.get_backend(ctx)` to pick the
backend-specific snippet (per-row arg name, base-address arithmetic,
extra constexpr args).

For now `custom_ops.find("normalize", "trtllm")` raises `NotImplementedError`
and the codegen still goes through `compiler/triton_impl/`. When this
op grows runtime-selectable variants (e.g. per-shape kernel sources or
different scoring strategies) drop `kernel.py` / `kernel.cu` files
alongside this README, add a level-2 dispatch case in
`custom_ops/__init__.py::find`, and migrate the kernel_gen call site.
