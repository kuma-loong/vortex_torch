"""Backend dispatch for cache compiler.

Mirrors :mod:`vortex_torch.indexer.compiler.impl`. Each backend exposes
a single ``generate_<backend>_impl(sub_graph, sub_graph_id, ctx) -> str``
function that emits the per-subgraph implementation as Python source.
"""

AVAILABLE_IMPL_BACKENDS = {}

from .triton_impl import generate_triton_impl
AVAILABLE_IMPL_BACKENDS["triton"] = generate_triton_impl
