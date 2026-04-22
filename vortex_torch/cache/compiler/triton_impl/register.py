"""Registry of cache-op-class -> codegen-function for the Triton backend."""

from ...elementwise import Elementwise
from ...elementwise_binary import Elementwise_Binary
from ...matmul import GeMM
from ...reduce import Reduce
from ...fill import Fill

from .elementwise import generate_elementwise_impl
from .elementwise_binary import generate_elementwise_binary_impl
from .gemm import generate_gemm_impl
from .reduce import generate_reduce_impl
from .fill import generate_fill_impl

IMPL_REGISTRY = {
    Elementwise: generate_elementwise_impl,
    Elementwise_Binary: generate_elementwise_binary_impl,
    GeMM: generate_gemm_impl,
    Reduce: generate_reduce_impl,
    Fill: generate_fill_impl,
}


def get_impl_func(op):
    for op_type, impl_func in IMPL_REGISTRY.items():
        if issubclass(op.__class__, op_type):
            return impl_func
    raise NotImplementedError(
        f"No cache codegen implementation found for op type {op.__class__}"
    )
