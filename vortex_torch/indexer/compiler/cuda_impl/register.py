"""Registry of ``(op_class, Schedule) -> codegen-function`` for CUDA.

Schedule.W only — these generators are inlined into the per-block fused
kernel emitted by :func:`cuda_impl.kernel_gen.generate_cuda_kernel`.

Schedule.S codegen lives under :mod:`indexer.compiler.custom_impl` and
is shared with the Triton backend. The dispatch in
:mod:`indexer.compiler.impl` routes Schedule.S subgraphs there directly,
so this registry never needs to know about S ops.
"""

from ....utils import Schedule

from ...reduce import Reduce
from ...elementwise import Elementwise
from ...elementwise_binary import Elementwise_Binary
from ...mask import MaskSlice
from ...transpose import Transpose
from ...save_load import Save, Load
from ...reshape import Reshape
from ...matmul import GeMM
from ...kron import Kron

from .elementwise import generate_elementwise_impl
from .elementwise_binary import generate_elementwise_binary_impl
from .mask import generate_mask_slice_impl
from .transpose import generate_transpose_impl
from .save_load import generate_save_impl, generate_load_impl
from .reshape import generate_reshape_impl
from .reduce import generate_reduce_impl
from .gemm import generate_gemm_impl
from .kron import generate_kron_impl


IMPL_REGISTRY = {
    # Schedule.W — fused into the per-block CUDA kernel. Registered
    # against base classes; ``get_impl_func``'s MRO walk catches
    # subclasses (Relu / Sigmoid / Silu / ... for Elementwise;
    # Add / Mul / Maximum / Minimum / Where* for Elementwise_Binary).
    (Elementwise,        Schedule.W): generate_elementwise_impl,
    (Elementwise_Binary, Schedule.W): generate_elementwise_binary_impl,
    (MaskSlice,          Schedule.W): generate_mask_slice_impl,
    (Transpose,          Schedule.W): generate_transpose_impl,
    (Save,               Schedule.W): generate_save_impl,
    (Load,               Schedule.W): generate_load_impl,
    (Reshape,            Schedule.W): generate_reshape_impl,
    (Reduce,             Schedule.W): generate_reduce_impl,
    (GeMM,               Schedule.W): generate_gemm_impl,
    (Kron,               Schedule.W): generate_kron_impl,
}


def get_impl_func(op):
    """Resolve an op instance to its CUDA Schedule.W codegen function.

    Exact ``(class, schedule)`` match wins; otherwise an MRO walk picks
    the closest registered ancestor at the same schedule. Raises
    ``NotImplementedError`` when nothing matches — Schedule.S ops never
    reach this registry (they're routed through
    :func:`indexer.compiler.custom_impl.get_impl_func`).
    """
    schedule = op.schedule
    cls = op.__class__

    exact = IMPL_REGISTRY.get((cls, schedule))
    if exact is not None:
        return exact

    for parent in cls.__mro__[1:]:
        impl_func = IMPL_REGISTRY.get((parent, schedule))
        if impl_func is not None:
            return impl_func

    raise NotImplementedError(
        f"No CUDA Schedule.W indexer codegen for op {cls.__name__} "
        f"with schedule {schedule}"
    )
