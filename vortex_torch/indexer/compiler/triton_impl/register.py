"""Registry of ``(op_class, Schedule) -> codegen-function`` for Triton.

Schedule.W only — these generators inline into the per-block fused
kernel emitted by :func:`triton_impl.kernel_gen.generate_triton_impl`.

Schedule.S codegen for the indexer compiler lives under
:mod:`indexer.compiler.custom_impl` and is shared between the Triton
and CUDA Schedule.W backends. ``Reduce`` here only registers the
``dim in {1, 2}`` form; the ``dim == 0`` cross-row form is Schedule.S
and lives in ``custom_impl``.
"""

from ....utils import Schedule

from ...matmul import GeMM
from ...reduce import Reduce
from ...elementwise_binary import Elementwise_Binary
from ...elementwise import Elementwise
from ...transpose import Transpose
from ...save_load import Save, Load
from ...mask import MaskSlice
from ...kron import Kron
from ...reshape import Reshape

from .gemm import generate_gemm_impl
from .reduce import generate_reduce_impl
from .elementwise_binary import generate_elementwise_binary_impl
from .elementwise import generate_elementwise_impl
from .transpose import generate_transpose_impl
from .save_load import generate_save_impl, generate_load_impl
from .mask import generate_mask_slice_impl
from .kron import generate_kron_impl
from .reshape import generate_reshape_impl

IMPL_REGISTRY = {
    # Schedule.W — fused inline into the per-workload kernel.
    (GeMM,               Schedule.W): generate_gemm_impl,
    (Reduce,             Schedule.W): generate_reduce_impl,
    (Elementwise_Binary, Schedule.W): generate_elementwise_binary_impl,
    (Elementwise,        Schedule.W): generate_elementwise_impl,
    (Transpose,          Schedule.W): generate_transpose_impl,
    (Save,               Schedule.W): generate_save_impl,
    (Load,               Schedule.W): generate_load_impl,
    (MaskSlice,          Schedule.W): generate_mask_slice_impl,
    (Kron,               Schedule.W): generate_kron_impl,
    (Reshape,            Schedule.W): generate_reshape_impl,
}


def get_impl_func(op):
    """Resolve an op instance to its Schedule.W codegen function.

    Exact class match wins; falls back to an MRO walk for subclasses.
    Schedule.S ops never reach this registry — they're handled by
    :func:`indexer.compiler.custom_impl.get_impl_func`.
    """
    schedule = op.schedule
    cls = op.__class__

    # 1) Exact match: most specific entry, never shadowed by a superclass.
    exact = IMPL_REGISTRY.get((cls, schedule))
    if exact is not None:
        return exact

    # 2) MRO walk: closest registered ancestor at this schedule.
    for parent in cls.__mro__[1:]:
        impl_func = IMPL_REGISTRY.get((parent, schedule))
        if impl_func is not None:
            return impl_func

    raise NotImplementedError(
        f"No Triton Schedule.W indexer codegen for op {cls.__name__} "
        f"with schedule {schedule}"
    )
