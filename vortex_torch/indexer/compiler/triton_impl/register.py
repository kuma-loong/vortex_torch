"""Registry of ``(op_class, Schedule) -> codegen-function`` for Triton.

Keying on the schedule (in addition to the op class) lets the same op
have different generators depending on how it's scheduled. For example
a :class:`Reduce` over the workload-fused axes (``dim in (1, 2)``) is
``Schedule.W`` and inlines into the per-block kernel, while a future
``Reduce`` over ``dim == 0`` would be ``Schedule.S`` (standalone) and
needs its own kernel-launching generator. Same op class, two
generators — only the schedule disambiguates them.
"""

from ....utils import Schedule

from ...matmul import GeMM
from ...reduce import Reduce
from ...elementwise_binary import Elementwise_Binary
from ...elementwise import Elementwise
from ...output_func import topK, approxTopK
from ...scan import Softmax, Normalize, Conv1d
from ...transpose import Transpose
from ...save_load import Save, Load
from ...mask import MaskSlice
from ...kron import Kron
from ...reshape import Reshape

from .gemm import generate_gemm_impl
from .reduce import generate_reduce_impl, generate_reduce_dim0_impl
from .elementwise_binary import generate_elementwise_binary_impl
from .elementwise import generate_elementwise_impl
from .topk import generate_topk_impl, generate_approx_topk_impl
from .softmax import generate_softmax_impl
from .normalize import generate_normalize_impl
from .transpose import generate_transpose_impl
from .conv1d import generate_conv1d_impl
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

    # Schedule.S — standalone, each op launches its own kernel.
    (topK,       Schedule.S): generate_topk_impl,
    (approxTopK, Schedule.S): generate_approx_topk_impl,
    (Softmax,   Schedule.S): generate_softmax_impl,
    (Normalize, Schedule.S): generate_normalize_impl,
    (Conv1d,    Schedule.S): generate_conv1d_impl,
    # Reduce dispatches by schedule too: ``dim in {1, 2}`` is fused (W),
    # ``dim == 0`` (cross-row, RAGGED → BATCHED) is standalone (S).
    (Reduce,    Schedule.S): generate_reduce_dim0_impl,
}


def get_impl_func(op):
    """Resolve an op instance to its codegen function.

    Exact class match wins (so a subclass like ``approxTopK`` gets its
    own generator instead of inheriting ``topK``'s). Falls back to an
    MRO walk: the closest registered ancestor at the same schedule
    provides the generator.
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
        f"No indexer codegen for op {cls.__name__} "
        f"with schedule {schedule}"
    )