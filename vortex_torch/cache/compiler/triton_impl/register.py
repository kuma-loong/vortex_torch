"""Registry of ``(op_class, Schedule) -> codegen-function`` for Triton.

Mirror of the indexer-side registry. Keying on ``Schedule`` lets a
single op class have separate codegens for fused (``W``) vs standalone
(``S``) execution — important once ops like :class:`Reduce` need both
(e.g. reduce-along-block-axis fuses inline; reduce-across-blocks would
need to be standalone).
"""

from ....utils import Schedule

from ...elementwise import Elementwise
from ...elementwise_binary import Elementwise_Binary
from ...matmul import GeMM
from ...reduce import Reduce
from ...fill import Fill
from ...mask import MaskSlice

from .elementwise import generate_elementwise_impl
from .elementwise_binary import generate_elementwise_binary_impl
from .gemm import generate_gemm_impl
from .reduce import generate_reduce_impl
from .fill import generate_fill_impl
from .mask import generate_mask_slice_impl

IMPL_REGISTRY = {
    # Schedule.W — fused into the per-block cache kernel.
    (Elementwise,        Schedule.W): generate_elementwise_impl,
    (Elementwise_Binary, Schedule.W): generate_elementwise_binary_impl,
    (GeMM,               Schedule.W): generate_gemm_impl,
    (Reduce,             Schedule.W): generate_reduce_impl,
    (Fill,               Schedule.W): generate_fill_impl,
    (MaskSlice,          Schedule.W): generate_mask_slice_impl,
}


def get_impl_func(op):
    schedule = op.schedule
    for (op_type, sched), impl_func in IMPL_REGISTRY.items():
        if sched == schedule and issubclass(op.__class__, op_type):
            return impl_func
    raise NotImplementedError(
        f"No cache codegen for op {op.__class__.__name__} "
        f"with schedule {schedule}"
    )
