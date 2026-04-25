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
from ...output_func import topK
from ...scan import Softmax, Normalize, Conv1d
from ...transpose import Transpose
from ...save_load import Save, Load
from ...mask import MaskSlice

from .gemm import generate_gemm_impl
from .reduce import generate_reduce_impl, generate_reduce_dim0_impl
from .elementwise_binary import generate_elementwise_binary_impl
from .elementwise import generate_elementwise_impl
from .topk import generate_topk_impl
from .softmax import generate_softmax_impl
from .normalize import generate_normalize_impl
from .transpose import generate_transpose_impl
from .conv1d import generate_conv1d_impl
from .save_load import generate_save_impl, generate_load_impl
from .mask import generate_mask_slice_impl

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

    # Schedule.S — standalone, each op launches its own kernel.
    (topK,      Schedule.S): generate_topk_impl,
    (Softmax,   Schedule.S): generate_softmax_impl,
    (Normalize, Schedule.S): generate_normalize_impl,
    (Conv1d,    Schedule.S): generate_conv1d_impl,
    # Reduce dispatches by schedule too: ``dim in {1, 2}`` is fused (W),
    # ``dim == 0`` (cross-row, RAGGED → BATCHED) is standalone (S).
    (Reduce,    Schedule.S): generate_reduce_dim0_impl,
}


def get_impl_func(op):
    schedule = op.schedule
    for (op_type, sched), impl_func in IMPL_REGISTRY.items():
        if sched == schedule and issubclass(op.__class__, op_type):
            return impl_func
    raise NotImplementedError(
        f"No indexer codegen for op {op.__class__.__name__} "
        f"with schedule {schedule}"
    )