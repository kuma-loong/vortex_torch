"""Registry of ``(op_class, Schedule) -> codegen-function`` for CUDA.

Mirrors :mod:`vortex_torch.indexer.compiler.triton_impl.register` — the
schedule is part of the key so the same op class can have different
generators for ``Schedule.W`` (fused inline) and ``Schedule.S``
(standalone kernel).

Schedule.S ops are reused **verbatim** from the Triton backend:

  * Each Schedule.S codegen emits a Python wrapper body that contains
    a ``@triton.jit`` (or precompiled-kernel-dispatch) kernel launch.
    The body is plain Python that runs in the generated module — it
    has no coupling to the fused W kernel.
  * So the same wrapper works whether the W kernels are Triton or
    CUDA, and we save a parallel reimplementation of topK / softmax
    / normalize / conv1d / cross-row reduce.

Schedule.W ops require CUDA-side codegen because they're inlined into
the per-block fused kernel emitted by
:func:`cuda_impl.kernel_gen.generate_cuda_kernel`. None are registered
yet — they'll be added one at a time as the CUDA W-op codegens land.

:func:`cuda_impl.kernel_gen.generate_cuda_impl` injects
``import triton`` / ``import triton.language as tl`` into the
generated module's header lines so the reused Triton wrapper bodies
have the names they reference.
"""

from ....utils import Schedule

from ...output_func import topK, approxTopK
from ...scan import Softmax, Normalize, Conv1d
from ...reduce import Reduce
from ...elementwise import Elementwise
from ...elementwise_binary import Elementwise_Binary
from ...mask import MaskSlice
from ...transpose import Transpose
from ...save_load import Save, Load
from ...reshape import Reshape
from ...matmul import GeMM
from ...kron import Kron

from ..triton_impl.topk import (
    generate_topk_impl,
    generate_approx_topk_impl,
)
from ..triton_impl.softmax import generate_softmax_impl
from ..triton_impl.normalize import generate_normalize_impl
from ..triton_impl.conv1d import generate_conv1d_impl
# Reduce dispatches by schedule: ``dim in {1, 2}`` is fused (W), and
# ``dim == 0`` (cross-row, RAGGED → BATCHED) is standalone (S). Only
# the S form is reused from Triton here; the W form will get its own
# CUDA codegen.
from ..triton_impl.reduce import generate_reduce_dim0_impl

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
    # Schedule.S — standalone kernel launches; reused from Triton.
    (topK,       Schedule.S): generate_topk_impl,
    (approxTopK, Schedule.S): generate_approx_topk_impl,
    (Softmax,    Schedule.S): generate_softmax_impl,
    (Normalize,  Schedule.S): generate_normalize_impl,
    (Conv1d,     Schedule.S): generate_conv1d_impl,
    (Reduce,     Schedule.S): generate_reduce_dim0_impl,

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
    """Resolve an op instance to its CUDA codegen function.

    Exact ``(class, schedule)`` match wins; otherwise an MRO walk picks
    the closest registered ancestor at the same schedule. Raises
    ``NotImplementedError`` when nothing matches.
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
        f"No CUDA indexer codegen for op {cls.__name__} "
        f"with schedule {schedule}"
    )
