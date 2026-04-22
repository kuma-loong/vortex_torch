from ...matmul import GeMM
from ...reduce import Reduce
from ...elementwise_binary import Elementwise_Binary
from ...elementwise import Elementwise
from ...output_func import topK
from ...scan import Softmax, Normalize, Conv1d
from ...transpose import Transpose
from ...save_load import Save, Load

from .gemm import generate_gemm_impl
from .reduce import generate_reduce_impl
from .elementwise_binary import generate_elementwise_binary_impl
from .elementwise import generate_elementwise_impl
from .topk import generate_topk_impl
from .softmax import generate_softmax_impl
from .normalize import generate_normalize_impl
from .transpose import generate_transpose_impl
from .conv1d import generate_conv1d_impl
from .save_load import generate_save_impl, generate_load_impl
IMPL_REGISTRY = {
    GeMM: generate_gemm_impl,
    Reduce: generate_reduce_impl,
    Elementwise_Binary: generate_elementwise_binary_impl,
    Elementwise: generate_elementwise_impl,
    topK: generate_topk_impl,
    Softmax: generate_softmax_impl,
    Normalize: generate_normalize_impl,
    Transpose: generate_transpose_impl,
    Conv1d: generate_conv1d_impl,
    Save: generate_save_impl,
    Load: generate_load_impl,
}

def get_impl_func(op) -> str:
    for op_type, impl_func in IMPL_REGISTRY.items():
        if issubclass(op.__class__, op_type):
            return impl_func
    raise NotImplementedError(f"No implementation function found for op type {op.__class__}")