"""reduce_dim0 / flashinfer / sum — leaf dispatch."""
from .._kernels import reduce_dim0_sum_kernel
from vortex_torch.custom_ops._triton_launcher import make_launcher


def dispatch():
    """Return a plain-callable launcher (unified with CUDA leaves).

    Call convention: ``launch(*kernel_args, eff_batch_size)``.
    """
    return make_launcher(reduce_dim0_sum_kernel)
