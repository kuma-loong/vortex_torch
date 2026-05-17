"""reduce_dim0 / flashinfer / l2norm — leaf dispatch."""
from .._kernels import reduce_dim0_l2norm_kernel
from vortex_torch.custom_ops._triton_launcher import make_launcher


def dispatch():
    """Return a plain-callable launcher (unified with CUDA leaves).

    Call convention: ``launch(*kernel_args, eff_batch_size)``.
    """
    return make_launcher(reduce_dim0_l2norm_kernel)
