"""softmax / flashinfer / default — leaf dispatch."""
from .kernel import softmax_kernel
from vortex_torch.custom_ops._triton_launcher import make_launcher


def dispatch():
    """Return a plain-callable launcher (unified with CUDA leaves).

    Call convention: ``launch(*kernel_args, eff_batch_size)`` — the
    trailing arg is the 1D grid; everything before it maps 1-to-1 onto
    :func:`softmax_kernel`'s positional parameters.
    """
    return make_launcher(softmax_kernel)
