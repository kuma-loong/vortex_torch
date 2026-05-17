"""softmax / trtllm / default — leaf dispatch."""
from .kernel import softmax_kernel
from vortex_torch.custom_ops._triton_launcher import make_launcher


def dispatch():
    """Return a plain-callable launcher (unified with CUDA leaves).

    Call convention: ``launch(*kernel_args, eff_batch_size)``.
    """
    return make_launcher(softmax_kernel)
