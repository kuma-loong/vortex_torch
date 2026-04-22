"""Shared helpers for cache Triton kernels.

Centralizes the FP8/BF16 reinterpret used by every wrapper before
launching a Triton kernel, so each impl file does not have to repeat it.
"""

import torch

from ...utils import QuantizationType


def _quant_view(x: torch.Tensor, quantization_type: QuantizationType) -> torch.Tensor:
    """Reinterpret an FP8 tensor as ``uint8`` so it can be passed through Triton.

    BF16 inputs are returned untouched. The kernel side is expected to
    bitcast the ``uint8`` block back to ``tl.float8e5`` / ``tl.float8e4nv``
    based on the matching ``QUANT_TYPE`` constexpr.
    """
    if quantization_type != QuantizationType.BF16:
        return x.view(torch.uint8)
    return x
