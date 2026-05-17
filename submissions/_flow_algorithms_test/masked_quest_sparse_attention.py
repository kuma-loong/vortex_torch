import torch
from typing import Dict

from vortex_torch.flow import vFlow, register
from vortex_torch.indexer import (
    topK, approxTopK, GeMV, Softmax, Max, Sum, GeMM,
    Maximum, Multiply, Add, L2Norm, Save, Load, Mean, MaskSlice, Kron,
)
from vortex_torch.cache import (
    Mean as CMean, Max as CMax, Min as CMin, L2Norm as CL2Norm,
    Fill as CFill, MaxInterleave as CMaxInterleave, MinInterleave as CMinInterleave,
)
from vortex_torch.abs import ContextBase


@register("masked_quest_sparse_attention_sub")
class MaskedQuestSparseAttention(vFlow):
    r"""
    QUEST-style sparse attention with a feature-axis :class:`MaskSlice`.

    Identical to :class:`GQAQuestSparseAttention` in its routing logic,
    but before summing over the feature dimension it multiplies the
    envelope tensor by a position-dependent mask built with
    :class:`MaskSlice`:

    .. math::

        m[\ldots, d] =
        \begin{cases}
            0, & d < \text{MASK\_END}, \\
            1, & d \ge \text{MASK\_END}.
        \end{cases}

    This suppresses the leading ``MASK_END`` feature planes of the
    QUEST envelope score — a cheap, position-only way to exclude
    low-signal channels. Because :class:`MaskSlice` is a pure
    position-based writer (its output does not depend on the input
    values), no extra state is threaded through ``ctx``.

    The mask is applied along ``dim=2`` (the head / feature dim ``D``),
    so ``MASK_END`` must be :math:`\le D` at runtime. The default
    ``MASK_END = 8`` is safe for all head dims in the verification
    sweep (``D \in \{32, 64, 128\}``).
    """

    MASK_END = 8  # mask [0, MASK_END) features; safe for D in {32, 64, 128}

    def __init__(self):
        super().__init__()

        # Indexer-side ops
        self.mul_max = Multiply()
        self.mul_min = Multiply()
        self.maximum_op = Maximum()
        # Position-only mask on the feature axis: α=0 on [0, MASK_END), β=1 elsewhere.
        self.feature_mask = MaskSlice(
            start=0, end=self.MASK_END, dim=2, alpha=0.0, beta=1.0
        )
        self.mul_mask = Multiply()
        self.sum = Sum(dim=2)
        self.max_op = Max(dim=1)
        self.output_func = topK()

        # Cache-side ops
        self.reduction_max = CMax(dim=1)
        self.reduction_min = CMin(dim=1)

    def forward_indexer(
        self,
        q: torch.Tensor,
        o: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        ctx: ContextBase,
    ):
        s_max = self.mul_max(q, cache["max"], ctx=ctx)      # [S, H_q, D]
        s_min = self.mul_min(q, cache["min"], ctx=ctx)      # [S, H_q, D]
        s = self.maximum_op(s_max, s_min, ctx=ctx)          # [S, H_q, D]
        mask = self.feature_mask(s, ctx=ctx)                # [S, H_q, D]
        masked_s = self.mul_mask(s, mask, ctx=ctx)          # [S, H_q, D]
        score = self.sum(masked_s, ctx=ctx)                 # [S, H_q, 1]
        aggr_score = self.max_op(score, ctx=ctx)            # [S, 1, 1]
        self.output_func(aggr_score, o, ctx=ctx)

    def forward_cache(
        self,
        cache: Dict[str, torch.Tensor],
        loc: torch.Tensor,
        ctx: ContextBase,
    ):
        self.reduction_max(cache["k"], cache["max"], loc=loc, ctx=ctx)
        self.reduction_min(cache["k"], cache["min"], loc=loc, ctx=ctx)

    def create_cache(self, block_size: int, head_dim: int):
        return {
            "max": (1, head_dim),
            "min": (1, head_dim),
        }


