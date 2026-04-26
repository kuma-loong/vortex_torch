"""GQA-QUEST sparse attention — identical to `gqa_quest_sparse_attention`
but with `approxTopK(tolerate_ratio=0.25)` swapped in for `topK()`.

Used for inspecting the difference in the emitted indexer kernel.
"""

import torch
from typing import Dict

from vortex_torch.flow import vFlow, register
from vortex_torch.indexer import (
    approxTopK, Multiply, Maximum, Sum, Max,
)
from vortex_torch.cache import Max as CMax, Min as CMin
from vortex_torch.abs import ContextBase


@register("gqa_quest_approx_cls")
class GQAQuestApprox(vFlow):
    def __init__(self):
        super().__init__()

        # Indexer-side ops — identical to GQAQuestSparseAttention …
        self.mul_max = Multiply()
        self.mul_min = Multiply()
        self.maximum_op = Maximum()
        self.sum = Sum(dim=2)
        self.max_op = Max(dim=1)
        # … except for the terminal op:
        self.output_func = approxTopK(tolerate_ratio=0.25)

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
        s_max = self.mul_max(q, cache["max"], ctx=ctx)
        s_min = self.mul_min(q, cache["min"], ctx=ctx)
        s = self.maximum_op(s_max, s_min, ctx=ctx)
        score = self.sum(s, ctx=ctx)
        aggr_score = self.max_op(score, ctx=ctx)
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
