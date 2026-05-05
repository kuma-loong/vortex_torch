"""id1 — Min-over-heads QUEST (inversion of Max aggregation).

Novelty (§16.3 inversion): every QUEST-style flow in this framework
aggregates per-page-per-head scores with `Max(dim=1)` — "any aligned
head wins". We invert to `Min(dim=1)`: a page is highly scored only if
*every* query head finds it relevant. Stricter consensus across heads,
hypothesised to be more robust on aggregation tasks where MagicPIG
warns the top-k recipe is fragile.
"""

import torch
from typing import Dict

from vortex_torch.flow import vFlow, register
from vortex_torch.indexer import topK, Multiply, Maximum, Sum, Min
from vortex_torch.cache import Max as CMax, Min as CMin
from vortex_torch.abs import ContextBase


@register("claude_opus_4_7_innovate_0_id1_cls")
class Id1Cls(vFlow):
    def __init__(self):
        super().__init__()
        self.mul_max = Multiply()
        self.mul_min = Multiply()
        self.maximum_op = Maximum()
        self.sum = Sum(dim=2)
        self.min_op = Min(dim=1)            # ← Min instead of Max over heads
        self.output_func = topK()

        self.cmax = CMax(dim=1)
        self.cmin = CMin(dim=1)

    def create_cache(self, block_size: int, head_dim: int):
        return {"max": (1, head_dim), "min": (1, head_dim)}

    def forward_cache(self, cache, loc, ctx: ContextBase):
        self.cmax(cache["k"], cache["max"], loc=loc, ctx=ctx)
        self.cmin(cache["k"], cache["min"], loc=loc, ctx=ctx)

    def forward_indexer(self, q, o, cache, ctx: ContextBase):
        s_max = self.mul_max(q, cache["max"], ctx=ctx)               # [S, H_q, D]
        s_min = self.mul_min(q, cache["min"], ctx=ctx)               # [S, H_q, D]
        s = self.maximum_op(s_max, s_min, ctx=ctx)                    # [S, H_q, D]
        score = self.sum(s, ctx=ctx)                                  # [S, H_q, 1]
        aggr = self.min_op(score, ctx=ctx)                            # [S, 1, 1]
        self.output_func(aggr, o, ctx=ctx)
