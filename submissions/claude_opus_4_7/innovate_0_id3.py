"""id3 — Anti-QUEST (lower-bound envelope via Minimum).

Novelty (§16.3 inversion): QUEST takes `Maximum(q*max, q*min)` — an
elementwise *upper* bound on q·k similarity (optimism). We swap to
`Minimum(q*max, q*min)` — the *lower* bound (pessimism). A page is
selected only if its conservatively-bounded similarity is high. Tests
the inverse hypothesis to QUEST: do tight pessimistic bounds pick a
better page set than loose optimistic ones?
"""

import torch
from typing import Dict

from vortex_torch.flow import vFlow, register
from vortex_torch.indexer import topK, Multiply, Minimum, Sum, Max
from vortex_torch.cache import Max as CMax, Min as CMin
from vortex_torch.abs import ContextBase


@register("claude_opus_4_7_innovate_0_id3_cls")
class Id3Cls(vFlow):
    def __init__(self):
        super().__init__()
        self.mul_max = Multiply()
        self.mul_min = Multiply()
        self.minimum_op = Minimum()         # ← Minimum instead of Maximum
        self.sum = Sum(dim=2)
        self.max_op = Max(dim=1)
        self.output_func = topK()

        self.cmax = CMax(dim=1)
        self.cmin = CMin(dim=1)

    def create_cache(self, block_size: int, head_dim: int):
        return {"max": (1, head_dim), "min": (1, head_dim)}

    def forward_cache(self, cache, loc, ctx: ContextBase):
        self.cmax(cache["k"], cache["max"], loc=loc, ctx=ctx)
        self.cmin(cache["k"], cache["min"], loc=loc, ctx=ctx)

    def forward_indexer(self, q, o, cache, ctx: ContextBase):
        s_max = self.mul_max(q, cache["max"], ctx=ctx)
        s_min = self.mul_min(q, cache["min"], ctx=ctx)
        s = self.minimum_op(s_max, s_min, ctx=ctx)                    # lower bound
        score = self.sum(s, ctx=ctx)
        aggr = self.max_op(score, ctx=ctx)
        self.output_func(aggr, o, ctx=ctx)
