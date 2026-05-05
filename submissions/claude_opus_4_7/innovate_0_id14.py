"""id14 — Envelope-width-driven score (q · spread).

Novelty (op-set thread + §16.4): combine the QUEST envelope's `(max,
min)` cache fields not as an upper-bound, but as a *spread* signal.
Mathematically `q · (max - min) = q·max - q·min` — we compute the
right-hand side because the indexer dispatch supports `(BATCHED,
PAGED)` Multiply but not `(PAGED, PAGED)` Add. The score is the per-q
energy weighted by where each page's keys vary the most. Routes pages
by "where is q's energy concentrated *in features that this page
actually exercises*". No paper here treats envelope width as a routing
modulator.
"""

import torch
from typing import Dict

from vortex_torch.flow import vFlow, register
from vortex_torch.indexer import topK, Multiply, Add, Sum, Mean
from vortex_torch.cache import Max as CMax, Min as CMin
from vortex_torch.abs import ContextBase


@register("claude_opus_4_7_innovate_0_id14_cls")
class Id14Cls(vFlow):
    def __init__(self):
        super().__init__()
        self.mul_max = Multiply()
        self.mul_min = Multiply()
        self.diff = Add(alpha=1.0, beta=-1.0)         # q*max - q*min
        self.sum_d = Sum(dim=2)
        self.mean_h = Mean(dim=1)
        self.output_func = topK()

        self.cmax = CMax(dim=1)
        self.cmin = CMin(dim=1)

    def create_cache(self, block_size: int, head_dim: int):
        return {"max": (1, head_dim), "min": (1, head_dim)}

    def forward_cache(self, cache, loc, ctx: ContextBase):
        self.cmax(cache["k"], cache["max"], loc=loc, ctx=ctx)
        self.cmin(cache["k"], cache["min"], loc=loc, ctx=ctx)

    def forward_indexer(self, q, o, cache, ctx: ContextBase):
        # (BATCHED, PAGED) → RAGGED for both products.
        s_max = self.mul_max(q, cache["max"], ctx=ctx)               # [S, H_q, D]
        s_min = self.mul_min(q, cache["min"], ctx=ctx)               # [S, H_q, D]
        # Subtract — both operands are now RAGGED.
        weighted = self.diff(s_max, s_min, ctx=ctx)                  # q*(max - min)
        score_d = self.sum_d(weighted, ctx=ctx)                       # [S, H_q, 1]
        score = self.mean_h(score_d, ctx=ctx)                         # [S, 1, 1]
        self.output_func(score, o, ctx=ctx)
