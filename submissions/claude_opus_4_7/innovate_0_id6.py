"""id6 — Anti-redundancy decayed selection counter (Save/Load).

Novelty (op-set thread + §16.3 inversion): every Save/Load flow
*positively* rewards historically-relevant pages (heavy hitters). We
build the inverse: a per-page decayed accumulator of *prior relevance*,
and *subtract* it from the current step's score. Pages that have been
hot for many steps in a row get pushed down the ranking, encouraging
the topK budget to spread to *fresh* pages each step. The hypothesis is
that revisiting the same hot pages decode-step after decode-step burns
budget that could be exploring complementary information.
"""

import torch
from typing import Dict

from vortex_torch.flow import vFlow, register
from vortex_torch.indexer import topK, Mean, GeMM, Load, Save, Add
from vortex_torch.cache import Mean as CMean, Fill as CFill
from vortex_torch.abs import ContextBase


@register("claude_opus_4_7_innovate_0_id6_cls")
class Id6Cls(vFlow):
    DECAY = 0.85
    PENALTY = 0.4

    def __init__(self):
        super().__init__()
        self.q_mean = Mean(dim=1)
        self.gemm = GeMM()
        self.load_count = Load()
        self.save_count = Save()
        self.update_count = Add(alpha=self.DECAY, beta=1.0)
        self.penalize = Add(alpha=1.0, beta=-self.PENALTY)
        self.output_func = topK()

        self.k_mean = CMean(dim=1)
        self.init_count = CFill(alpha=0.0)

    def create_cache(self, block_size: int, head_dim: int):
        return {
            "centroids": (1, head_dim),
            "counter":   (1, 1),
        }

    def forward_cache(self, cache, loc, ctx: ContextBase):
        self.k_mean(cache["k"], cache["centroids"], loc=loc, ctx=ctx)
        self.init_count(cache["counter"], loc=loc, ctx=ctx)

    def forward_indexer(self, q, o, cache, ctx: ContextBase):
        qm = self.q_mean(q, ctx=ctx)                                 # [1, 1, D]
        current = self.gemm(qm, cache["centroids"], ctx=ctx)         # [S, 1, 1]
        prev_count = self.load_count(cache["counter"], ctx=ctx)      # [S, 1, 1]
        # decayed accumulation of *current* score (proxy for "did I get picked?")
        new_count = self.update_count(prev_count, current, ctx=ctx)
        self.save_count(new_count, cache["counter"], ctx=ctx)
        # subtract the *previous* counter (per page) from current score
        score = self.penalize(current, prev_count, ctx=ctx)
        self.output_func(score, o, ctx=ctx)
