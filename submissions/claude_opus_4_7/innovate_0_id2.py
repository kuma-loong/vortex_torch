"""id2 — Multi-resolution Min-band routing.

Novelty (op-set thread): use `CMeanInterleave(dim=1, k=group)` to emit
G mini-centroids per block (sub-block means) and aggregate the per-band
score via `Min(dim=1)`. A page is selected only if it aligns with q in
*every* sub-block — strict consistency across temporally-local key
clusters. Inverts the dual-band Max/Mean aggregation in §14.4 / Prism;
Min is what nobody tries because it's pessimistic.
"""

import torch
from typing import Dict

from vortex_torch.flow import vFlow, register
from vortex_torch.indexer import topK, Mean, Multiply, Sum, Min
from vortex_torch.cache import MeanInterleave as CMeanInterleave
from vortex_torch.abs import ContextBase


# block_size = 16 in the engine JSON; group = 4 → G = 16/4 = 4 mini-centroids.
_GROUP = 4


@register("claude_opus_4_7_innovate_0_id2_cls")
class Id2Cls(vFlow):
    def __init__(self):
        super().__init__()
        self.q_mean = Mean(dim=1)
        self.mul = Multiply()
        self.sum_band = Sum(dim=2)
        self.min_band = Min(dim=1)
        self.output_func = topK()

        self.k_mini = CMeanInterleave(dim=1, k=_GROUP)

    def create_cache(self, block_size: int, head_dim: int):
        # block_size // group mini-centroids per block
        return {"mini_centroids": (block_size // _GROUP, head_dim)}

    def forward_cache(self, cache, loc, ctx: ContextBase):
        self.k_mini(cache["k"], cache["mini_centroids"], loc=loc, ctx=ctx)

    def forward_indexer(self, q, o, cache, ctx: ContextBase):
        qm = self.q_mean(q, ctx=ctx)                                 # [1, 1, D]
        s = self.mul(qm, cache["mini_centroids"], ctx=ctx)            # [S, G, D]
        score_per_band = self.sum_band(s, ctx=ctx)                    # [S, G, 1]
        aggr = self.min_band(score_per_band, ctx=ctx)                 # [S, 1, 1]
        self.output_func(aggr, o, ctx=ctx)
