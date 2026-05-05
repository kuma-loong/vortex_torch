"""id8 — Sigmoid-saturated per-feature contributions before Sum.

Novelty (op-set thread): every score function in the literature reduces
along the feature axis with a *linear* sum of `q_d * c_d` products.
We insert a `Sigmoid` *between* the elementwise product and the
`Sum(dim=2)`, saturating each feature's contribution to (0, 1) before
summing. Effect: a page with one huge feature alignment can no longer
dominate via a single channel — every feature contributes at most `~1`,
so the score becomes a count-like measure of "how many features
align". Mid-pipeline saturation is unexplored (a Sigmoid on the final
[S,1,1] would be a topK no-op; only mid-pipeline placement matters).
"""

import torch
from typing import Dict

from vortex_torch.flow import vFlow, register
from vortex_torch.indexer import topK, Mean, Multiply, Sigmoid, Sum
from vortex_torch.cache import Mean as CMean
from vortex_torch.abs import ContextBase


@register("claude_opus_4_7_innovate_0_id8_cls")
class Id8Cls(vFlow):
    def __init__(self):
        super().__init__()
        self.q_mean = Mean(dim=1)
        self.product = Multiply()
        self.saturate = Sigmoid(alpha=0.0, beta=-1.0)   # standard logistic σ(x)
        self.sum_d = Sum(dim=2)
        self.mean_h = Mean(dim=1)
        self.output_func = topK()

        self.k_mean = CMean(dim=1)

    def create_cache(self, block_size: int, head_dim: int):
        return {"centroids": (1, head_dim)}

    def forward_cache(self, cache, loc, ctx: ContextBase):
        self.k_mean(cache["k"], cache["centroids"], loc=loc, ctx=ctx)

    def forward_indexer(self, q, o, cache, ctx: ContextBase):
        qm = self.q_mean(q, ctx=ctx)                                 # [1, 1, D]
        s = self.product(qm, cache["centroids"], ctx=ctx)            # [S, 1, D]
        sat = self.saturate(s, ctx=ctx)                               # [S, 1, D] in (0, 1)
        score_d = self.sum_d(sat, ctx=ctx)                            # [S, 1, 1]
        score = self.mean_h(score_d, ctx=ctx)                         # [S, 1, 1]
        self.output_func(score, o, ctx=ctx)
