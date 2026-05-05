import torch
from typing import Dict

from vortex_torch.flow import vFlow, register
from vortex_torch.abs import ContextBase
from vortex_torch.indexer import (
    topK, Mean, Multiply, Silu, Sum,
)
from vortex_torch.cache import (
    Mean as CMean,
)


@register("claude_sonnet_4_6_innovate_0_id7_cls")
class SiluActivatedCentroidScore(vFlow):
    """
    Apply per-feature SiLU gating to the element-wise product of the
    query and key centroid before summing to a scalar page score.

    SiLU(x) = x·σ(x) suppresses features where q and centroid are
    small or anti-correlated, amplifying features where both are large
    and same-sign.  Contrast with the plain dot product, which lets
    negative contributions cancel positive ones.

    Novelty: SiLU activation on per-feature (q·centroid) components
    before Sum(dim=2); no paper applies a non-linear gate to individual
    feature contributions of the attention score.
    Op-set derived (Silu op exploited for per-feature soft co-activation).
    """

    def __init__(self):
        super().__init__()
        # indexer-side
        self.q_mean = Mean(dim=1)
        self.prod = Multiply()
        self.silu = Silu(alpha=0.0, beta=-1.0)   # standard SiLU: x·σ(x)
        self.sum_d = Sum(dim=2)
        self.output_func = topK()

        # cache-side
        self.k_mean = CMean(dim=1)

    def create_cache(self, block_size: int, head_dim: int):
        return {
            "centroids": (1, head_dim),
        }

    def forward_cache(self, cache, loc, ctx):
        self.k_mean(cache["k"], cache["centroids"], loc=loc, ctx=ctx)

    def forward_indexer(self, q, o, cache, ctx):
        q_mean = self.q_mean(q, ctx=ctx)                        # [1, 1, D]
        prod = self.prod(q_mean, cache["centroids"], ctx=ctx)   # [S, 1, D]
        gated = self.silu(prod, ctx=ctx)                        # [S, 1, D]
        score = self.sum_d(gated, ctx=ctx)                      # [S, 1, 1]
        self.output_func(score, o, ctx=ctx)
