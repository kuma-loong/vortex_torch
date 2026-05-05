import torch
from typing import Dict

from vortex_torch.flow import vFlow, register
from vortex_torch.abs import ContextBase
from vortex_torch.indexer import (
    topK, Mean, GeMM, Add, Abs, Sum, Add_Mul,
)
from vortex_torch.cache import (
    Mean as CMean,
)


@register("claude_sonnet_4_6_innovate_0_id3_cls")
class L1DistanceScoring(vFlow):
    """
    Score pages by negative L1 distance between the query mean and the
    key centroid.  Pages closest to the query in L1 norm receive the
    highest score, inverting the usual inner-product assumption.

    Novelty: L1 distance (rather than dot product) as the relevance
    metric; exploits Abs + Sum(dim=2) to compute ‖q − centroid‖₁.
    §16.3 — inversion of the dot-product direction-alignment assumption.
    """

    def __init__(self):
        super().__init__()
        # indexer-side
        self.q_mean = Mean(dim=1)
        self.diff = Add(alpha=1.0, beta=-1.0)     # q_mean - centroid
        self.abs_op = Abs(alpha=0.0, beta=1.0)
        self.l1 = Sum(dim=2)                       # sum over features → [S, 1, 1]
        self.negate = Add_Mul(alpha=0.0, beta=-1.0)
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
        q_mean = self.q_mean(q, ctx=ctx)                            # [1, 1, D]
        diff = self.diff(q_mean, cache["centroids"], ctx=ctx)       # [S, 1, D]
        abs_diff = self.abs_op(diff, ctx=ctx)                       # [S, 1, D]
        l1_dist = self.l1(abs_diff, ctx=ctx)                        # [S, 1, 1]
        score = self.negate(l1_dist, ctx=ctx)                       # [S, 1, 1] (negative distance)
        self.output_func(score, o, ctx=ctx)
