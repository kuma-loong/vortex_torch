import torch
from typing import Dict

from vortex_torch.flow import vFlow, register
from vortex_torch.abs import ContextBase
from vortex_torch.indexer import (
    topK, Mean, GeMM, Abs,
)
from vortex_torch.cache import (
    Mean as CMean, Abs as CAbs,
)


@register("claude_sonnet_4_6_innovate_0_id4_cls")
class AbsKeyCentroid(vFlow):
    """
    Build per-block centroids from |k| (absolute key values) and score by
    |q_mean| · E[|k|].  This measures feature-magnitude co-activation
    rather than directional alignment; a page where large-magnitude query
    features meet large-magnitude key features scores highest regardless
    of sign.

    Novelty: magnitude co-activation score via CAbs → CMean in cache and
    Abs → GeMM in indexer — no paper uses unsigned similarity for routing.
    §16.3 — inversion of the directional dot-product assumption.
    """

    def __init__(self):
        super().__init__()
        # indexer-side
        self.q_abs = Abs(alpha=0.0, beta=1.0)
        self.q_abs_mean = Mean(dim=1)
        self.gemm = GeMM()
        self.output_func = topK()

        # cache-side
        self.k_abs = CAbs(alpha=0.0, beta=1.0)
        self.abs_mean = CMean(dim=1)

    def create_cache(self, block_size: int, head_dim: int):
        return {
            "abs_centroids": (1, head_dim),
        }

    def forward_cache(self, cache, loc, ctx):
        abs_k = self.k_abs(cache["k"], None, loc=loc, ctx=ctx)                  # [1, block_size, D]
        self.abs_mean(abs_k, cache["abs_centroids"], loc=loc, ctx=ctx)           # [1, 1, D]

    def forward_indexer(self, q, o, cache, ctx):
        q_abs = self.q_abs(q, ctx=ctx)                                   # [1, H_q, D]
        q_abs_mean = self.q_abs_mean(q_abs, ctx=ctx)                     # [1, 1, D]
        score = self.gemm(q_abs_mean, cache["abs_centroids"], ctx=ctx)   # [S, 1, 1]
        self.output_func(score, o, ctx=ctx)
