import torch
from typing import Dict

from vortex_torch.flow import vFlow, register
from vortex_torch.abs import ContextBase
from vortex_torch.indexer import (
    topK, Mean, GeMM,
)
from vortex_torch.cache import (
    Mean as CMean,
)


@register("claude_sonnet_4_6_innovate_0_id16_cls")
class VCentroidOnlyScoring(vFlow):
    """
    Score pages purely by the alignment of the query with the VALUE
    centroid, ignoring the key centroid entirely.

    The hypothesis: the value centroid represents "what this page will
    contribute to the output"; a page whose value direction aligns with
    the query will produce output tokens more relevant to the current
    decoding step, even if its key direction is weakly aligned.

    This is the purest test of the §16.4 question "what signal in
    cache['v'] does no paper here use?" — all existing flows route on keys.

    Novelty: exclusive use of value centroid for page routing — inverting
    the universal assumption that keys (not values) predict page relevance.
    §16.4 — pure value-side scoring.
    """

    def __init__(self):
        super().__init__()
        # indexer-side
        self.q_mean = Mean(dim=1)
        self.gemm = GeMM()
        self.output_func = topK()

        # cache-side
        self.v_mean = CMean(dim=1)

    def create_cache(self, block_size: int, head_dim: int):
        return {
            "v_centroids": (1, head_dim),
        }

    def forward_cache(self, cache, loc, ctx):
        self.v_mean(cache["v"], cache["v_centroids"], loc=loc, ctx=ctx)

    def forward_indexer(self, q, o, cache, ctx):
        q_mean = self.q_mean(q, ctx=ctx)                            # [1, 1, D]
        score = self.gemm(q_mean, cache["v_centroids"], ctx=ctx)    # [S, 1, 1]
        self.output_func(score, o, ctx=ctx)
