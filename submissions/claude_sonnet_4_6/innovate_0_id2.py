import torch
from typing import Dict

from vortex_torch.flow import vFlow, register
from vortex_torch.abs import ContextBase
from vortex_torch.indexer import (
    topK, Mean, GeMM,
)
from vortex_torch.cache import (
    Mean as CMean, L2Norm as CL2Norm,
    Multiply as CMultiply,
)


@register("claude_sonnet_4_6_innovate_0_id2_cls")
class VWeightedCentroid(vFlow):
    """
    Build the key centroid weighted by each token's value L2 norm, so
    keys whose corresponding values carry more signal dominate the centroid.
    Scoring with q · centroid then captures pages where the query aligns
    with high-value-energy keys.

    Novelty: value-token norms as per-token importance weights for the key
    centroid — no paper weights centroids by value magnitude.
    §16.4 — value-side signal exploitation.
    """

    def __init__(self):
        super().__init__()
        # indexer-side
        self.q_mean = Mean(dim=1)
        self.gemm = GeMM()
        self.output_func = topK()

        # cache-side
        self.v_tok_norm = CL2Norm(dim=2)    # [1, block_size, D] → [1, block_size, 1]
        self.wt_mul = CMultiply()           # [1, block_size, D] * [1, block_size, 1] → [1, block_size, D]
        self.wt_mean = CMean(dim=1)         # [1, block_size, D] → [1, 1, D]

    def create_cache(self, block_size: int, head_dim: int):
        return {
            "vw_centroids": (1, head_dim),
        }

    def forward_cache(self, cache, loc, ctx):
        v_tok = self.v_tok_norm(cache["v"], None, loc=loc, ctx=ctx)              # [1, block_size, 1]
        wt_k = self.wt_mul(cache["k"], v_tok, None, loc=loc, ctx=ctx)            # [1, block_size, D]
        self.wt_mean(wt_k, cache["vw_centroids"], loc=loc, ctx=ctx)              # [1, 1, D]

    def forward_indexer(self, q, o, cache, ctx):
        q_mean = self.q_mean(q, ctx=ctx)                                # [1, 1, D]
        score = self.gemm(q_mean, cache["vw_centroids"], ctx=ctx)       # [S, 1, 1]
        self.output_func(score, o, ctx=ctx)
