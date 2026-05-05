"""id18 — Per-token v-mean GeMM dual + multiplicative gate.

Novelty (op-set thread): use *both* `CMean(dim=1)(cache["k"])` (key
centroid, the standard signal) and `CMean(dim=1)(cache["v"])` (value
centroid, mostly unused in routing literature). In the indexer, score
`q · k_centroid` AND `q · v_centroid` separately, then Multiply the two
per-page scalars. Pages must align with both the key direction AND the
value direction to be picked. Functionally distinct from id10 (which
uses *only* v-centroids): the *agreement* between the two centroid
spaces is the discriminator. No paper here scores against both.
"""

import torch
from typing import Dict

from vortex_torch.flow import vFlow, register
from vortex_torch.indexer import topK, Mean, GeMM, Multiply
from vortex_torch.cache import Mean as CMean
from vortex_torch.abs import ContextBase


@register("claude_opus_4_7_innovate_0_id18_cls")
class Id18Cls(vFlow):
    def __init__(self):
        super().__init__()
        self.q_mean = Mean(dim=1)
        self.gemm_k = GeMM()
        self.gemm_v = GeMM()
        self.combine = Multiply()
        self.output_func = topK()

        self.k_mean = CMean(dim=1)
        self.v_mean = CMean(dim=1)

    def create_cache(self, block_size: int, head_dim: int):
        return {
            "k_centroids": (1, head_dim),
            "v_centroids": (1, head_dim),
        }

    def forward_cache(self, cache, loc, ctx: ContextBase):
        self.k_mean(cache["k"], cache["k_centroids"], loc=loc, ctx=ctx)
        self.v_mean(cache["v"], cache["v_centroids"], loc=loc, ctx=ctx)

    def forward_indexer(self, q, o, cache, ctx: ContextBase):
        qm = self.q_mean(q, ctx=ctx)                                  # [1, 1, D]
        score_k = self.gemm_k(qm, cache["k_centroids"], ctx=ctx)      # [S, 1, 1]
        score_v = self.gemm_v(qm, cache["v_centroids"], ctx=ctx)      # [S, 1, 1]
        score = self.combine(score_k, score_v, ctx=ctx)               # non-monotonic
        self.output_func(score, o, ctx=ctx)
