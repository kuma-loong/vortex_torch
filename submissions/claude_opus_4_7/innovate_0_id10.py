"""id10 — V-mean centroid routing (route by values, not keys).

Novelty (§16.4 first-principles): every flow in the literature computes
the centroid from `cache["k"]` and scores `q · k_centroid`. The query
projection and key projection share an inner product, so this is
natural. But the *value* projection — `cache["v"]` — is what actually
gets attended to in the dense kernel. We swap: build `v_centroids`
from `CMean(cache["v"])` and score `q · v_centroid`. Tests the asymmetry
of routing on the carrier signal vs. the alignment signal. No paper
here scores against a value mean.
"""

import torch
from typing import Dict

from vortex_torch.flow import vFlow, register
from vortex_torch.indexer import topK, Mean, GeMM
from vortex_torch.cache import Mean as CMean
from vortex_torch.abs import ContextBase


@register("claude_opus_4_7_innovate_0_id10_cls")
class Id10Cls(vFlow):
    def __init__(self):
        super().__init__()
        self.q_mean = Mean(dim=1)
        self.gemm = GeMM()
        self.output_func = topK()

        self.v_mean = CMean(dim=1)            # ← reduce cache["v"], not cache["k"]

    def create_cache(self, block_size: int, head_dim: int):
        return {"v_centroids": (1, head_dim)}

    def forward_cache(self, cache, loc, ctx: ContextBase):
        self.v_mean(cache["v"], cache["v_centroids"], loc=loc, ctx=ctx)

    def forward_indexer(self, q, o, cache, ctx: ContextBase):
        qm = self.q_mean(q, ctx=ctx)                                 # [1, 1, D]
        score = self.gemm(qm, cache["v_centroids"], ctx=ctx)         # [S, 1, 1]
        self.output_func(score, o, ctx=ctx)
