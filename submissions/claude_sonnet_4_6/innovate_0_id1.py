import torch
from typing import Dict

from vortex_torch.flow import vFlow, register
from vortex_torch.abs import ContextBase
from vortex_torch.indexer import (
    topK, Mean, GeMM, Minimum,
)
from vortex_torch.cache import (
    Mean as CMean,
)


@register("claude_sonnet_4_6_innovate_0_id1_cls")
class VKeyCoalignment(vFlow):
    """
    Select pages where both the key centroid AND the value centroid align
    with the query — using the minimum of the two scores as the gate.
    A page must be relevant on both key and value axes to be selected.

    Novelty: Minimum(k_score, v_score) as a logical-AND co-alignment
    score; no paper uses v-centroid alignment as an independent signal.
    §16.4 — value-side signal exploitation.
    """

    def __init__(self):
        super().__init__()
        # indexer-side
        self.q_mean = Mean(dim=1)
        self.gemm_k = GeMM()
        self.gemm_v = GeMM()
        self.min_score = Minimum()
        self.output_func = topK()

        # cache-side
        self.k_mean = CMean(dim=1)
        self.v_mean = CMean(dim=1)

    def create_cache(self, block_size: int, head_dim: int):
        return {
            "k_cents": (1, head_dim),
            "v_cents": (1, head_dim),
        }

    def forward_cache(self, cache, loc, ctx):
        self.k_mean(cache["k"], cache["k_cents"], loc=loc, ctx=ctx)
        self.v_mean(cache["v"], cache["v_cents"], loc=loc, ctx=ctx)

    def forward_indexer(self, q, o, cache, ctx):
        q_mean = self.q_mean(q, ctx=ctx)                            # [1, 1, D]
        k_score = self.gemm_k(q_mean, cache["k_cents"], ctx=ctx)    # [S, 1, 1]
        v_score = self.gemm_v(q_mean, cache["v_cents"], ctx=ctx)    # [S, 1, 1]
        score = self.min_score(k_score, v_score, ctx=ctx)           # [S, 1, 1]
        self.output_func(score, o, ctx=ctx)
