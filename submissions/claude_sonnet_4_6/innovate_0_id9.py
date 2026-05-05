import torch
from typing import Dict

from vortex_torch.flow import vFlow, register
from vortex_torch.abs import ContextBase
from vortex_torch.indexer import (
    topK, Mean, GeMM, WhereGreater, Add,
)
from vortex_torch.cache import (
    Mean as CMean,
)


@register("claude_sonnet_4_6_innovate_0_id9_cls")
class AboveMeanGating(vFlow):
    """
    Compute per-page centroid scores, threshold them against the
    per-sequence mean, and suppress below-mean pages to -∞ before topK.

    Pages below average are masked; topK then selects exclusively from the
    above-average pool.  This is a different constraint than plain topK:
    if fewer than k pages score above average, only those pages are
    selected (the BOS/EOS reserved pages fill the rest).

    Novelty: WhereGreater(score, Mean(dim=0)(score)) as a per-sequence
    adaptive hard threshold before topK — effectively a two-tier gate
    where only above-average pages compete for the sparse budget.
    §16.4 — cheapest effective flow + threshold gating.
    """

    def __init__(self):
        super().__init__()
        # indexer-side
        self.q_mean = Mean(dim=1)
        self.gemm = GeMM()
        self.mean_seq = Mean(dim=0)         # per-sequence mean across pages
        self.gate = WhereGreater()          # 0 if score > mean_score, -∞ otherwise
        self.add_gate = Add(alpha=1.0, beta=1.0)
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
        q_mean = self.q_mean(q, ctx=ctx)                          # [1, 1, D]
        score = self.gemm(q_mean, cache["centroids"], ctx=ctx)    # [S, 1, 1]
        mean_score = self.mean_seq(score, ctx=ctx)                # [1, 1, 1]
        gate = self.gate(score, mean_score, ctx=ctx)              # [S, 1, 1]: 0 or -∞
        gated = self.add_gate(score, gate, ctx=ctx)               # [S, 1, 1]: score or -∞
        self.output_func(gated, o, ctx=ctx)
