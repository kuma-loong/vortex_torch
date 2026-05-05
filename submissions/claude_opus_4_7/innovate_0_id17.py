"""id17 — Two-resolution (Mean + Half-block Max) centroid.

Novelty (op-set thread): pair the standard whole-block `CMean(dim=1)`
centroid with a half-block `CMaxInterleave(dim=1, k=block_size//2)` —
i.e., 2 sub-block max-summaries per block (NOT a Mean. Maxes are
non-linear and survive RoPE rotation, complementing the linear mean).
Indexer combines the two sub-scores with a *Multiply* — pages must
align with both the bulk mean AND have at least one half-block whose
max keys point in q's direction. Distinct from §14.1 (Mean + MeanInterleave
+ L2Norm dual-band) which is all-mean.
"""

import torch
from typing import Dict

from vortex_torch.flow import vFlow, register
from vortex_torch.indexer import topK, Mean, GeMM, Multiply, Sum, Max
from vortex_torch.cache import Mean as CMean, MaxInterleave as CMaxInterleave
from vortex_torch.abs import ContextBase


_HALF = 8   # block_size = 16 → 2 half-block max summaries per block


@register("claude_opus_4_7_innovate_0_id17_cls")
class Id17Cls(vFlow):
    def __init__(self):
        super().__init__()
        self.q_mean = Mean(dim=1)
        self.gemm_centroid = GeMM()
        self.mul_max = Multiply()
        self.sum_max = Sum(dim=2)
        self.max_subblock = Max(dim=1)
        self.combine = Multiply()
        self.output_func = topK()

        self.k_mean = CMean(dim=1)
        self.k_halfmax = CMaxInterleave(dim=1, k=_HALF)

    def create_cache(self, block_size: int, head_dim: int):
        return {
            "centroids": (1, head_dim),
            "halfmax":   (block_size // _HALF, head_dim),
        }

    def forward_cache(self, cache, loc, ctx: ContextBase):
        self.k_mean(cache["k"], cache["centroids"], loc=loc, ctx=ctx)
        self.k_halfmax(cache["k"], cache["halfmax"], loc=loc, ctx=ctx)

    def forward_indexer(self, q, o, cache, ctx: ContextBase):
        qm = self.q_mean(q, ctx=ctx)                                 # [1, 1, D]
        score_centroid = self.gemm_centroid(qm, cache["centroids"], ctx=ctx)   # [S, 1, 1]
        # halfmax: [S, 2, D]; broadcast q_mean → [S, 2, D]
        s_max = self.mul_max(qm, cache["halfmax"], ctx=ctx)
        sum_d = self.sum_max(s_max, ctx=ctx)                          # [S, 2, 1]
        score_max = self.max_subblock(sum_d, ctx=ctx)                 # [S, 1, 1]
        score = self.combine(score_centroid, score_max, ctx=ctx)
        self.output_func(score, o, ctx=ctx)
