"""id11 — Multiplicative dual-feature score: centroid × per-feature norm.

Novelty (op-set thread): pair `CMean(dim=1)(k)` (the standard centroid)
with `CL2Norm(dim=1)(k)` (the per-feature L2 norm of the block's keys —
a magnitude profile, *not* a centroid norm) and combine via
multiplication of two q-driven per-page sub-scores. Both signals depend
on q non-trivially per page, so multiplying them is non-monotonic in
either alone — the picked set differs from "centroid only" or "norm
only". The hypothesis is that pages with both well-aligned mean *and*
matching feature-magnitude profile are the truly informative ones.
"""

import torch
from typing import Dict

from vortex_torch.flow import vFlow, register
from vortex_torch.indexer import topK, Mean, GeMM, Multiply, Sum
from vortex_torch.cache import Mean as CMean, L2Norm as CL2Norm
from vortex_torch.abs import ContextBase


@register("claude_opus_4_7_innovate_0_id11_cls")
class Id11Cls(vFlow):
    def __init__(self):
        super().__init__()
        self.q_mean = Mean(dim=1)
        self.gemm = GeMM()
        self.mul_norm = Multiply()
        self.sum_norm = Sum(dim=2)
        self.combine = Multiply()
        self.output_func = topK()

        self.k_mean = CMean(dim=1)
        self.k_norms = CL2Norm(dim=1)

    def create_cache(self, block_size: int, head_dim: int):
        return {
            "centroids": (1, head_dim),
            "k_norms":   (1, head_dim),
        }

    def forward_cache(self, cache, loc, ctx: ContextBase):
        self.k_mean(cache["k"], cache["centroids"], loc=loc, ctx=ctx)
        self.k_norms(cache["k"], cache["k_norms"], loc=loc, ctx=ctx)

    def forward_indexer(self, q, o, cache, ctx: ContextBase):
        qm = self.q_mean(q, ctx=ctx)                                  # [1, 1, D]
        score_mean = self.gemm(qm, cache["centroids"], ctx=ctx)       # [S, 1, 1]
        # q · k_norms (per-feature broadcast multiply), then sum
        s_norm = self.mul_norm(qm, cache["k_norms"], ctx=ctx)         # [S, 1, D]
        score_norm = self.sum_norm(s_norm, ctx=ctx)                   # [S, 1, 1]
        score = self.combine(score_mean, score_norm, ctx=ctx)         # non-monotonic
        self.output_func(score, o, ctx=ctx)
