"""id19 — Negative Conv1d (Laplacian) — emphasise *isolated* peaks.

Novelty (op-set thread + §16.3 inversion of id7): id7 *smooths* the
score along the page axis to reward locally-clustered relevance. We
invert: a Laplacian-like kernel `[-0.5, 1.5, -0.5]` *sharpens* the
score, penalising pages whose neighbours also score highly. The
hypothesis is that when relevance is clumpy, the topK budget gets
spent on near-duplicate neighbouring pages; sharpening picks one peak
per cluster and frees budget to cover more disjoint regions of the
sequence. Pure first-principles use of `Conv1d(dim=0)` with a
non-smoothing kernel.
"""

import torch
from typing import Dict

from vortex_torch.flow import vFlow, register
from vortex_torch.indexer import topK, Mean, GeMM, Conv1d
from vortex_torch.cache import Mean as CMean
from vortex_torch.abs import ContextBase


@register("claude_opus_4_7_innovate_0_id19_cls")
class Id19Cls(vFlow):
    def __init__(self):
        super().__init__()
        self.q_mean = Mean(dim=1)
        self.gemm = GeMM()
        # Laplacian-like sharpening kernel along the page axis.
        # weight shape [K, D_0, D_1] = [3, 1, 1].
        self.sharpen = Conv1d(
            weight=[[[-0.5]], [[1.5]], [[-0.5]]],
            dim=0,
        )
        self.output_func = topK()

        self.k_mean = CMean(dim=1)

    def create_cache(self, block_size: int, head_dim: int):
        return {"centroids": (1, head_dim)}

    def forward_cache(self, cache, loc, ctx: ContextBase):
        self.k_mean(cache["k"], cache["centroids"], loc=loc, ctx=ctx)

    def forward_indexer(self, q, o, cache, ctx: ContextBase):
        qm = self.q_mean(q, ctx=ctx)                                 # [1, 1, D]
        score = self.gemm(qm, cache["centroids"], ctx=ctx)           # [S, 1, 1]
        sharp = self.sharpen(score, ctx=ctx)                          # [S, 1, 1]
        self.output_func(sharp, o, ctx=ctx)
