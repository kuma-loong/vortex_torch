"""id7 — Conv1d page-axis smoothing of the per-page score.

Novelty (op-set thread): the framework's `Conv1d(dim=0)` slides a
user-chosen kernel along the *page* axis within each sequence segment.
No paper here proposes smoothing routing scores across neighbouring
pages, but if relevance is locally clustered (sentence-level coherence),
a 3-tap [0.25, 0.5, 0.25] smoother lets a page borrow score from its
neighbours, lifting boundary pages whose centroids barely missed the
threshold. Tests whether positional smoothness is a free Pareto lever.
"""

import torch
from typing import Dict

from vortex_torch.flow import vFlow, register
from vortex_torch.indexer import topK, Mean, GeMM, Conv1d
from vortex_torch.cache import Mean as CMean
from vortex_torch.abs import ContextBase


@register("claude_opus_4_7_innovate_0_id7_cls")
class Id7Cls(vFlow):
    def __init__(self):
        super().__init__()
        self.q_mean = Mean(dim=1)
        self.gemm = GeMM()
        # 3-tap smoothing kernel along the page axis. weight shape [K, D_0, D_1].
        # For score [S, 1, 1], D_0=1, D_1=1.
        self.smooth = Conv1d(
            weight=[[[0.25]], [[0.5]], [[0.25]]],
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
        smoothed = self.smooth(score, ctx=ctx)                        # [S, 1, 1]
        self.output_func(smoothed, o, ctx=ctx)
