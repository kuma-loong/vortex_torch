"""id12 — Bipolar Softmax: pick *moderate-relevance* pages.

Novelty (§16.3 inversion): every routing kernel sharpens the score
distribution with `Softmax(scale > 0)` then takes the top. We multiply
the raw score by `Softmax(dim=0, scale=-1)` — a *negative-temperature*
softmax that emphasises low-score pages. The product `score *
neg_softmax(score)` is non-monotonic in score, peaking at moderate
relevance and damping both extremes. Tests the inverse of "extremes
win" — does picking moderately-relevant pages preserve more diverse
context than chasing the sharpest peaks? Falsifies or confirms the
implicit "argmax-like" picker assumption.
"""

import torch
from typing import Dict

from vortex_torch.flow import vFlow, register
from vortex_torch.indexer import topK, Mean, GeMM, Softmax, Multiply
from vortex_torch.cache import Mean as CMean
from vortex_torch.abs import ContextBase


@register("claude_opus_4_7_innovate_0_id12_cls")
class Id12Cls(vFlow):
    def __init__(self):
        super().__init__()
        self.q_mean = Mean(dim=1)
        self.gemm = GeMM()
        self.neg_softmax = Softmax(dim=0, scale=-1.0)   # emphasises low-score pages
        self.bipolar = Multiply()
        self.output_func = topK()

        self.k_mean = CMean(dim=1)

    def create_cache(self, block_size: int, head_dim: int):
        return {"centroids": (1, head_dim)}

    def forward_cache(self, cache, loc, ctx: ContextBase):
        self.k_mean(cache["k"], cache["centroids"], loc=loc, ctx=ctx)

    def forward_indexer(self, q, o, cache, ctx: ContextBase):
        qm = self.q_mean(q, ctx=ctx)                                 # [1, 1, D]
        score = self.gemm(qm, cache["centroids"], ctx=ctx)           # [S, 1, 1]
        weights = self.neg_softmax(score, ctx=ctx)                    # [S, 1, 1]
        moderate = self.bipolar(score, weights, ctx=ctx)              # non-monotonic
        self.output_func(moderate, o, ctx=ctx)
