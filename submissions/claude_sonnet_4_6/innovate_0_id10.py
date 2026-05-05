import torch
from typing import Dict

from vortex_torch.flow import vFlow, register
from vortex_torch.abs import ContextBase
from vortex_torch.indexer import (
    topK, Mean, GeMM, Conv1d,
)
from vortex_torch.cache import (
    Mean as CMean,
)


@register("claude_sonnet_4_6_innovate_0_id10_cls")
class Conv1dScoreSmoothing(vFlow):
    """
    Apply a symmetric 3-tap smoothing kernel along the page axis after
    computing per-page centroid dot-product scores.  Adjacent KV blocks
    in the same context window often encode related content; smoothing
    propagates relevance signal to neighbours of high-scoring pages.

    Novelty: Conv1d(dim=0) used as a page-axis score smoother — no paper
    or existing flow applies a 1-D convolutional stencil to the page
    scores before topK.
    §16.2 — untried Conv1d knob for score propagation.
    """

    def __init__(self):
        super().__init__()
        # indexer-side
        self.q_mean = Mean(dim=1)
        self.gemm = GeMM()
        # 3-tap Gaussian-like smoothing along the page axis (dim=0)
        # weight shape: [K, D_0, D_1] = [3, 1, 1]
        self.smooth = Conv1d(weight=[[[0.25]], [[0.5]], [[0.25]]], dim=0)
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
        raw_score = self.gemm(q_mean, cache["centroids"], ctx=ctx) # [S, 1, 1]
        score = self.smooth(raw_score, ctx=ctx)                    # [S, 1, 1]
        self.output_func(score, o, ctx=ctx)
