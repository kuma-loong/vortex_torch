import torch
from typing import Dict

from vortex_torch.flow import vFlow, register
from vortex_torch.abs import ContextBase
from vortex_torch.indexer import (
    topK, Mean, GeMM, Abs, Add, Conv1d,
)
from vortex_torch.cache import (
    Mean as CMean,
)


@register("claude_sonnet_4_6_innovate_0_id15_cls")
class BlockBoundaryBonus(vFlow):
    """
    After computing centroid dot-product scores, detect page-score
    boundaries using a 1-D second-derivative (Laplacian) filter along
    the page axis, take its absolute value, and add it as a bonus.

    Pages flanked by high-scoring and low-scoring neighbours — i.e. at
    the boundary of relevant/irrelevant regions — receive an extra bonus.
    The intuition: boundary pages are the last pages worth attending to
    and the first pages that aren't; they often contain the "transition"
    content where the relevant context ends.

    Novelty: Conv1d second-derivative (curvature) of page scores as an
    additive relevance bonus — exploits Conv1d with a Laplacian kernel
    to detect score discontinuities.
    §16.2 — untried Conv1d knob (curvature, not smoothing).
    """

    def __init__(self):
        super().__init__()
        # indexer-side
        self.q_mean = Mean(dim=1)
        self.gemm = GeMM()
        # Laplacian (second derivative) stencil along the page axis
        self.laplacian = Conv1d(weight=[[[1.0]], [[-2.0]], [[1.0]]], dim=0)
        self.abs_curve = Abs(alpha=0.0, beta=1.0)
        self.add_bonus = Add(alpha=1.0, beta=0.5)   # score + 0.5 * |curvature|
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
        q_mean = self.q_mean(q, ctx=ctx)                               # [1, 1, D]
        raw = self.gemm(q_mean, cache["centroids"], ctx=ctx)           # [S, 1, 1]
        curve = self.laplacian(raw, ctx=ctx)                           # [S, 1, 1]
        abs_c = self.abs_curve(curve, ctx=ctx)                         # [S, 1, 1]
        score = self.add_bonus(raw, abs_c, ctx=ctx)                    # [S, 1, 1]
        self.output_func(score, o, ctx=ctx)
