import torch
from typing import Dict

from vortex_torch.flow import vFlow, register
from vortex_torch.abs import ContextBase
from vortex_torch.indexer import (
    topK, Mean, GeMM, Softmax, Log, Add_Mul, Multiply,
)
from vortex_torch.cache import (
    Mean as CMean,
)


@register("claude_sonnet_4_6_innovate_0_id14_cls")
class EntropyBasedSelection(vFlow):
    """
    Convert per-page centroid dot-product scores to a probability
    distribution via Softmax, then score each page by its entropy
    contribution: p * (-log p).  Pages with intermediate probability
    (neither near-certain nor negligible) receive the highest score.

    This inverts the typical "pick the highest score" heuristic: instead
    of maximising the top-1 certainty, it maximises the uncertainty of
    the selection set, favouring pages the model is most undecided about.

    Novelty: entropy-contribution p*(-log p) as the page ranking signal —
    derived from the Softmax + Log + Multiply ops in the framework and the
    information-theoretic intuition that uncertain pages carry the most
    potential information.
    Op-set derived (Softmax → Log → Add_Mul → Multiply pipeline).
    """

    def __init__(self):
        super().__init__()
        # indexer-side
        self.q_mean = Mean(dim=1)
        self.gemm = GeMM()
        self.softmax = Softmax(dim=0, scale=0.1)
        self.log_p = Log(alpha=1e-8, beta=1.0)        # safe log(p)
        self.neg_log = Add_Mul(alpha=0.0, beta=-1.0)  # -log(p)
        self.entropy_w = Multiply()                    # p * (-log p)
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
        q_mean = self.q_mean(q, ctx=ctx)                            # [1, 1, D]
        raw = self.gemm(q_mean, cache["centroids"], ctx=ctx)        # [S, 1, 1]
        soft = self.softmax(raw, ctx=ctx)                           # [S, 1, 1]  ← probs
        log_p = self.log_p(soft, ctx=ctx)                           # [S, 1, 1]
        neg_log_p = self.neg_log(log_p, ctx=ctx)                    # [S, 1, 1] = -log(p)
        score = self.entropy_w(soft, neg_log_p, ctx=ctx)            # [S, 1, 1] = p*(-log p)
        self.output_func(score, o, ctx=ctx)
