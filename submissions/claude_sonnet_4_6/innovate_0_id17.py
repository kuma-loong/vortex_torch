import torch
from typing import Dict

from vortex_torch.flow import vFlow, register
from vortex_torch.abs import ContextBase
from vortex_torch.indexer import (
    topK, Mean, GeMM, Relu, Multiply,
)
from vortex_torch.cache import (
    Mean as CMean,
)


@register("claude_sonnet_4_6_innovate_0_id17_cls")
class ReluGatedKVProduct(vFlow):
    """
    Compute key-centroid and value-centroid dot-product scores with the
    query mean.  ReLU-zero both scores, then multiply them.  A page scores
    non-zero ONLY if it is simultaneously positively aligned with the query
    in BOTH the key space and the value space.

    Contrast with V-key coalignment (Minimum): the product here is a
    "smooth AND" that amplifies doubly-aligned pages more than it
    penalises singly-aligned ones.

    Novelty: ReLU-then-Multiply on key and value scores for a logical-AND
    with smooth amplification — no paper uses element-wise product of
    independently gated k and v scores for routing.
    Op-set derived (Relu + Multiply pipeline on two GeMM scores).
    """

    def __init__(self):
        super().__init__()
        # indexer-side
        self.q_mean = Mean(dim=1)
        self.gemm_k = GeMM()
        self.gemm_v = GeMM()
        self.relu_k = Relu(alpha=0.0, beta=0.0)
        self.relu_v = Relu(alpha=0.0, beta=0.0)
        self.product = Multiply()
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
        q_mean = self.q_mean(q, ctx=ctx)                              # [1, 1, D]
        k_score = self.gemm_k(q_mean, cache["k_cents"], ctx=ctx)     # [S, 1, 1]
        v_score = self.gemm_v(q_mean, cache["v_cents"], ctx=ctx)     # [S, 1, 1]
        rk = self.relu_k(k_score, ctx=ctx)
        rv = self.relu_v(v_score, ctx=ctx)
        score = self.product(rk, rv, ctx=ctx)                        # [S, 1, 1]
        self.output_func(score, o, ctx=ctx)
