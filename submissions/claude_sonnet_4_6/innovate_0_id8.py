import torch
from typing import Dict

from vortex_torch.flow import vFlow, register
from vortex_torch.abs import ContextBase
from vortex_torch.indexer import (
    topK, Mean, GeMM, Add,
)
from vortex_torch.cache import (
    Mean as CMean, L2Norm as CL2Norm, Log as CLog,
)


@register("claude_sonnet_4_6_innovate_0_id8_cls")
class LogMagBiasCentroid(vFlow):
    """
    Add log(‖centroid‖) as a per-page scalar bias to the centroid dot
    product score.  This favours pages whose centroids have moderate-to-
    large magnitude without dividing by it (which is what cosine similarity
    does).  The log compresses large norms to avoid domination.

    Unlike centroid centering or scaling (which are order-preserving for
    all pages uniformly), the bias here varies per page and thus
    genuinely changes the selected set.

    Novelty: per-page log-magnitude centroid norm as an additive bias
    on the relevance score — a per-page temperature derived from the
    centroid itself.
    §16.2 — per-page temperature (untried knob).
    """

    def __init__(self):
        super().__init__()
        # indexer-side
        self.q_mean = Mean(dim=1)
        self.gemm = GeMM()
        self.add_bias = Add(alpha=1.0, beta=0.1)   # score + 0.1 * log_norm
        self.output_func = topK()

        # cache-side
        self.k_mean = CMean(dim=1)
        self.cent_norm = CL2Norm(dim=2)   # [1, 1, D] → [1, 1, 1]
        self.log_norm = CLog(alpha=1e-6, beta=1.0)   # safe log

    def create_cache(self, block_size: int, head_dim: int):
        return {
            "centroids":     (1, head_dim),
            "log_cent_norm": (1, 1),
        }

    def forward_cache(self, cache, loc, ctx):
        self.k_mean(cache["k"], cache["centroids"], loc=loc, ctx=ctx)
        norm_raw = self.cent_norm(cache["centroids"], None, loc=loc, ctx=ctx)    # [1, 1, 1]
        self.log_norm(norm_raw, cache["log_cent_norm"], loc=loc, ctx=ctx)        # [1, 1, 1]

    def forward_indexer(self, q, o, cache, ctx):
        q_mean = self.q_mean(q, ctx=ctx)                              # [1, 1, D]
        dot = self.gemm(q_mean, cache["centroids"], ctx=ctx)          # [S, 1, 1]
        score = self.add_bias(dot, cache["log_cent_norm"], ctx=ctx)   # [S, 1, 1]
        self.output_func(score, o, ctx=ctx)
