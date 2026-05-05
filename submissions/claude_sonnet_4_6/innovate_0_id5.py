import torch
from typing import Dict

from vortex_torch.flow import vFlow, register
from vortex_torch.abs import ContextBase
from vortex_torch.indexer import (
    topK, Multiply, Relu, Sum, Max,
)
from vortex_torch.cache import (
    Max as CMax,
)


@register("claude_sonnet_4_6_innovate_0_id5_cls")
class ReluQuestScoring(vFlow):
    """
    Inversion of QUEST: instead of max(q·k_max, q·k_min) keep only the
    positive components of q·k_max via ReLU.  This selects pages where
    the query direction reinforces the maximum-key direction — ignoring
    features where the query is counter-directional.

    Novelty: ReLU-gated single-envelope score discards negative-alignment
    features; QUEST takes max over two envelopes but never hard-zeros them.
    §16.3 — inversion of the QUEST symmetric-envelope assumption.
    """

    def __init__(self):
        super().__init__()
        # indexer-side
        self.mul_max = Multiply()
        self.relu = Relu(alpha=0.0, beta=0.0)
        self.sum_d = Sum(dim=2)
        self.max_h = Max(dim=1)
        self.output_func = topK()

        # cache-side
        self.k_max = CMax(dim=1)

    def create_cache(self, block_size: int, head_dim: int):
        return {
            "k_max": (1, head_dim),
        }

    def forward_cache(self, cache, loc, ctx):
        self.k_max(cache["k"], cache["k_max"], loc=loc, ctx=ctx)

    def forward_indexer(self, q, o, cache, ctx):
        s = self.mul_max(q, cache["k_max"], ctx=ctx)     # [S, H_q, D]
        s_relu = self.relu(s, ctx=ctx)                   # [S, H_q, D] (zero out negatives)
        score_h = self.sum_d(s_relu, ctx=ctx)            # [S, H_q, 1]
        score = self.max_h(score_h, ctx=ctx)             # [S, 1,   1]
        self.output_func(score, o, ctx=ctx)
