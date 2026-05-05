"""id0 — Score velocity (temporal derivative of relevance).

Novelty (op-set thread, not in §16): the framework's `Save`/`Load` exposes
per-page state across decode steps. No paper here uses the *temporal
derivative* of per-page relevance as a routing signal. We Save the
previous step's per-page score and route by `current - 0.5*prev` —
favouring pages whose alignment is *rising* and dampening pages whose
relevance is fading. Distinct from Keyformer/H2O accumulators, which
*positively* sum past scores.
"""

import torch
from typing import Dict

from vortex_torch.flow import vFlow, register
from vortex_torch.indexer import topK, Mean, GeMM, Load, Save, Add
from vortex_torch.cache import Mean as CMean, Fill as CFill
from vortex_torch.abs import ContextBase


@register("claude_opus_4_7_innovate_0_id0_cls")
class Id0Cls(vFlow):
    BETA = -0.5  # subtract half of prev — fresh-rise bias

    def __init__(self):
        super().__init__()
        # Indexer-side ops
        self.q_mean = Mean(dim=1)
        self.gemm = GeMM()
        self.load_prev = Load()
        self.save_curr = Save()
        self.fresh = Add(alpha=1.0, beta=self.BETA)
        self.output_func = topK()
        # Cache-side ops
        self.k_mean = CMean(dim=1)
        self.init_prev = CFill(alpha=0.0)

    def create_cache(self, block_size: int, head_dim: int):
        return {
            "centroids":  (1, head_dim),
            "prev_score": (1, 1),
        }

    def forward_cache(self, cache, loc, ctx: ContextBase):
        self.k_mean(cache["k"], cache["centroids"], loc=loc, ctx=ctx)
        self.init_prev(cache["prev_score"], loc=loc, ctx=ctx)

    def forward_indexer(self, q, o, cache, ctx: ContextBase):
        qm = self.q_mean(q, ctx=ctx)                                 # [1, 1, D]
        current = self.gemm(qm, cache["centroids"], ctx=ctx)         # [S, 1, 1]
        prev = self.load_prev(cache["prev_score"], ctx=ctx)          # [S, 1, 1]
        score = self.fresh(current, prev, ctx=ctx)                   # current - 0.5*prev
        self.save_curr(current, cache["prev_score"], ctx=ctx)        # persist current
        self.output_func(score, o, ctx=ctx)
