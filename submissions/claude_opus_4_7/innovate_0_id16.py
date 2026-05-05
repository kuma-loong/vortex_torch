"""id16 — Per-page running-min relevance baseline.

Novelty (op-set thread + §16.3 inversion): heavy-hitter and EMA flows
track the running *mean* (or sum, or max) of per-page scores. We track
the running *minimum* via `Save/Load + Minimum()` and route by `current
- running_min`: pages that just rebounded from a low baseline get a
boost; pages whose alignment is stably low or stably high net to
near-zero. The key trick is initialisation — we `CFill(alpha=1e6)` so
the first `Minimum` step takes the first real score (otherwise zero
would clamp negative scores out). Tests whether *score elevation above
its own historical floor* is a better picker than absolute score.
"""

import torch
from typing import Dict

from vortex_torch.flow import vFlow, register
from vortex_torch.indexer import topK, Mean, GeMM, Load, Save, Minimum, Add
from vortex_torch.cache import Mean as CMean, Fill as CFill
from vortex_torch.abs import ContextBase


@register("claude_opus_4_7_innovate_0_id16_cls")
class Id16Cls(vFlow):
    INIT_HIGH = 1.0e6   # bf16-safe sentinel for "no min seen yet"

    def __init__(self):
        super().__init__()
        self.q_mean = Mean(dim=1)
        self.gemm = GeMM()
        self.load_min = Load()
        self.update_min = Minimum()
        self.save_min = Save()
        self.elevation = Add(alpha=1.0, beta=-1.0)   # current - running_min
        self.output_func = topK()

        self.k_mean = CMean(dim=1)
        self.init_min = CFill(alpha=self.INIT_HIGH)

    def create_cache(self, block_size: int, head_dim: int):
        return {
            "centroids": (1, head_dim),
            "min_score": (1, 1),
        }

    def forward_cache(self, cache, loc, ctx: ContextBase):
        self.k_mean(cache["k"], cache["centroids"], loc=loc, ctx=ctx)
        self.init_min(cache["min_score"], loc=loc, ctx=ctx)

    def forward_indexer(self, q, o, cache, ctx: ContextBase):
        qm = self.q_mean(q, ctx=ctx)                                 # [1, 1, D]
        current = self.gemm(qm, cache["centroids"], ctx=ctx)         # [S, 1, 1]
        prev_min = self.load_min(cache["min_score"], ctx=ctx)        # [S, 1, 1]
        new_min = self.update_min(prev_min, current, ctx=ctx)         # min(prev, curr)
        self.save_min(new_min, cache["min_score"], ctx=ctx)
        score = self.elevation(current, new_min, ctx=ctx)             # ≥ 0
        self.output_func(score, o, ctx=ctx)
