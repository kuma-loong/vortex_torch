"""id15 — Cross-step Q-EMA via Save/Load.

Novelty (op-set thread): every Save/Load flow in the catalog accumulates
a per-page *score* (heavy hitter) or *count* (anti-redundancy). We do
the opposite — Save/Load a sequence-level *query EMA*: each step we
load the previous EMA, blend in the current `q_mean`, save the new
EMA, and route by `q_ema · centroid`. The router gets a *temporally
smoothed* query that reflects what the model has been asking for over
the last few decode steps, not just the most recent token. No paper
here computes a moving-average of past queries; H2O / Keyformer move
averages of *attention*, which is a fundamentally different signal.
"""

import torch
from typing import Dict

from vortex_torch.flow import vFlow, register
from vortex_torch.indexer import topK, Mean, GeMM, Load, Save, Add
from vortex_torch.cache import Mean as CMean, Fill as CFill
from vortex_torch.abs import ContextBase


@register("claude_opus_4_7_innovate_0_id15_cls")
class Id15Cls(vFlow):
    DECAY = 0.7   # weight on previous EMA
    NEW = 0.3     # weight on current q_mean

    def __init__(self):
        super().__init__()
        self.q_mean = Mean(dim=1)
        self.load_qema = Load()
        self.fuse = Add(alpha=self.DECAY, beta=self.NEW)
        self.save_qema = Save()
        self.gemm = GeMM()
        self.output_func = topK()

        self.k_mean = CMean(dim=1)
        self.init_qema = CFill(alpha=0.0)

    def create_cache(self, block_size: int, head_dim: int):
        return {
            "centroids": (1, head_dim),
            "q_ema":     (1, head_dim),   # broadcast-uniform across pages
        }

    def forward_cache(self, cache, loc, ctx: ContextBase):
        self.k_mean(cache["k"], cache["centroids"], loc=loc, ctx=ctx)
        self.init_qema(cache["q_ema"], loc=loc, ctx=ctx)

    def forward_indexer(self, q, o, cache, ctx: ContextBase):
        qm = self.q_mean(q, ctx=ctx)                                  # [1, 1, D]
        prev_ema = self.load_qema(cache["q_ema"], ctx=ctx)            # [S, 1, D]
        new_ema = self.fuse(prev_ema, qm, ctx=ctx)                    # 0.7*prev + 0.3*qm
        self.save_qema(new_ema, cache["q_ema"], ctx=ctx)
        score = self.gemm(new_ema, cache["centroids"], ctx=ctx)        # [S, 1, 1]
        self.output_func(score, o, ctx=ctx)
