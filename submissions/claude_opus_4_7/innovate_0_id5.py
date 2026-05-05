"""id5 — Cross-step score variance routing (Save/Load).

Novelty (op-set thread): maintain TWO Save/Load fields per page —
running-mean of score (E[X]) and running-mean-of-squares (E[X^2]). The
per-page variance estimate `var = E[X^2] - E[X]^2` measures how
*uncertain* a page's relevance has been across decode steps. We bias
the current score by `+0.5*var`: pages whose alignment is volatile get
re-checked. No paper computes per-page score variance across steps.
"""

import torch
from typing import Dict

from vortex_torch.flow import vFlow, register
from vortex_torch.indexer import topK, Mean, GeMM, Load, Save, Add, Multiply
from vortex_torch.cache import Mean as CMean, Fill as CFill
from vortex_torch.abs import ContextBase


@register("claude_opus_4_7_innovate_0_id5_cls")
class Id5Cls(vFlow):
    DECAY = 0.7
    NEW = 0.3
    VAR_W = 0.5

    def __init__(self):
        super().__init__()
        self.q_mean = Mean(dim=1)
        self.gemm = GeMM()

        self.load_x = Load()
        self.load_x2 = Load()
        self.save_x = Save()
        self.save_x2 = Save()

        self.ema_x = Add(alpha=self.DECAY, beta=self.NEW)
        self.sq = Multiply()                              # current * current
        self.ema_x2 = Add(alpha=self.DECAY, beta=self.NEW)
        self.x_squared = Multiply()                       # E[X]^2
        self.var = Add(alpha=1.0, beta=-1.0)              # var = E[X^2] - E[X]^2
        self.score_with_var = Add(alpha=1.0, beta=self.VAR_W)

        self.output_func = topK()

        self.k_mean = CMean(dim=1)
        self.init_x = CFill(alpha=0.0)
        self.init_x2 = CFill(alpha=0.0)

    def create_cache(self, block_size: int, head_dim: int):
        return {
            "centroids": (1, head_dim),
            "ema_x":     (1, 1),
            "ema_x2":    (1, 1),
        }

    def forward_cache(self, cache, loc, ctx: ContextBase):
        self.k_mean(cache["k"], cache["centroids"], loc=loc, ctx=ctx)
        self.init_x(cache["ema_x"], loc=loc, ctx=ctx)
        self.init_x2(cache["ema_x2"], loc=loc, ctx=ctx)

    def forward_indexer(self, q, o, cache, ctx: ContextBase):
        qm = self.q_mean(q, ctx=ctx)                                  # [1, 1, D]
        current = self.gemm(qm, cache["centroids"], ctx=ctx)          # [S, 1, 1]

        prev_x = self.load_x(cache["ema_x"], ctx=ctx)                  # [S, 1, 1]
        prev_x2 = self.load_x2(cache["ema_x2"], ctx=ctx)               # [S, 1, 1]

        new_x = self.ema_x(prev_x, current, ctx=ctx)                   # 0.7*prev + 0.3*curr
        cur_sq = self.sq(current, current, ctx=ctx)                    # current^2
        new_x2 = self.ema_x2(prev_x2, cur_sq, ctx=ctx)

        self.save_x(new_x, cache["ema_x"], ctx=ctx)
        self.save_x2(new_x2, cache["ema_x2"], ctx=ctx)

        ex_sq = self.x_squared(new_x, new_x, ctx=ctx)                  # E[X]^2
        var = self.var(new_x2, ex_sq, ctx=ctx)                         # var ≥ 0

        score = self.score_with_var(current, var, ctx=ctx)             # current + 0.5*var
        self.output_func(score, o, ctx=ctx)
