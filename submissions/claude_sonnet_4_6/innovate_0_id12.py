import torch
from typing import Dict

from vortex_torch.flow import vFlow, register
from vortex_torch.abs import ContextBase
from vortex_torch.indexer import (
    topK, Mean, GeMM, Add, Minimum, Load, Save,
)
from vortex_torch.cache import (
    Mean as CMean, Fill as CFill,
)


@register("claude_sonnet_4_6_innovate_0_id12_cls")
class TwoTimescaleEMA(vFlow):
    """
    Maintain two EMAs of the per-page centroid dot-product score across
    decode steps: a fast one (momentum=0.5) and a slow one (momentum=0.9).
    Select pages by the minimum of the two EMAs.

    A page must sustain high scores on both the short-horizon (recent few
    steps) AND long-horizon (many steps ago) timescales to be selected.
    This filters out pages that spike momentarily but are otherwise cold.

    Novelty: two Save/Load EMA slots at different timescales, combined
    via Minimum — pages must be relevant at both timescales simultaneously.
    §16.2 — multi-timescale persistent score (untried knob).
    """

    def __init__(self):
        super().__init__()
        # indexer-side
        self.q_mean = Mean(dim=1)
        self.gemm = GeMM()
        self.load_fast = Load()
        self.load_slow = Load()
        self.ema_fast = Add(alpha=0.5, beta=0.5)    # fast: equal blend
        self.ema_slow = Add(alpha=0.9, beta=0.1)    # slow: heavy momentum
        self.save_fast = Save()
        self.save_slow = Save()
        self.score_min = Minimum()                  # require both timescales
        self.output_func = topK()

        # cache-side
        self.k_mean = CMean(dim=1)
        self.init_fast = CFill(alpha=0.0)
        self.init_slow = CFill(alpha=0.0)

    def create_cache(self, block_size: int, head_dim: int):
        return {
            "centroids": (1, head_dim),
            "fast_ema":  (1, 1),
            "slow_ema":  (1, 1),
        }

    def forward_cache(self, cache, loc, ctx):
        self.k_mean(cache["k"], cache["centroids"], loc=loc, ctx=ctx)
        self.init_fast(cache["fast_ema"], loc=loc, ctx=ctx)
        self.init_slow(cache["slow_ema"], loc=loc, ctx=ctx)

    def forward_indexer(self, q, o, cache, ctx):
        q_mean = self.q_mean(q, ctx=ctx)                             # [1, 1, D]
        curr = self.gemm(q_mean, cache["centroids"], ctx=ctx)        # [S, 1, 1]
        prev_f = self.load_fast(cache["fast_ema"], ctx=ctx)          # [S, 1, 1]
        prev_s = self.load_slow(cache["slow_ema"], ctx=ctx)          # [S, 1, 1]
        new_f = self.ema_fast(prev_f, curr, ctx=ctx)
        new_s = self.ema_slow(prev_s, curr, ctx=ctx)
        self.save_fast(new_f, cache["fast_ema"], ctx=ctx)
        self.save_slow(new_s, cache["slow_ema"], ctx=ctx)
        score = self.score_min(new_f, new_s, ctx=ctx)                # [S, 1, 1]
        self.output_func(score, o, ctx=ctx)
