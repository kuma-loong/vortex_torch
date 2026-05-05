import torch
from typing import Dict

from vortex_torch.flow import vFlow, register
from vortex_torch.abs import ContextBase
from vortex_torch.indexer import (
    topK, Mean, GeMM, Multiply, Add, Load, Save,
)
from vortex_torch.cache import (
    Mean as CMean, L2Norm as CL2Norm, Fill as CFill,
)


@register("claude_sonnet_4_6_innovate_0_id11_cls")
class RunningVNormAccum(vFlow):
    """
    Accumulate an EMA of each page's mean-value-token-magnitude across
    decode steps and use it to gate the centroid dot-product score.

    Pages that consistently emit large-magnitude values are up-weighted;
    pages with small value activity are down-weighted even if their key
    centroid temporarily aligns with the query.

    Novelty: Save/Load EMA over per-page value-norm (not key score) as a
    persistent importance weight; no paper accumulates value-magnitude
    history as a decaying attention budget.
    §16.2 (Save/Load + v-signal, untried combination).
    """

    ALPHA = 0.95   # EMA momentum for accumulated v-norm

    def __init__(self):
        super().__init__()
        # indexer-side
        self.q_mean = Mean(dim=1)
        self.gemm = GeMM()
        self.load_v = Load()
        self.ema_update = Add(alpha=self.ALPHA, beta=1.0 - self.ALPHA)
        self.save_v = Save()
        self.gate = Multiply()
        self.output_func = topK()

        # cache-side
        self.k_mean = CMean(dim=1)
        self.v_tok_norm = CL2Norm(dim=2)   # [1, block_size, D] → [1, block_size, 1]
        self.v_mean_norm = CMean(dim=1)    # [1, block_size, 1] → [1, 1, 1]
        self.init_accum = CFill(alpha=0.0)

    def create_cache(self, block_size: int, head_dim: int):
        return {
            "centroids":    (1, head_dim),
            "v_norm_block": (1, 1),        # mean token norm in this block
            "v_norm_accum": (1, 1),        # EMA across decode steps (Save/Load)
        }

    def forward_cache(self, cache, loc, ctx):
        self.k_mean(cache["k"], cache["centroids"], loc=loc, ctx=ctx)
        v_tok = self.v_tok_norm(cache["v"], None, loc=loc, ctx=ctx)            # [1, block_size, 1]
        self.v_mean_norm(v_tok, cache["v_norm_block"], loc=loc, ctx=ctx)       # [1, 1, 1]
        self.init_accum(cache["v_norm_accum"], loc=loc, ctx=ctx)

    def forward_indexer(self, q, o, cache, ctx):
        q_mean = self.q_mean(q, ctx=ctx)                             # [1, 1, D]
        dot = self.gemm(q_mean, cache["centroids"], ctx=ctx)         # [S, 1, 1]
        prev_v = self.load_v(cache["v_norm_accum"], ctx=ctx)         # [S, 1, 1]
        new_v = self.ema_update(prev_v, cache["v_norm_block"], ctx=ctx)  # EMA update
        self.save_v(new_v, cache["v_norm_accum"], ctx=ctx)
        score = self.gate(dot, new_v, ctx=ctx)                       # [S, 1, 1]
        self.output_func(score, o, ctx=ctx)
