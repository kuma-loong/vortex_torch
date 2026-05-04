"""Off-catalog (papers/guide.md §16.1): Prism (high-frequency centroid) ×
Keyformer (heavy-hitter accumulator).

Hypothesis: a slash-aware score (CMean centroid + per-feature CL2Norm
of keys) wrapped in a Save/Load momentum accumulator picks pages that
neither plain CMean+topK nor plain heavy-hitter alone catches —
because the dual-band term recovers RoPE high-frequency information
that mean-pooling destroys, and the accumulator amplifies pages that
are repeatedly relevant across decode steps.
"""

import torch
from typing import Dict

from vortex_torch.flow import vFlow, register
from vortex_torch.indexer import (
    topK, Mean, GeMM, Multiply, Sum, Add, Load, Save,
)
from vortex_torch.cache import Mean as CMean, L2Norm as CL2Norm, Fill as CFill
from vortex_torch.abs import ContextBase


@register("claude_opus_4_7_batch_0_id8_cls")
class ClaudeOpus47Batch0Id8Cls(vFlow):
    ALPHA = 0.5   # weight on raw dot product (semantic / low-frequency)
    BETA  = 0.5   # weight on magnitude-interaction (high-frequency / Prism)
    DECAY = 0.9   # heavy-hitter momentum

    def __init__(self):
        super().__init__()
        # Indexer-side ops
        self.q_mean      = Mean(dim=1)
        self.gemm_dot    = GeMM()
        self.mul_norm    = Multiply()
        self.sum_norm    = Sum(dim=2)
        self.blend       = Add(alpha=self.ALPHA, beta=self.BETA)
        self.load_accum  = Load()
        self.fuse_accum  = Add(alpha=self.DECAY, beta=1.0)
        self.save_accum  = Save()
        self.output_func = topK()
        # Cache-side ops
        self.k_mean      = CMean(dim=1)
        self.k_norm      = CL2Norm(dim=1)
        self.init_accum  = CFill(alpha=0.0)

    def create_cache(self, block_size: int, head_dim: int):
        return {
            "centroids": (1, head_dim),
            "k_norm":    (1, head_dim),
            "accum":     (1, 1),
        }

    def forward_cache(self, cache: Dict[str, torch.Tensor], loc: torch.Tensor, ctx: ContextBase):
        self.k_mean(cache["k"], cache["centroids"], loc=loc, ctx=ctx)
        self.k_norm(cache["k"], cache["k_norm"],    loc=loc, ctx=ctx)
        self.init_accum(cache["accum"], loc=loc, ctx=ctx)

    def forward_indexer(self, q: torch.Tensor, o: torch.Tensor, cache: Dict[str, torch.Tensor], ctx: ContextBase):
        q_mean   = self.q_mean(q, ctx=ctx)                                 # [1, 1, D]
        dot      = self.gemm_dot(q_mean, cache["centroids"], ctx=ctx)      # [S, 1, 1]
        per_feat = self.mul_norm(q_mean, cache["k_norm"], ctx=ctx)         # [S, 1, D]
        sum_pf   = self.sum_norm(per_feat, ctx=ctx)                        # [S, 1, 1]
        current  = self.blend(dot, sum_pf, ctx=ctx)                        # α·dot + β·sum_pf
        prev     = self.load_accum(cache["accum"], ctx=ctx)                # [S, 1, 1]
        mixed    = self.fuse_accum(prev, current, ctx=ctx)                 # decay·prev + current
        self.save_accum(mixed, cache["accum"], ctx=ctx)
        self.output_func(mixed, o, ctx=ctx)
