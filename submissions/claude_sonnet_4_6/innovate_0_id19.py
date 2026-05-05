import torch
from typing import Dict

from vortex_torch.flow import vFlow, register
from vortex_torch.abs import ContextBase
from vortex_torch.indexer import (
    topK, Multiply, Maximum, Sum, Max, Add, Sigmoid,
)
from vortex_torch.cache import (
    Max as CMax, Min as CMin,
)


@register("claude_sonnet_4_6_innovate_0_id19_cls")
class SigmoidGatedQuestScoring(vFlow):
    """
    Apply sigmoid-gating to the QUEST envelope before the final sum.

    In standard QUEST, max(q·k_max, q·k_min) is summed linearly.  Here,
    the difference (q·k_max − q·k_min) modulates a sigmoid gate applied
    to the QUEST envelope: when max dominates (same sign as q), the gate
    is close to 1; when min dominates (q and max are anti-directional),
    the gate suppresses that feature.

    This creates a feature-wise soft-mask on the QUEST score: features
    where the max envelope is the dominant term are trusted; features
    where the envelopes are nearly symmetric are down-weighted.

    Novelty: per-feature Sigmoid gate derived from the QUEST envelope
    difference, applied before summing the QUEST score — no paper uses
    a non-linear gate on individual QUEST feature contributions.
    Op-set derived (Sigmoid on envelope difference as a soft-select gate).
    """

    def __init__(self):
        super().__init__()
        # indexer-side
        self.mul_max = Multiply()
        self.mul_min = Multiply()
        self.quest = Maximum()                 # max(q·k_max, q·k_min)
        self.diff = Add(alpha=1.0, beta=-1.0)  # (q·k_max) − (q·k_min)
        self.gate = Sigmoid(alpha=0.0, beta=1.0)   # σ(diff): high when max dominates
        self.apply_gate = Multiply()
        self.sum_d = Sum(dim=2)
        self.max_h = Max(dim=1)
        self.output_func = topK()

        # cache-side
        self.k_max = CMax(dim=1)
        self.k_min = CMin(dim=1)

    def create_cache(self, block_size: int, head_dim: int):
        return {
            "k_max": (1, head_dim),
            "k_min": (1, head_dim),
        }

    def forward_cache(self, cache, loc, ctx):
        self.k_max(cache["k"], cache["k_max"], loc=loc, ctx=ctx)
        self.k_min(cache["k"], cache["k_min"], loc=loc, ctx=ctx)

    def forward_indexer(self, q, o, cache, ctx):
        s_max = self.mul_max(q, cache["k_max"], ctx=ctx)     # [S, H_q, D]
        s_min = self.mul_min(q, cache["k_min"], ctx=ctx)     # [S, H_q, D]
        quest_s = self.quest(s_max, s_min, ctx=ctx)          # [S, H_q, D]
        envelope_diff = self.diff(s_max, s_min, ctx=ctx)     # [S, H_q, D]
        gate_w = self.gate(envelope_diff, ctx=ctx)           # [S, H_q, D] ∈ (0,1)
        gated = self.apply_gate(quest_s, gate_w, ctx=ctx)    # [S, H_q, D]
        score_h = self.sum_d(gated, ctx=ctx)                 # [S, H_q, 1]
        score = self.max_h(score_h, ctx=ctx)                 # [S, 1,   1]
        self.output_func(score, o, ctx=ctx)
