"""id13 — Multi-head × multi-band Kron grid + Min aggregation.

Novelty (op-set thread): the framework's `Kron(dim=1)` row-Kron
op produces an `[S, H_q * G, D]` cross product of all query-head ×
sub-block mini-centroid pairings, sharing the feature axis. We then
reduce the feature axis with `Sum(dim=2)`, giving a per-pair score, and
take `Min(dim=1)` over the H_q * G grid: a page is selected only if
the *worst* (head, sub-block) pairing still has a high score. Strict
multi-axis consensus — neither §14.4 (Max-of-Kron) nor any paper has
tried Min-of-Kron at this granularity.
"""

import torch
from typing import Dict

from vortex_torch.flow import vFlow, register
from vortex_torch.indexer import topK, Kron, Sum, Min
from vortex_torch.cache import MeanInterleave as CMeanInterleave
from vortex_torch.abs import ContextBase


_GROUP = 4   # block_size=16 / 4 → 4 mini-centroids per block


@register("claude_opus_4_7_innovate_0_id13_cls")
class Id13Cls(vFlow):
    def __init__(self):
        super().__init__()
        # Row-Kron over the head-axis × sub-block-axis (D axis is elementwise,
        # so it broadcasts since q has [1, H_q, D] and mini has [S, G, D]).
        self.kron = Kron(dim=1)
        self.sum_d = Sum(dim=2)
        self.min_grid = Min(dim=1)
        self.output_func = topK()

        self.k_mini = CMeanInterleave(dim=1, k=_GROUP)

    def create_cache(self, block_size: int, head_dim: int):
        return {"mini_centroids": (block_size // _GROUP, head_dim)}

    def forward_cache(self, cache, loc, ctx: ContextBase):
        self.k_mini(cache["k"], cache["mini_centroids"], loc=loc, ctx=ctx)

    def forward_indexer(self, q, o, cache, ctx: ContextBase):
        # q: [1, H_q, D]  ;  mini_centroids: [S, G, D]
        # Kron(dim=1): [S, H_q * G, D]
        grid = self.kron(q, cache["mini_centroids"], ctx=ctx)
        score_d = self.sum_d(grid, ctx=ctx)                           # [S, H_q*G, 1]
        aggr = self.min_grid(score_d, ctx=ctx)                        # [S, 1, 1]
        self.output_func(aggr, o, ctx=ctx)
