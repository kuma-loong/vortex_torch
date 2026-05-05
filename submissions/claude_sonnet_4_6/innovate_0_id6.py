import torch
from typing import Dict

from vortex_torch.flow import vFlow, register
from vortex_torch.abs import ContextBase
from vortex_torch.indexer import (
    topK, Kron, Sum, Max,
)
from vortex_torch.cache import (
    MeanInterleave as CMeanInterleave,
)

# Number of mini-centroids per block (group size = 4 tokens → block_size/4 groups)
_GROUPS = 4


@register("claude_sonnet_4_6_innovate_0_id6_cls")
class MiniCentroidKronHead(vFlow):
    """
    Divide each page into GROUPS sub-blocks and compute per-sub-block
    mean centroids.  Use Kron(dim=1) to form all (query-head × sub-block)
    pairings, sum over features, and take the max pairing per page.

    This lets different query heads fire on different sub-blocks within
    a page, recovering locality patterns that a single centroid misses.

    Novelty: Kron(dim=1) to create head × mini-centroid score grid;
    no existing flow pairs multi-head queries with multi-resolution
    sub-block centroids this way.
    §16.2 — untried Kron usage.
    """

    def __init__(self):
        super().__init__()
        # indexer-side
        self.kron = Kron(dim=1)
        self.sum_d = Sum(dim=2)
        self.max_pairs = Max(dim=1)
        self.output_func = topK()

        # cache-side  (group_size = 4 → block_size // 4 sub-blocks)
        self.mini_mean = CMeanInterleave(dim=1, k=_GROUPS)

    def create_cache(self, block_size: int, head_dim: int):
        return {
            "mini_centroids": (block_size // _GROUPS, head_dim),
        }

    def forward_cache(self, cache, loc, ctx):
        self.mini_mean(cache["k"], cache["mini_centroids"], loc=loc, ctx=ctx)

    def forward_indexer(self, q, o, cache, ctx):
        # q: [1, H_q, D],  mini_centroids: [S, block_size//_GROUPS, D]
        # Kron(dim=1) → [S, H_q * (block_size//_GROUPS), D]
        kron_out = self.kron(q, cache["mini_centroids"], ctx=ctx)
        scores = self.sum_d(kron_out, ctx=ctx)               # [S, H_q*(bs//_GROUPS), 1]
        score = self.max_pairs(scores, ctx=ctx)              # [S, 1, 1]
        self.output_func(score, o, ctx=ctx)
