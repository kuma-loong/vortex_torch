import torch
from typing import Dict

from vortex_torch.flow import vFlow, register
from vortex_torch.abs import ContextBase
from vortex_torch.indexer import (
    topK, Mean, Add, Multiply, Sum, Max,
)
from vortex_torch.cache import (
    MaxInterleave as CMaxInterleave, MinInterleave as CMinInterleave,
)

# Each block is divided into sub-blocks of this many tokens
_SUB_GROUP = 4


@register("claude_sonnet_4_6_innovate_0_id13_cls")
class QueryWeightedSpread(vFlow):
    """
    Score pages by the query-aligned key spread: for each sub-block
    within a page, compute (k_max - k_min) as the feature-wise range,
    then take the dot product of the query with this range vector.

    Pages where the query direction aligns with the highest-variance key
    features score highest.  This favours heterogeneous blocks containing
    the extreme key vectors most relevant to the current query.

    Novelty: query-weighted key spread via (MaxInterleave - MinInterleave)
    as the page importance signal; no paper uses within-block key range
    (aligned to the query direction) for routing.
    Op-set derived — exploits CMaxInterleave/CMinInterleave asymmetry.
    """

    def __init__(self):
        super().__init__()
        # indexer-side
        self.q_mean = Mean(dim=1)
        self.span = Add(alpha=1.0, beta=-1.0)        # k_max_sub - k_min_sub
        self.wt_span = Multiply()
        self.sum_d = Sum(dim=2)
        self.max_sub = Max(dim=1)
        self.output_func = topK()

        # cache-side  (group size = _SUB_GROUP → block_size//_SUB_GROUP sub-blocks)
        self.k_max_sub = CMaxInterleave(dim=1, k=_SUB_GROUP)
        self.k_min_sub = CMinInterleave(dim=1, k=_SUB_GROUP)

    def create_cache(self, block_size: int, head_dim: int):
        n_sub = block_size // _SUB_GROUP
        return {
            "k_max_sub": (n_sub, head_dim),
            "k_min_sub": (n_sub, head_dim),
        }

    def forward_cache(self, cache, loc, ctx):
        self.k_max_sub(cache["k"], cache["k_max_sub"], loc=loc, ctx=ctx)
        self.k_min_sub(cache["k"], cache["k_min_sub"], loc=loc, ctx=ctx)

    def forward_indexer(self, q, o, cache, ctx):
        q_mean = self.q_mean(q, ctx=ctx)                                # [1, 1, D]
        # k_max_sub, k_min_sub: [S, n_sub, D]
        spread = self.span(cache["k_max_sub"], cache["k_min_sub"], ctx=ctx)   # [S, n_sub, D]
        wt = self.wt_span(q_mean, spread, ctx=ctx)                      # [S, n_sub, D]
        scores = self.sum_d(wt, ctx=ctx)                                # [S, n_sub, 1]
        score = self.max_sub(scores, ctx=ctx)                           # [S, 1,     1]
        self.output_func(score, o, ctx=ctx)
