import torch
from typing import Dict

from vortex_torch.flow import vFlow, register
from vortex_torch.abs import ContextBase
from vortex_torch.indexer import (
    topK, Multiply, Sum, Max,
)
from vortex_torch.cache import (
    Min as CMin,
)


@register("claude_sonnet_4_6_innovate_0_id18_cls")
class MinEnvelopeConservativeScoring(vFlow):
    """
    Score pages by the query dot product with the per-feature MINIMUM of
    keys in the block.  This is a conservative lower-bound score: a page
    can only achieve a high score if EVERY token in the block has a large
    component in the query direction (the minimum envelope is high).

    Inversion of QUEST (which upper-bounds via the max envelope): the min
    score selects pages that are uniformly relevant to the query, not
    merely pages that contain at least one relevant extreme token.

    Novelty: exclusive use of the min key envelope as the scoring signal —
    the inverse conservativeness criterion versus QUEST's max-envelope.
    §16.3 — inversion of the QUEST max-envelope assumption.
    """

    def __init__(self):
        super().__init__()
        # indexer-side
        self.mul = Multiply()
        self.sum_d = Sum(dim=2)
        self.max_h = Max(dim=1)
        self.output_func = topK()

        # cache-side
        self.k_min = CMin(dim=1)

    def create_cache(self, block_size: int, head_dim: int):
        return {
            "k_min": (1, head_dim),
        }

    def forward_cache(self, cache, loc, ctx):
        self.k_min(cache["k"], cache["k_min"], loc=loc, ctx=ctx)

    def forward_indexer(self, q, o, cache, ctx):
        # q: [1, H_q, D],  k_min: [S, 1, D]
        prod = self.mul(q, cache["k_min"], ctx=ctx)    # [S, H_q, D]
        score_h = self.sum_d(prod, ctx=ctx)            # [S, H_q, 1]
        score = self.max_h(score_h, ctx=ctx)           # [S, 1,   1]
        self.output_func(score, o, ctx=ctx)
