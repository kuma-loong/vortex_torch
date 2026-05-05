"""id9 — Pure magnitude routing (no q, no direction).

Novelty (§16.4 first-principles falsification): every routing function
in the literature computes `q · summary` — a directional alignment
score. We strip the directional signal entirely: the score is the
per-page average of per-token L2 norms of the keys, and `q` is unused.
This tests whether *any* directional structure matters at all, or
whether pages with intrinsically-large keys carry enough signal that
direction can be ignored. A serious accuracy collapse here would
falsify the "magnitude only" inversion hypothesis hinted at by
MagicPIG/RetrievalAttention's OOD findings.

Implementation note: indexer-side `Mean` doesn't dispatch on a raw
PAGED cache field, so we use `Load` to bring `cache["k_token_norm"]`
into RAGGED format before reducing.
"""

import torch
from typing import Dict

from vortex_torch.flow import vFlow, register
from vortex_torch.indexer import topK, Mean, Load
from vortex_torch.cache import L2Norm as CL2Norm
from vortex_torch.abs import ContextBase


@register("claude_opus_4_7_innovate_0_id9_cls")
class Id9Cls(vFlow):
    def __init__(self):
        super().__init__()
        self.load_ktn = Load()                       # PAGED → RAGGED
        self.mean_per_page = Mean(dim=1)             # collapse block_size axis
        self.output_func = topK()

        # CL2Norm(dim=2): [1, block_size, D] → [1, block_size, 1]
        self.k_token_norm = CL2Norm(dim=2)

    def create_cache(self, block_size: int, head_dim: int):
        return {"k_token_norm": (block_size, 1)}

    def forward_cache(self, cache, loc, ctx: ContextBase):
        self.k_token_norm(cache["k"], cache["k_token_norm"], loc=loc, ctx=ctx)

    def forward_indexer(self, q, o, cache, ctx: ContextBase):
        # q is intentionally unused — we route by intrinsic key magnitude only.
        ktn = self.load_ktn(cache["k_token_norm"], ctx=ctx)          # [S, block_size, 1]
        score = self.mean_per_page(ktn, ctx=ctx)                      # [S, 1, 1]
        self.output_func(score, o, ctx=ctx)
