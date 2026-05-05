"""id4 — Per-token v-norm gate (block_size, 1) field.

Novelty (op-set thread): the framework supports `(block_size, 1)`-shaped
cache fields — one scalar per token-slot per block. We exploit this to
store the per-token L2 norm of values via `CL2Norm(dim=2)(cache["v"])`
giving `[1, block_size, 1]` per block, then on the indexer side
`Load` it (PAGED → RAGGED), reduce with `Mean(dim=1)` over block_size
to a per-page scalar, and use it as a multiplicative gate on the
centroid score. Pages whose values carry strong signal across the
block win — a v-routing mass term no paper here computes at sub-block
granularity.
"""

import torch
from typing import Dict

from vortex_torch.flow import vFlow, register
from vortex_torch.indexer import topK, Mean, GeMM, Multiply, Load
from vortex_torch.cache import Mean as CMean, L2Norm as CL2Norm
from vortex_torch.abs import ContextBase


@register("claude_opus_4_7_innovate_0_id4_cls")
class Id4Cls(vFlow):
    def __init__(self):
        super().__init__()
        self.q_mean = Mean(dim=1)
        self.gemm = GeMM()
        self.load_vtn = Load()                # PAGED → RAGGED
        self.mean_vmass = Mean(dim=1)         # collapse block_size → per-page scalar
        self.gate = Multiply()
        self.output_func = topK()

        self.k_mean = CMean(dim=1)
        self.v_token_norm = CL2Norm(dim=2)    # [1, block_size, D] → [1, block_size, 1]

    def create_cache(self, block_size: int, head_dim: int):
        return {
            "centroids":    (1, head_dim),
            "v_token_norm": (block_size, 1),
        }

    def forward_cache(self, cache, loc, ctx: ContextBase):
        self.k_mean(cache["k"], cache["centroids"], loc=loc, ctx=ctx)
        self.v_token_norm(cache["v"], cache["v_token_norm"], loc=loc, ctx=ctx)

    def forward_indexer(self, q, o, cache, ctx: ContextBase):
        qm = self.q_mean(q, ctx=ctx)                                 # [1, 1, D]
        centroid_score = self.gemm(qm, cache["centroids"], ctx=ctx)  # [S, 1, 1]
        # Bring the PAGED v_token_norm into a RAGGED workspace before reducing.
        vtn = self.load_vtn(cache["v_token_norm"], ctx=ctx)          # [S, block_size, 1]
        vmass = self.mean_vmass(vtn, ctx=ctx)                         # [S, 1, 1]
        score = self.gate(centroid_score, vmass, ctx=ctx)             # per-page non-monotonic
        self.output_func(score, o, ctx=ctx)
