import torch
from typing import Dict

from vortex_torch.flow import vFlow, register
from vortex_torch.indexer import (
    topK, approxTopK, GeMV, Softmax, Max, Sum, GeMM,
    Maximum, Multiply, Add, L2Norm, Save, Load, Mean, MaskSlice, Kron,
)
from vortex_torch.cache import (
    Mean as CMean, Max as CMax, Min as CMin, L2Norm as CL2Norm,
    Fill as CFill, MaxInterleave as CMaxInterleave, MinInterleave as CMinInterleave,
)
from vortex_torch.abs import ContextBase


@register("venergy_gated_centroid_sub")
class VEnergyGatedCentroid(vFlow):
    """
    Score pages by centroid dot-product gated by the mean token magnitude
    of the value block.  Pages whose values carry little energy are muted
    even when the key centroid aligns with the query.

    Novelty: v-block energy as a multiplicative gate on the key-centroid
    score via CL2Norm(dim=2) → CMean(dim=1) chain in forward_cache.
    §16.4 — value-side signal exploitation.
    """

    def __init__(self):
        super().__init__()
        # indexer-side
        self.q_mean = Mean(dim=1)
        self.gemm = GeMM()
        self.gate = Multiply()
        self.output_func = topK()

        # cache-side
        self.k_mean = CMean(dim=1)
        self.v_tok_norm = CL2Norm(dim=2)   # [1, block_size, D] → [1, block_size, 1]
        self.v_energy = CMean(dim=1)        # [1, block_size, 1] → [1, 1, 1]

    def create_cache(self, block_size: int, head_dim: int):
        return {
            "centroids": (1, head_dim),
            "v_energy":  (1, 1),
        }

    def forward_cache(self, cache, loc, ctx):
        self.k_mean(cache["k"], cache["centroids"], loc=loc, ctx=ctx)
        v_tok = self.v_tok_norm(cache["v"], None, loc=loc, ctx=ctx)   # [1, block_size, 1]
        self.v_energy(v_tok, cache["v_energy"], loc=loc, ctx=ctx)      # [1, 1, 1]

    def forward_indexer(self, q, o, cache, ctx):
        q_mean = self.q_mean(q, ctx=ctx)                         # [1, 1, D]
        dot = self.gemm(q_mean, cache["centroids"], ctx=ctx)     # [S, 1, 1]
        score = self.gate(dot, cache["v_energy"], ctx=ctx)       # [S, 1, 1]
        self.output_func(score, o, ctx=ctx)

