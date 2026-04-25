import torch
from typing import Dict

from vortex_torch.flow import vFlow, register
from vortex_torch.indexer import topK, Mean, GeMM
from vortex_torch.cache import Mean as CMean
from vortex_torch.abs import ContextBase


@register("kimi_v0_cls")
class KimiV0Cls(vFlow):
    def __init__(self):
        super().__init__()
        # Indexer-side ops: cheap centroid routing.
        self.q_mean = Mean(dim=1)
        self.score_gemm = GeMM()
        self.output_func = topK()

        # Cache-side ops: one key centroid per block.
        self.k_mean = CMean(dim=1)

    def create_cache(self, block_size: int, head_dim: int):
        return {
            "centroids": (1, head_dim),
        }

    def forward_cache(
        self,
        cache: Dict[str, torch.Tensor],
        loc: torch.Tensor,
        ctx: ContextBase,
    ):
        self.k_mean(cache["k"], cache["centroids"], loc=loc, ctx=ctx)

    def forward_indexer(
        self,
        q: torch.Tensor,
        o: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        ctx: ContextBase,
    ):
        q_mean = self.q_mean(q, ctx=ctx)
        score = self.score_gemm(q_mean, cache["centroids"], ctx=ctx)
        self.output_func(score, o, ctx=ctx)
