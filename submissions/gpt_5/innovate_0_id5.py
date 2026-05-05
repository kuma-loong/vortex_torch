import torch
from typing import Dict

from vortex_torch.abs import ContextBase
from vortex_torch.flow import vFlow, register
from vortex_torch.indexer import GeMM, Max, Mean, topK
from vortex_torch.cache import MeanInterleave as CMeanInterleave


@register("gpt_5_innovate_0_id5_cls")
class MiniCentroidPeakRouter(vFlow):
    def __init__(self):
        super().__init__()
        self.q_mean = Mean(dim=1)
        self.dot = GeMM()
        self.pick_peak = Max(dim=1)
        self.output_func = topK()
        self.k_mini = CMeanInterleave(dim=1, k=4)

    def create_cache(self, block_size: int, head_dim: int):
        return {"mini_centroids": (block_size // 4, head_dim)}

    def forward_cache(self, cache: Dict[str, torch.Tensor], loc: torch.Tensor, ctx: ContextBase):
        self.k_mini(cache["k"], cache["mini_centroids"], loc=loc, ctx=ctx)

    def forward_indexer(self, q: torch.Tensor, o: torch.Tensor, cache: Dict[str, torch.Tensor], ctx: ContextBase):
        q_mean = self.q_mean(q, ctx=ctx)
        per_mini = self.dot(q_mean, cache["mini_centroids"], ctx=ctx)
        score = self.pick_peak(per_mini, ctx=ctx)
        self.output_func(score, o, ctx=ctx)
