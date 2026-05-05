import torch
from typing import Dict

from vortex_torch.abs import ContextBase
from vortex_torch.flow import vFlow, register
from vortex_torch.indexer import Conv1d, GeMM, Mean, topK
from vortex_torch.cache import Mean as CMean


@register("gpt_5_innovate_0_id18_cls")
class SmoothedCentroidSecondTier(vFlow):
    def __init__(self):
        super().__init__()
        self.q_mean = Mean(dim=1)
        self.dot = GeMM()
        self.smooth = Conv1d(weight=[[[0.25]], [[0.5]], [[0.25]]], dim=0)
        self.output_func = topK()
        self.k_mean = CMean(dim=1)

    def create_cache(self, block_size: int, head_dim: int):
        return {"centroids": (1, head_dim)}

    def forward_cache(self, cache: Dict[str, torch.Tensor], loc: torch.Tensor, ctx: ContextBase):
        self.k_mean(cache["k"], cache["centroids"], loc=loc, ctx=ctx)

    def forward_indexer(self, q: torch.Tensor, o: torch.Tensor, cache: Dict[str, torch.Tensor], ctx: ContextBase):
        q_mean = self.q_mean(q, ctx=ctx)
        raw = self.dot(q_mean, cache["centroids"], ctx=ctx)
        score = self.smooth(raw, ctx=ctx)
        self.output_func(score, o, ctx=ctx)
