import torch
from typing import Dict

from vortex_torch.abs import ContextBase
from vortex_torch.flow import vFlow, register
from vortex_torch.indexer import Add, GeMM, Mean, topK
from vortex_torch.cache import Mean as CMean, Min as CMin


@register("gpt_5_innovate_0_id11_cls")
class MinimumEnvelopePenalty(vFlow):
    def __init__(self):
        super().__init__()
        self.q_mean = Mean(dim=1)
        self.dot_centroid = GeMM()
        self.dot_min = GeMM()
        self.subtract_min = Add(alpha=1.0, beta=-0.5)
        self.output_func = topK()
        self.k_mean = CMean(dim=1)
        self.k_min = CMin(dim=1)

    def create_cache(self, block_size: int, head_dim: int):
        return {"centroids": (1, head_dim), "min": (1, head_dim)}

    def forward_cache(self, cache: Dict[str, torch.Tensor], loc: torch.Tensor, ctx: ContextBase):
        self.k_mean(cache["k"], cache["centroids"], loc=loc, ctx=ctx)
        self.k_min(cache["k"], cache["min"], loc=loc, ctx=ctx)

    def forward_indexer(self, q: torch.Tensor, o: torch.Tensor, cache: Dict[str, torch.Tensor], ctx: ContextBase):
        q_mean = self.q_mean(q, ctx=ctx)
        raw = self.dot_centroid(q_mean, cache["centroids"], ctx=ctx)
        penalty = self.dot_min(q_mean, cache["min"], ctx=ctx)
        score = self.subtract_min(raw, penalty, ctx=ctx)
        self.output_func(score, o, ctx=ctx)
