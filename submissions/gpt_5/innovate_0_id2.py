import torch
from typing import Dict

from vortex_torch.abs import ContextBase
from vortex_torch.flow import vFlow, register
from vortex_torch.indexer import Add, GeMM, Mean, topK
from vortex_torch.cache import Abs as CAbs, Mean as CMean


@register("gpt_5_innovate_0_id2_cls")
class SignedMagnitudeDisagreement(vFlow):
    def __init__(self):
        super().__init__()
        self.q_mean = Mean(dim=1)
        self.dot_signed = GeMM()
        self.dot_abs = GeMM()
        self.penalize_magnitude = Add(alpha=1.0, beta=-0.25)
        self.output_func = topK()
        self.k_mean = CMean(dim=1)
        self.k_abs = CAbs()
        self.abs_mean = CMean(dim=1)

    def create_cache(self, block_size: int, head_dim: int):
        return {"centroids": (1, head_dim), "abs_centroids": (1, head_dim)}

    def forward_cache(self, cache: Dict[str, torch.Tensor], loc: torch.Tensor, ctx: ContextBase):
        self.k_mean(cache["k"], cache["centroids"], loc=loc, ctx=ctx)
        abs_k = self.k_abs(cache["k"], None, loc=loc, ctx=ctx)
        self.abs_mean(abs_k, cache["abs_centroids"], loc=loc, ctx=ctx)

    def forward_indexer(self, q: torch.Tensor, o: torch.Tensor, cache: Dict[str, torch.Tensor], ctx: ContextBase):
        q_mean = self.q_mean(q, ctx=ctx)
        signed = self.dot_signed(q_mean, cache["centroids"], ctx=ctx)
        magnitude = self.dot_abs(q_mean, cache["abs_centroids"], ctx=ctx)
        score = self.penalize_magnitude(signed, magnitude, ctx=ctx)
        self.output_func(score, o, ctx=ctx)
