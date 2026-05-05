import torch
from typing import Dict

from vortex_torch.abs import ContextBase
from vortex_torch.flow import vFlow, register
from vortex_torch.indexer import Add, GeMM, Mean, topK
from vortex_torch.cache import Mean as CMean


@register("gpt_5_innovate_0_id12_cls")
class ValueCentroidAlignment(vFlow):
    def __init__(self):
        super().__init__()
        self.q_mean = Mean(dim=1)
        self.key_dot = GeMM()
        self.value_dot = GeMM()
        self.mix = Add(alpha=0.75, beta=0.25)
        self.output_func = topK()
        self.k_mean = CMean(dim=1)
        self.v_mean = CMean(dim=1)

    def create_cache(self, block_size: int, head_dim: int):
        return {"centroids": (1, head_dim), "v_centroids": (1, head_dim)}

    def forward_cache(self, cache: Dict[str, torch.Tensor], loc: torch.Tensor, ctx: ContextBase):
        self.k_mean(cache["k"], cache["centroids"], loc=loc, ctx=ctx)
        self.v_mean(cache["v"], cache["v_centroids"], loc=loc, ctx=ctx)

    def forward_indexer(self, q: torch.Tensor, o: torch.Tensor, cache: Dict[str, torch.Tensor], ctx: ContextBase):
        q_mean = self.q_mean(q, ctx=ctx)
        key_score = self.key_dot(q_mean, cache["centroids"], ctx=ctx)
        value_score = self.value_dot(q_mean, cache["v_centroids"], ctx=ctx)
        score = self.mix(key_score, value_score, ctx=ctx)
        self.output_func(score, o, ctx=ctx)
