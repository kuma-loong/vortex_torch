import torch
from typing import Dict

from vortex_torch.abs import ContextBase
from vortex_torch.flow import vFlow, register
from vortex_torch.indexer import GeMM, Max, Mean, Multiply, topK
from vortex_torch.cache import Mean as CMean, L2Norm as CL2Norm


@register("gpt_5_innovate_0_id15_cls")
class HeadDispersionValueScale(vFlow):
    def __init__(self):
        super().__init__()
        self.head_scores = GeMM()
        self.max_head = Max(dim=2)
        self.q_mean = Mean(dim=1)
        self.value_energy = GeMM()
        self.scale = Multiply()
        self.output_func = topK()
        self.k_mean = CMean(dim=1)
        self.v_norm = CL2Norm(dim=1)

    def create_cache(self, block_size: int, head_dim: int):
        return {"centroids": (1, head_dim), "v_norm": (1, head_dim)}

    def forward_cache(self, cache: Dict[str, torch.Tensor], loc: torch.Tensor, ctx: ContextBase):
        self.k_mean(cache["k"], cache["centroids"], loc=loc, ctx=ctx)
        self.v_norm(cache["v"], cache["v_norm"], loc=loc, ctx=ctx)

    def forward_indexer(self, q: torch.Tensor, o: torch.Tensor, cache: Dict[str, torch.Tensor], ctx: ContextBase):
        per_head = self.head_scores(q, cache["centroids"], ctx=ctx)
        raw = self.max_head(per_head, ctx=ctx)
        q_mean = self.q_mean(q, ctx=ctx)
        value = self.value_energy(q_mean, cache["v_norm"], ctx=ctx)
        score = self.scale(raw, value, ctx=ctx)
        self.output_func(score, o, ctx=ctx)
