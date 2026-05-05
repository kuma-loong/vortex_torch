import torch
from typing import Dict

from vortex_torch.abs import ContextBase
from vortex_torch.flow import vFlow, register
from vortex_torch.indexer import GeMM, Mean, topK
from vortex_torch.cache import L2Norm as CL2Norm


@register("gpt_5_innovate_0_id17_cls")
class KeyOnlyEnergyRouter(vFlow):
    def __init__(self):
        super().__init__()
        self.q_mean = Mean(dim=1)
        self.key_energy = GeMM()
        self.output_func = topK()
        self.k_norm = CL2Norm(dim=1)

    def create_cache(self, block_size: int, head_dim: int):
        return {"k_norm": (1, head_dim)}

    def forward_cache(self, cache: Dict[str, torch.Tensor], loc: torch.Tensor, ctx: ContextBase):
        self.k_norm(cache["k"], cache["k_norm"], loc=loc, ctx=ctx)

    def forward_indexer(self, q: torch.Tensor, o: torch.Tensor, cache: Dict[str, torch.Tensor], ctx: ContextBase):
        q_mean = self.q_mean(q, ctx=ctx)
        score = self.key_energy(q_mean, cache["k_norm"], ctx=ctx)
        self.output_func(score, o, ctx=ctx)
