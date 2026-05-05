import torch
from typing import Dict

from vortex_torch.abs import ContextBase
from vortex_torch.flow import vFlow, register
from vortex_torch.indexer import Max, Maximum, Multiply, Sum, topK
from vortex_torch.cache import Max as CMax, Min as CMin, L2Norm as CL2Norm


@register("gpt_5_innovate_0_id1_cls")
class ValueScaledQuest(vFlow):
    def __init__(self):
        super().__init__()
        self.mul_max = Multiply()
        self.mul_min = Multiply()
        self.envelope = Maximum()
        self.value_scale = Multiply()
        self.sum_d = Sum(dim=2)
        self.max_h = Max(dim=1)
        self.output_func = topK()
        self.k_max = CMax(dim=1)
        self.k_min = CMin(dim=1)
        self.v_norm = CL2Norm(dim=1)

    def create_cache(self, block_size: int, head_dim: int):
        return {"max": (1, head_dim), "min": (1, head_dim), "v_norm": (1, head_dim)}

    def forward_cache(self, cache: Dict[str, torch.Tensor], loc: torch.Tensor, ctx: ContextBase):
        self.k_max(cache["k"], cache["max"], loc=loc, ctx=ctx)
        self.k_min(cache["k"], cache["min"], loc=loc, ctx=ctx)
        self.v_norm(cache["v"], cache["v_norm"], loc=loc, ctx=ctx)

    def forward_indexer(self, q: torch.Tensor, o: torch.Tensor, cache: Dict[str, torch.Tensor], ctx: ContextBase):
        s_max = self.mul_max(q, cache["max"], ctx=ctx)
        s_min = self.mul_min(q, cache["min"], ctx=ctx)
        env = self.envelope(s_max, s_min, ctx=ctx)
        weighted = self.value_scale(env, cache["v_norm"], ctx=ctx)
        score_h = self.sum_d(weighted, ctx=ctx)
        score = self.max_h(score_h, ctx=ctx)
        self.output_func(score, o, ctx=ctx)
