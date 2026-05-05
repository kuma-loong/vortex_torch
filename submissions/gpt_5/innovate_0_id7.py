import torch
from typing import Dict

from vortex_torch.abs import ContextBase
from vortex_torch.flow import vFlow, register
from vortex_torch.indexer import Kron, Max, Maximum, Sum, topK
from vortex_torch.cache import MaxInterleave as CMaxInterleave, MinInterleave as CMinInterleave


@register("gpt_5_innovate_0_id7_cls")
class InterleavedEnvelopeKron(vFlow):
    def __init__(self):
        super().__init__()
        self.kron_max = Kron(dim=1)
        self.kron_min = Kron(dim=1)
        self.envelope = Maximum()
        self.sum_d = Sum(dim=2)
        self.max_axis = Max(dim=1)
        self.output_func = topK()
        self.k_maxi = CMaxInterleave(dim=1, k=4)
        self.k_mini = CMinInterleave(dim=1, k=4)

    def create_cache(self, block_size: int, head_dim: int):
        return {"maxi": (block_size // 4, head_dim), "mini": (block_size // 4, head_dim)}

    def forward_cache(self, cache: Dict[str, torch.Tensor], loc: torch.Tensor, ctx: ContextBase):
        self.k_maxi(cache["k"], cache["maxi"], loc=loc, ctx=ctx)
        self.k_mini(cache["k"], cache["mini"], loc=loc, ctx=ctx)

    def forward_indexer(self, q: torch.Tensor, o: torch.Tensor, cache: Dict[str, torch.Tensor], ctx: ContextBase):
        s_max = self.kron_max(q, cache["maxi"], ctx=ctx)
        s_min = self.kron_min(q, cache["mini"], ctx=ctx)
        env = self.envelope(s_max, s_min, ctx=ctx)
        score_axis = self.sum_d(env, ctx=ctx)
        score = self.max_axis(score_axis, ctx=ctx)
        self.output_func(score, o, ctx=ctx)
