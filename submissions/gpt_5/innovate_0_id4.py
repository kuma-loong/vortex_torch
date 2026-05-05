import torch
from typing import Dict

from vortex_torch.abs import ContextBase
from vortex_torch.flow import vFlow, register
from vortex_torch.indexer import Add, Load, Mean, Multiply, Save, Sum, topK
from vortex_torch.cache import Fill as CFill, Mean as CMean, L2Norm as CL2Norm


@register("gpt_5_innovate_0_id4_cls")
class RunningValueWeightedScore(vFlow):
    def __init__(self):
        super().__init__()
        self.q_mean = Mean(dim=1)
        self.align = Multiply()
        self.value_scale = Multiply()
        self.sum_d = Sum(dim=2)
        self.load_score = Load()
        self.fuse = Add(alpha=0.7, beta=1.0)
        self.save_score = Save()
        self.output_func = topK()
        self.k_mean = CMean(dim=1)
        self.v_norm = CL2Norm(dim=1)
        self.init_running = CFill(alpha=0.0)

    def create_cache(self, block_size: int, head_dim: int):
        return {"centroids": (1, head_dim), "v_norm": (1, head_dim), "running_score": (1, 1)}

    def forward_cache(self, cache: Dict[str, torch.Tensor], loc: torch.Tensor, ctx: ContextBase):
        self.k_mean(cache["k"], cache["centroids"], loc=loc, ctx=ctx)
        self.v_norm(cache["v"], cache["v_norm"], loc=loc, ctx=ctx)
        self.init_running(cache["running_score"], loc=loc, ctx=ctx)

    def forward_indexer(self, q: torch.Tensor, o: torch.Tensor, cache: Dict[str, torch.Tensor], ctx: ContextBase):
        q_mean = self.q_mean(q, ctx=ctx)
        raw_vec = self.align(q_mean, cache["centroids"], ctx=ctx)
        value_vec = self.value_scale(raw_vec, cache["v_norm"], ctx=ctx)
        current = self.sum_d(value_vec, ctx=ctx)
        last = self.load_score(cache["running_score"], ctx=ctx)
        running = self.fuse(last, current, ctx=ctx)
        self.save_score(running, cache["running_score"], ctx=ctx)
        self.output_func(running, o, ctx=ctx)
