import torch
from typing import Dict

from vortex_torch.abs import ContextBase
from vortex_torch.flow import vFlow, register
from vortex_torch.indexer import Abs, Add, GeMM, Load, Mean, Save, topK
from vortex_torch.cache import Fill as CFill, Mean as CMean


@register("gpt_5_innovate_0_id14_cls")
class RunningSurpriseScore(vFlow):
    def __init__(self):
        super().__init__()
        self.q_mean = Mean(dim=1)
        self.dot = GeMM()
        self.seq_mean = Mean(dim=0)
        self.center = Add(alpha=1.0, beta=-1.0)
        self.abs_surprise = Abs()
        self.load_surprise = Load()
        self.fuse = Add(alpha=0.6, beta=1.0)
        self.save_surprise = Save()
        self.output_func = topK()
        self.k_mean = CMean(dim=1)
        self.init_surprise = CFill(alpha=0.0)

    def create_cache(self, block_size: int, head_dim: int):
        return {"centroids": (1, head_dim), "surprise": (1, 1)}

    def forward_cache(self, cache: Dict[str, torch.Tensor], loc: torch.Tensor, ctx: ContextBase):
        self.k_mean(cache["k"], cache["centroids"], loc=loc, ctx=ctx)
        self.init_surprise(cache["surprise"], loc=loc, ctx=ctx)

    def forward_indexer(self, q: torch.Tensor, o: torch.Tensor, cache: Dict[str, torch.Tensor], ctx: ContextBase):
        q_mean = self.q_mean(q, ctx=ctx)
        raw = self.dot(q_mean, cache["centroids"], ctx=ctx)
        baseline = self.seq_mean(raw, ctx=ctx)
        centered = self.center(raw, baseline, ctx=ctx)
        current = self.abs_surprise(centered, ctx=ctx)
        last = self.load_surprise(cache["surprise"], ctx=ctx)
        running = self.fuse(last, current, ctx=ctx)
        self.save_surprise(running, cache["surprise"], ctx=ctx)
        self.output_func(running, o, ctx=ctx)
