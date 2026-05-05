import torch
from typing import Dict

from vortex_torch.abs import ContextBase
from vortex_torch.flow import vFlow, register
from vortex_torch.indexer import Add, GeMM, Mean, Multiply, Sum, WhereGreater, topK
from vortex_torch.cache import Mean as CMean, L2Norm as CL2Norm


@register("gpt_5_innovate_0_id3_cls")
class SequenceValueGate(vFlow):
    def __init__(self):
        super().__init__()
        self.q_mean = Mean(dim=1)
        self.dot = GeMM()
        self.value_align = Multiply()
        self.value_score = Sum(dim=2)
        self.seq_mean = Mean(dim=0)
        self.keep_if_above = WhereGreater()
        self.add_gate = Add(alpha=1.0, beta=1.0)
        self.output_func = topK()
        self.k_mean = CMean(dim=1)
        self.v_norm = CL2Norm(dim=1)

    def create_cache(self, block_size: int, head_dim: int):
        return {"centroids": (1, head_dim), "v_norm": (1, head_dim)}

    def forward_cache(self, cache: Dict[str, torch.Tensor], loc: torch.Tensor, ctx: ContextBase):
        self.k_mean(cache["k"], cache["centroids"], loc=loc, ctx=ctx)
        self.v_norm(cache["v"], cache["v_norm"], loc=loc, ctx=ctx)

    def forward_indexer(self, q: torch.Tensor, o: torch.Tensor, cache: Dict[str, torch.Tensor], ctx: ContextBase):
        q_mean = self.q_mean(q, ctx=ctx)
        raw = self.dot(q_mean, cache["centroids"], ctx=ctx)
        value_vec = self.value_align(q_mean, cache["v_norm"], ctx=ctx)
        value_score = self.value_score(value_vec, ctx=ctx)
        value_mean = self.seq_mean(value_score, ctx=ctx)
        gate = self.keep_if_above(value_score, value_mean, ctx=ctx)
        score = self.add_gate(raw, gate, ctx=ctx)
        self.output_func(score, o, ctx=ctx)
