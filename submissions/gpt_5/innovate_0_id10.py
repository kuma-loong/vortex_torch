import torch
from typing import Dict

from vortex_torch.abs import ContextBase
from vortex_torch.flow import vFlow, register
from vortex_torch.indexer import GeMM, Mean, Multiply, Sigmoid, topK
from vortex_torch.cache import Mean as CMean, L2Norm as CL2Norm


@register("gpt_5_innovate_0_id10_cls")
class SigmoidValueGate(vFlow):
    def __init__(self):
        super().__init__()
        self.q_mean = Mean(dim=1)
        self.dot = GeMM()
        self.value_energy = GeMM()
        self.value_gate = Sigmoid(alpha=0.0, beta=-0.02)
        self.apply_gate = Multiply()
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
        energy = self.value_energy(q_mean, cache["v_norm"], ctx=ctx)
        gate = self.value_gate(energy, ctx=ctx)
        score = self.apply_gate(raw, gate, ctx=ctx)
        self.output_func(score, o, ctx=ctx)
