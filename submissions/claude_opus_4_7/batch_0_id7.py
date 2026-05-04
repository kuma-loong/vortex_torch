import torch
from typing import Dict

from vortex_torch.flow import vFlow, register
from vortex_torch.indexer import topK, Multiply, Maximum, Sum, Max
from vortex_torch.cache import Max as CMax, Min as CMin
from vortex_torch.abs import ContextBase


@register("claude_opus_4_7_batch_0_id7_cls")
class ClaudeOpus47Batch0Id7Cls(vFlow):
    """GQA-Quest envelope (CMin+CMax) — score-function knob vs id0's CMean."""
    def __init__(self):
        super().__init__()
        self.mul_max = Multiply()
        self.mul_min = Multiply()
        self.maximum_op = Maximum()
        self.sum = Sum(dim=2)
        self.max_op = Max(dim=1)
        self.output_func = topK()

        self.k_max = CMax(dim=1)
        self.k_min = CMin(dim=1)

    def create_cache(self, block_size: int, head_dim: int):
        return {
            "max": (1, head_dim),
            "min": (1, head_dim),
        }

    def forward_cache(self, cache: Dict[str, torch.Tensor], loc: torch.Tensor, ctx: ContextBase):
        self.k_max(cache["k"], cache["max"], loc=loc, ctx=ctx)
        self.k_min(cache["k"], cache["min"], loc=loc, ctx=ctx)

    def forward_indexer(self, q: torch.Tensor, o: torch.Tensor, cache: Dict[str, torch.Tensor], ctx: ContextBase):
        s_max = self.mul_max(q, cache["max"], ctx=ctx)
        s_min = self.mul_min(q, cache["min"], ctx=ctx)
        s = self.maximum_op(s_max, s_min, ctx=ctx)
        score = self.sum(s, ctx=ctx)
        aggr = self.max_op(score, ctx=ctx)
        self.output_func(aggr, o, ctx=ctx)
