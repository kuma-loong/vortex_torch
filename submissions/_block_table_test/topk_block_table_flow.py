import torch
from typing import Dict

from vortex_torch.abs import ContextBase
from vortex_torch.flow import vFlow, register
from vortex_torch.indexer import GeMM, Mean, TopK, topK
from vortex_torch.cache import Mean as CMean


@register("bt_test_topk_block_table_cls")
class TopKBlockTableFlow(vFlow):
    """Mean + GeMM(score) + (intermediate TopK(k=29)) + topK(score, o) — exercises
    the new ``select.TopK`` op which returns ``(block_tables, seqlens)`` as
    two auto-allocated intermediate tensors. The trailing ``topK(score, o)``
    is here only to satisfy the framework's "``o`` must be produced by some
    op" invariant; a follow-up op will copy the new TopK outputs into the
    final buffers and replace this stub.
    """

    def __init__(self):
        super().__init__()
        self.q_mean = Mean(dim=1)
        self.gemm = GeMM()
        self.topk_block_table = TopK(k=29)   # new multi-output op
        self.output_func = topK()            # existing TopKOut — writes ``o``

        self.reduction = CMean(dim=1)

    def create_cache(self, block_size: int, head_dim: int):
        return {"centroids": (1, head_dim)}

    def forward_cache(
        self,
        cache: Dict[str, torch.Tensor],
        loc: torch.Tensor,
        ctx: ContextBase,
    ):
        self.reduction(cache["k"], cache["centroids"], loc=loc, ctx=ctx)

    def forward_indexer(
        self,
        q: torch.Tensor,
        o: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        ctx: ContextBase,
    ):
        q_mean = self.q_mean(q, ctx=ctx)
        score = self.gemm(q_mean, cache["centroids"], ctx=ctx)
        # New: pure-intermediate TopK. Returns two auto-allocated buffers.
        block_tables, seqlens = self.topk_block_table(score, ctx=ctx)
        # Stub: write the final ``o`` via the existing topK so the engine
        # still has a valid sparse path. Will be replaced once the
        # block_tables→o copy op exists.
        self.output_func(score, o, ctx=ctx)
