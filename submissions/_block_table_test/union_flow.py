import torch
from typing import Dict

from vortex_torch.abs import ContextBase
from vortex_torch.flow import vFlow, register
from vortex_torch.indexer import GeMM, Mean, TopK, Union
from vortex_torch.cache import Mean as CMean


@register("bt_test_union_cls")
class UnionFlow(vFlow):
    """Mean + GeMM(score) + two TopK(k) ops + Union — exercises the new
    Union() output op that merges two (block_table, seqlens) pairs into
    the final sparse_block_tables + sparse_seqlens. trtllm-only.

    The two TopK calls operate on the same score tensor — in a real flow
    you'd compute two distinct score tensors (e.g. one per scoring head)
    and union their top-k. Here we use the same score so the union ends
    up identical to a single TopK selection (a sanity-check that the
    Union dedup + tail placement works correctly).
    """

    def __init__(self):
        super().__init__()
        self.q_mean = Mean(dim=1)
        self.gemm = GeMM()
        self.topk_a = TopK(k=29)
        self.topk_b = TopK(k=29)
        self.output_func = Union()

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
        bt_a, sl_a = self.topk_a(score, ctx=ctx)
        bt_b, sl_b = self.topk_b(score, ctx=ctx)
        # Union overwrites ``o`` (sparse_block_tables) and the metadata
        # sparse_seqlens — pair is ready for trtllm decode.
        self.output_func((bt_a, sl_a), (bt_b, sl_b), o, ctx=ctx)
