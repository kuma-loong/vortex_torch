import torch
from typing import Dict

from vortex_torch.flow import vFlow, register
from vortex_torch.indexer import (
    topK, approxTopK, GeMV, Softmax, Max, Sum, GeMM,
    Maximum, Multiply, Add, L2Norm, Save, Load, Mean, MaskSlice, Kron,
)
from vortex_torch.cache import (
    Mean as CMean, Max as CMax, Min as CMin, L2Norm as CL2Norm,
    Fill as CFill, MaxInterleave as CMaxInterleave, MinInterleave as CMinInterleave,
)
from vortex_torch.abs import ContextBase


@register("centered_block_sparse_attention_sub")
class CenteredBlockSparseAttention(vFlow):
    r"""
    Block-sparse attention that **centers** per-page scores against a
    per-(batch, kv_head) mean before topK selection.

    The centering uses two features that landed together:

    1. :class:`Mean` with ``dim=0`` — a *cross-row* (Schedule.S) reduce
       that collapses the packed page axis into one value per
       (batch, kv_head). The result is a BATCHED intermediate of shape
       :math:`[B \cdot H_{kv}, 1, 1]`, allocated by the compiler with
       leading dim ``ctx.max_bs * ctx.num_kv_heads``.

    2. :class:`Add` with ``alpha=1, beta=-1`` over a (RAGGED, BATCHED)
       pair — uses the new mixed-format dispatch in
       :class:`Elementwise_Binary` so the per-page RAGGED score can be
       offset by the BATCHED summary.

    Pipeline (indexer)
    ------------------
    .. code-block:: text

        s         = q * cache["centroids"]      # RAGGED [S, H_q, D]
        score_d   = sum(s, dim=2)               # RAGGED [S, H_q, 1]
        score     = mean(score_d, dim=1)        # RAGGED [S, 1, 1]
        mean_seq  = mean(score, dim=0)          # BATCHED [B*H_kv, 1, 1]   (Schedule.S)
        centered  = score - mean_seq            # RAGGED [S, 1, 1]   (RAGGED + BATCHED)
        topK(centered, o)
    """

    def __init__(self):
        super().__init__()
        # Indexer-side ops
        self.mul = Multiply()
        self.sum_d = Sum(dim=2)
        self.mean_h = Mean(dim=1)
        self.mean_seq = Mean(dim=0)            # Schedule.S, RAGGED → BATCHED
        self.center = Add(alpha=1.0, beta=-1.0)  # score - mean_seq
        self.output_func = topK()

        # Cache-side ops
        self.reduction = CMean(dim=1)

    def forward_indexer(
        self,
        q: torch.Tensor,
        o: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        ctx: ContextBase,
    ):
        s = self.mul(q, cache["centroids"], ctx=ctx)        # RAGGED
        score_d = self.sum_d(s, ctx=ctx)                    # RAGGED
        score = self.mean_h(score_d, ctx=ctx)               # RAGGED [S, 1, 1]
        mean_seq = self.mean_seq(score, ctx=ctx)            # BATCHED [B*H_kv, 1, 1]
        centered = self.center(score, mean_seq, ctx=ctx)    # RAGGED via (R, B) dispatch
        self.output_func(centered, o, ctx=ctx)

    def forward_cache(
        self,
        cache: Dict[str, torch.Tensor],
        loc: torch.Tensor,
        ctx: ContextBase,
    ):
        self.reduction(cache["k"], cache["centroids"], loc=loc, ctx=ctx)

    def create_cache(self, block_size: int, head_dim: int):
        return {
            "centroids": (1, head_dim),
        }


