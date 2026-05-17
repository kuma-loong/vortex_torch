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


@register("running_avg_block_sparse_sub")
class RunningAvgBlockSparse(vFlow):
    r"""
    Block-sparse attention with a running-average page score.

    Each decode step we maintain a per-page persistent scalar
    ``cache["running_score"]`` updated by

    .. math::

        \text{running\_score} \leftarrow
        \alpha \cdot \text{last\_running\_score} + \text{current\_score},

    where ``current_score`` is the usual ``q_mean · centroid`` per-page
    score and ``alpha`` controls momentum. Pages that keep scoring
    highly accumulate; pages that lose relevance decay.

    Illustrates the :class:`Save` / :class:`Load` pattern for persistent
    state across decode steps — ``running_score`` is declared in
    :meth:`create_cache` but written entirely from
    :meth:`forward_indexer` via ``Save``; :meth:`forward_cache` never
    touches it.
    """
    ALPHA = 0.5

    def __init__(self):
        super().__init__()
        # Indexer-side ops
        self.mean        = Mean(dim=1)
        self.gemm        = GeMM()
        self.load_score  = Load()
        self.fuse        = Add(alpha=self.ALPHA, beta=1.0)
        self.save_score  = Save()
        self.output_func = topK()

        # Cache-side ops
        self.reduction = CMean(dim=1)
        # Zero-initialise the persistent per-block scalar when each new
        # block completes. Without this, the first ``Load`` after a
        # block is allocated reads whatever was in that memory slot
        # before — typically stale values from a prior sequence.
        self.init_running_score = CFill(alpha=0.0)

    def forward_indexer(
        self,
        q: torch.Tensor,
        o: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        ctx: ContextBase,
    ):
        q_mean       = self.mean(q, ctx=ctx)                               # [1, 1, D]
        current      = self.gemm(q_mean, cache["centroids"], ctx=ctx)      # [S, 1, 1]
        last_running = self.load_score(cache["running_score"], ctx=ctx)    # [S, 1, 1]
        running      = self.fuse(last_running, current, ctx=ctx)           # α*last + current
        self.save_score(running, cache["running_score"], ctx=ctx)          # persist
        self.output_func(running, o, ctx=ctx)

    def forward_cache(
        self,
        cache: Dict[str, torch.Tensor],
        loc: torch.Tensor,
        ctx: ContextBase,
    ):
        self.reduction(cache["k"], cache["centroids"], loc=loc, ctx=ctx)
        self.init_running_score(cache["running_score"], loc=loc, ctx=ctx)

    def create_cache(self, block_size: int, head_dim: int):
        return {
            "centroids":     (1, head_dim),  # maintained by forward_cache
            "running_score": (1, 1),         # maintained by forward_indexer (Save)
        }


