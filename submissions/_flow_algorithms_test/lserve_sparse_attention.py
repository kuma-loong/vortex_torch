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


@register("lserve_sparse_attention_sub")
class LServeSparseAttention(vFlow):
    r"""
    GQA-style QUEST sparse attention flow.

    This flow uses **query–envelope matching** similar to QUEST sparse
    attention (see https://arxiv.org/abs/2406.10774). For each request,
    it maintains per-page **max** and **min** envelopes of keys and uses
    them to compute a conservative upper bound on query–key similarity.

    Shapes
    ------
    - Queries ``q``: ``[B, H_q, D]`` (typically bfloat16).
    - Cache entries (inner shapes as declared in :meth:`create_cache`):

      - ``cache["max"]`` and ``cache["min"]``: ``(1, head_dim)``
        → viewed as

        - ``[S, 1, D]`` in :meth:`forward_indexer`,
        - ``[B, 1, D]`` in :meth:`forward_cache`.

      - ``cache["k"]``: standard key cache with inner shape
        ``(page_size, head_dim)``.

      Here :math:`S` is the leading page axis. Internally it is a packed
      axis (often denoted :math:`S_{\mathrm{pack}}`), obtained by
      concatenating the pages from all requests. As a user, you can simply
      think of :math:`S` as "the number of pages for this request"; the
      vFlow kernels and :class:`ContextBase` will take care of mapping
      between per-request page counts and the packed layout automatically.
      
    Routing intuition
    -----------------
    For each query and page envelope:

    1. Compute elementwise products with the **max** and **min** envelopes.
    2. Take an elementwise maximum of these two products to form a
       QUEST-style upper bound.
    3. Sum over the feature dimension and then take a max over the
       grouped-query axis to get a single scalar score per page.
    4. Feed the resulting per-page scores into :class:`topK` to obtain
       sparse page indices.
    """
    LSERVE_BLOCK_SIZE = 16
    def __init__(self):
        super().__init__()

        # Indexer-side ops
        self.mul_max = Kron(dim=1)     # q * max
        self.mul_min = Kron(dim=1)     # q * min
        self.maximum_op = Maximum()    # elementwise max(q*max, q*min)
        self.sum = Sum(dim=2)          # sum over feature dim D
        self.max_op = Max(dim=1)       # max over grouped-query axis
        self.output_func = topK()      # produce sparse indices

        # Cache-side ops
        self.reduction_max = CMaxInterleave(dim=1, k=self.LSERVE_BLOCK_SIZE)  # page-wise max envelope over k
        self.reduction_min = CMinInterleave(dim=1, k=self.LSERVE_BLOCK_SIZE)  # page-wise min envelope over k

    def forward_indexer(
        self,
        q: torch.Tensor,
        o: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        ctx: ContextBase,
    ):
        r"""
        Compute sparse page indices using QUEST-style envelope scores.

        Pipeline (indexer view)
        -----------------------
        Let:

        - ``q``: ``[B, H_q, D]``
        - ``cache["max"]``: ``[S, b//k, D]``
        - ``cache["min"]``: ``[S, b//k, D]``

        Steps:

        1. ``s_max = q * max_envelope``
        2. ``s_min = q * min_envelope``
        3. ``s = max(s_max, s_min)`` (elementwise)
        4. ``score = sum(s, dim=D)`` → ``[S, H_q, 1]``
        5. ``aggr_score = max(score, dim=H_q)`` → per-page scalar
        6. :class:`topK` converts ``aggr_score`` into sparse page
           indices ``o`` of shape ``[S_sparse, 1, 1]``.
        """
        s_max = self.mul_max(q, cache["max"], ctx=ctx)
        s_min = self.mul_min(q, cache["min"], ctx=ctx)
        s = self.maximum_op(s_max, s_min, ctx=ctx)
        score = self.sum(s, ctx=ctx)
        aggr_score = self.max_op(score, ctx=ctx)
        self.output_func(aggr_score, o, ctx=ctx)

    def forward_cache(
        self,
        cache: Dict[str, torch.Tensor],
        loc: torch.Tensor,
        ctx: ContextBase,
    ):
        r"""
        Update per-page max/min envelopes from the key cache.

        Cache-update view
        -----------------
        - ``cache["k"]``: ``[B, page_size, D]``
        - ``cache["max"]``: ``[B, 1, D]``
        - ``cache["min"]``: ``[B, 1, D]``

        The :class:`CMax` and :class:`CMin` ops (with ``dim=1``) take
        page-wise maxima and minima over keys (optionally masked/structured
        via ``loc``) and write the envelopes into ``cache["max"]`` and
        ``cache["min"]``.
        """
        self.reduction_max(cache["k"], cache["max"], loc=loc, ctx=ctx)
        self.reduction_min(cache["k"], cache["min"], loc=loc, ctx=ctx)

    def create_cache(self, block_size: int, head_dim: int):
        r"""
        Declare inner shapes for custom cache tensors.

        Parameters
        ----------
        page_size : int
            Number of tokens per page (unused here but part of the vFlow contract).

        head_dim : int
            Head dimension ``D`` used by the envelopes.

        Returns
        -------
        Dict[str, Tuple[int, int]]
            Custom cache metadata:

            - ``"max"``: inner shape ``(1, head_dim)``
            - ``"min"``: inner shape ``(1, head_dim)``
        """
        return {
            "max": (block_size // self.LSERVE_BLOCK_SIZE, head_dim),
            "min": (block_size // self.LSERVE_BLOCK_SIZE, head_dim),
        }


