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


@register("gqa_block_sparse_attention_sub")
class GQABlockSparseAttention(vFlow):
    r"""
    Grouped-query block-sparse attention flow.

    This flow uses a GQA-style block-sparse routing: queries are grouped,
    scored against per-request centroids, normalized with a softmax, then
    aggregated across groups before a top-k over pages is applied.

    - Queries ``q`` have shape ``[B, H_q, D]``.
    - Centroids cache ``cache["centroids"]`` has inner shape
      ``(1, head_dim)`` and is viewed as:

      - ``[S, 1, D]`` in :meth:`forward_indexer`,
      - ``[B, 1, D]`` in :meth:`forward_cache`.
      Here :math:`S` is the leading page axis. Internally it is a packed
      axis (often denoted :math:`S_{\mathrm{pack}}`), obtained by
      concatenating the pages from all requests. As a user, you can simply
      think of :math:`S` as "the number of pages for this request"; the
      vFlow kernels and :class:`ContextBase` will take care of mapping
      between per-request page counts and the packed layout automatically.
      
    For a design similar in spirit to grouped-query block sparsity, see
    the GQA sparse attention formulation in:

    - https://arxiv.org/abs/2502.11089
    """

    def __init__(self):
        super().__init__()
        # Indexer-side ops
        self.gemm = GeMM()
        self.softmax = Softmax(dim=0, scale=0.09)
        self.max_op = Max(dim=2)
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
        r"""
        Compute sparse page indices from grouped-query scores.

        Pipeline
        --------
        1. Apply :class:`GeMM` between queries and centroids (o = yx^t):

           - ``q``: ``[B, H_q, D]``
           - ``cache["centroids"]`` (indexer view): ``[S, 1, D]``
           - ``score``: ``[S, 1, H_q]`` (logical ``[S, Ny, Nx]``)

        2. Apply in-place softmax over the leading (page) axis with a
           scaling factor ``scale``:

           .. math::
              \mathrm{softmax}(x \cdot \mathrm{scale})

        3. Aggregate over the query-group dimension with :class:`Max`
           (``dim=2``), yielding a single scalar score per page.

        4. Use :class:`topK` on the aggregated scores to write packed
           sparse page indices into ``o`` with shape
           ``[S_sparse, 1, 1]``.
        """
        score = self.gemm(q, cache["centroids"], ctx=ctx)
        normalized_score = self.softmax(score, ctx=ctx)
        aggr_score = self.max_op(normalized_score, ctx=ctx)
        self.output_func(aggr_score, o, ctx=ctx)

    def forward_cache(
        self,
        cache: Dict[str, torch.Tensor],
        loc: torch.Tensor,
        ctx: ContextBase,
    ):
        r"""
        Update per-request centroids from the key cache.

        - ``cache["k"]``: ``[B, page_size, D]`` (batch-major view)
        - ``cache["centroids"]``: ``[B, 1, D]``

        The :class:`CMean` operator with ``dim=1`` computes a mean over
        the key axis (optionally masked/structured via ``loc``) and
        writes the result into ``cache["centroids"]`` in-place.
        """
        self.reduction(cache["k"], cache["centroids"], loc=loc, ctx=ctx)

    def create_cache(self, block_size: int, head_dim: int):
        r"""
        Declare inner shapes for custom cache tensors.

        Parameters
        ----------
        page_size : int
            Number of tokens per page (not used directly here).

        head_dim : int
            Head dimension ``D`` for centroids.

        Returns
        -------
        Dict[str, Tuple[int, int]]
            Custom cache metadata. This flow defines:

            - ``"centroids"``: inner shape ``(1, head_dim)``.
        """
        return {
            "centroids": (1, head_dim),
        }



