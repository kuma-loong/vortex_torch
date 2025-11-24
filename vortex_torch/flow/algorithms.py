import torch
from typing import Dict

from .flow import vFlow
from ..indexer import topK, GeMV, Softmax, Max, Sum, GeMM, Maximum, Multiply, Add
from ..cache import Mean as CMean, Max as CMax, Min as CMin
from ..abs import ContextBase
from .registry import register

# The ops are not reusable, even if they have the same semantic meaning. Internally, they will initialize different memory buffer.
# For example, in Quest attention, we need to define two multiply operators.

# In forward indexer, q can be viewed as [1, H_q, D] or [B, H_q, D] (B=1) and cache["xxx"] can be viewed as [S, r, c] (r, c defined in create_cache) logically.
# In forward cache, the cache["xxx"] is viewed as [B, r, c]  (r, c defined in create_cache) logically.
# Thus, all the tensors have 3 dimensions. Reduce operators (Mean, Max, Min, etc) will always keep the dims.
# GeMM(x, y) = yx^t, which might be different from typical definitions.



@register("block_sparse_attention")
class BlockSparseAttention(vFlow):
    r"""
    Block-sparse attention flow with centroid-based routing.

    This flow implements a simple **block-sparse routing** strategy
    inspired by the block-top-k routing used in Kinetics
    :cite:`sadhukhan2025kinetics` (arXiv:2506.05333). It maintains a
    per-request centroid over keys and uses query–centroid similarity to
    select a sparse set of pages.

    High-level behavior
    -------------------
    - During :meth:`forward_cache`, the flow computes a **centroid**
      vector for each request from its key cache ``cache["k"]`` and
      stores the result in ``cache["centroids"]`` with shape

      .. math::

          \text{cache["centroids"]} \in \mathbb{R}^{B \times 1 \times D},

      where :math:`B` is the number of requests and :math:`D` is the
      head dimension.

    - During :meth:`forward_indexer`, the flow:
      
      1. Averages query tokens per request to obtain a single
         **query summary** per request,
      2. Applies a generalized matrix–vector multiplication
         :class:`GeMV` between the query summaries and the cached
         centroids to obtain a scalar **score** for each (request, page),
      3. Uses :class:`topK` to convert these scores into sparse page
         indices ``o`` of shape

         .. math::

             o \in \mathbb{R}^{S} \times 1 \times 1},

         Here :math:`S` is the leading page axis. Internally it is a packed
         axis (often denoted :math:`S_{\mathrm{pack}}`), obtained by
         concatenating the pages from all requests. As a user, you can simply
         think of :math:`S` as "the number of pages for this request"; the
         vFlow kernels and :class:`ContextBase` will take care of mapping
         between per-request page counts and the packed layout automatically.

    Cache layout
    ------------
    This flow declares a single extra cache tensor via
    :meth:`create_cache`:

    .. code-block:: python

        {
            "centroids": (1, head_dim)
        }

    The runtime then also allocates ``"k"`` and ``"v"`` with inner shapes
    ``(page_size, head_dim)``. As per the :class:`vFlow` contract,
    each cache tensor has two logical views:

    - In :meth:`forward_indexer` (page-packed view):

      .. math::

          \text{cache["centroids"]} \sim
          \mathbb{R}^{S} \times 1 \times D},

    - In :meth:`forward_cache` (batch-major view):

      .. math::

          \text{cache["centroids"]} \sim
          \mathbb{R}^{B \times 1 \times D}.

    References
    ----------
    .. rubric:: Bibliography

    .. [sadhukhan2025kinetics]
       Ranajoy Sadhukhan, Zhuoming Chen, Haizhong Zheng, Yang Zhou,
       Emma Strubell, Beidi Chen.
       *Kinetics: Rethinking Test-Time Scaling Laws*.
       arXiv:2506.05333, 2025.
    """

    def __init__(self):
        super().__init__()
        # Indexer-side ops
        self.gemv = GeMV()
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
        Compute sparse page indices from queries and cached centroids.

        Parameters
        ----------
        q : torch.Tensor
            Query tensor with shape ``[B, H_q, D]`` (typically
            :class:`torch.bfloat16`), where :math:`B` is the batch–head
            axis, :math:`H_q` is the number of query positions per
            request, and :math:`D` is the head dimension.

        o : torch.Tensor
            Output tensor for sparse page indices with shape
            ``[S_sparse, 1, 1]`` and integer dtype. It is filled
            in-place by :class:`topK` according to the scores computed
            by :class:`GeMV`.

        cache : Dict[str, torch.Tensor]
            Cache dictionary in the **indexer view**, where:

            - ``cache["k"]`` and ``cache["v"]`` are page-packed key/value
              tensors,
            - ``cache["centroids"]`` is interpreted as
              ``[S, 1, D]`` (page-packed centroids).

        ctx : ContextBase
            Runtime context carrying page layout, top-k configuration
            (``topk_val``, ``page_reserved_bos``, ``page_reserved_eos``),
            and other metadata.

        Notes
        -----
        The implementation:

        1. Computes a per-request query summary

           .. math::

              q_{\mathrm{mean}}[b, 0, :]
              = \frac{1}{H_q} \sum_{h=0}^{H_q-1} q[b, h, :],

        2. Applies :class:`GeMV` between ``q_mean`` and
           ``cache["centroids"]`` to obtain scalar scores per page,
        3. Uses :class:`topK` to select a sparse set of pages per request
           and write the corresponding indices into ``o`` in the packed
           sparse layout.
        """
        q_mean = q.mean(dim=1, keepdim=True)
        score = self.gemv(q_mean, cache["centroids"], ctx=ctx)
        self.output_func(score, o, ctx=ctx)

    def forward_cache(
        self,
        cache: Dict[str, torch.Tensor],
        loc: torch.Tensor,
        ctx: ContextBase,
    ):
        r"""
        Update cache centroids from the key cache in batch-major view.

        Parameters
        ----------
        cache : Dict[str, torch.Tensor]
            Cache dictionary in the **batch-major view**, where:

            - ``cache["k"]`` has shape ``[B, page_size, D]``,
            - ``cache["centroids"]`` has shape ``[B, 1, D]``.

        loc : torch.Tensor
            Positional or layout metadata used by :class:`CMean` to
            aggregate keys into centroids (e.g. page boundaries or valid
            token masks).

        ctx : ContextBase
            Runtime context forwarded to the reduction op.

        Notes
        -----
        This method calls :class:`CMean` with ``dim=1`` so that for each
        request :math:`b` it computes a mean over the key axis and writes
        it to ``cache["centroids"][b, 0, :]``. The exact handling of
        padding or invalid positions is controlled by ``loc`` and the
        backend implementation of :class:`CMean`.
        """
        self.reduction(cache["k"], cache["centroids"], loc=loc, ctx=ctx)

    def create_cache(self, page_size: int, head_dim: int):
        r"""
        Declare inner shapes for custom cache tensors.

        Parameters
        ----------
        page_size : int
            Number of tokens per page (unused here but part of the
            generic vFlow contract).

        head_dim : int
            Head dimension :math:`D`. Used as the second dimension of
            the centroid tensor.

        Returns
        -------
        Dict[str, Tuple[int, int]]
            Mapping from cache tensor names to inner shapes ``(r, c)``.
            This flow defines a single extra tensor:

            - ``"centroids"`` with inner shape ``(1, head_dim)``, which
              becomes

              - ``[S, 1, head_dim]`` in :meth:`forward_indexer`,
              - ``[B, 1, head_dim]`` in :meth:`forward_cache`.
        """
        return {
            "centroids": (1, head_dim),
        }


@register("gqa_block_sparse_attention")
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
        self.softmax(score, ctx=ctx)
        aggr_score = self.max_op(score, ctx=ctx)
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

    def create_cache(self, page_size: int, head_dim: int):
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



@register("gqa_quest_sparse_attention")
class GQAQuestSparseAttention(vFlow):
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

    def __init__(self):
        super().__init__()

        # Indexer-side ops
        self.mul_max = Multiply()      # q * max
        self.mul_min = Multiply()      # q * min
        self.maximum_op = Maximum()    # elementwise max(q*max, q*min)
        self.sum = Sum(dim=2)          # sum over feature dim D
        self.max_op = Max(dim=1)       # max over grouped-query axis
        self.output_func = topK()      # produce sparse indices

        # Cache-side ops
        self.reduction_max = CMax(dim=1)  # page-wise max envelope over k
        self.reduction_min = CMin(dim=1)  # page-wise min envelope over k

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
        - ``cache["max"]``: ``[S, 1, D]``
        - ``cache["min"]``: ``[S, 1, D]``

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

    def create_cache(self, page_size: int, head_dim: int):
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
            "max": (1, head_dim),
            "min": (1, head_dim),
        }



@register("hybrid_quest_centroid_sparse_attention")
class HybridQuestCentroidSparseAttention(vFlow):
    r"""
    Hybrid QUEST+centroid dynamic sparse attention.

    Intuition
    ---------
    This variant combines two complementary routing signals per page:

    1) Centroid similarity (content-based):
       Per-request key centroids are dotted with queries via a GeMM.
       This captures global content alignment between queries and a
       page's overall key distribution.

    2) QUEST-style envelope bound (upper-bound based):
       Per-page max/min key envelopes are used to compute a conservative
       upper bound on achievable query–key similarity as in QUEST.

    The final per-page score is a convex combination of the two signals:

        score = alpha * score_centroid + (1 - alpha) * score_envelope,

    followed by a top-k over the packed page axis with reserved BOS/EOS
    pages handled by the runtime.

    Shapes
    ------
    - q: [B, H_q, D]
    - cache["centroids"], cache["max"], cache["min"] have inner shapes
      (1, D) and are viewed as:
        - [S, 1, D] in forward_indexer (page-packed),
        - [B, 1, D] in forward_cache (batch-major).

    Parameters
    ----------
    alpha : float, optional
        Mixing weight for centroid similarity vs. envelope bound. Must
        be in [0, 1]. Default: 0.5.
    softmax_scale : float, optional
        If > 0, applies an in-place softmax with this scale separately
        to each scoring branch before aggregation. Default: 0.0 (disabled).
    use_query_max : bool, optional
        If True, aggregate over the query axis using Max; otherwise Sum.
        Default: True.
    """

    def __init__(self, alpha: float = 0.5, softmax_scale: float = 0.0, use_query_max: bool = True):
        super().__init__()
        assert 0.0 <= alpha <= 1.0
        self.alpha = float(alpha)
        self.beta = 1.0 - float(alpha)
        self.use_query_max = bool(use_query_max)

        # Indexer-side ops
        self.gemm = GeMM()
        self.softmax_cent = Softmax(dim=0, scale=softmax_scale) if softmax_scale > 0 else None
        self.softmax_env = Softmax(dim=0, scale=softmax_scale) if softmax_scale > 0 else None
        self.mul_max = Multiply()      # q * max_envelope
        self.mul_min = Multiply()      # q * min_envelope
        self.maximum_op = Maximum()    # elementwise max(q*max, q*min)
        self.sum_feat = Sum(dim=2)     # sum over feature dim D
        # Query-axis aggregations depending on tensor layout
        self.max_q_last = Max(dim=2)    # for tensors shaped [S, *, H_q]
        self.max_q_first = Max(dim=1)   # for tensors shaped [S, H_q, *]
        self.sum_q_last = Sum(dim=2)    # for tensors shaped [S, *, H_q]
        self.sum_q_first = Sum(dim=1)   # for tensors shaped [S, H_q, *]
        self.combine = Add(alpha=self.alpha, beta=self.beta)
        self.output_func = topK()

        # Cache-side ops
        self.centroid_reduction = CMean(dim=1)
        self.env_max_reduction = CMax(dim=1)
        self.env_min_reduction = CMin(dim=1)

    def forward_indexer(
        self,
        q: torch.Tensor,
        o: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        ctx: ContextBase,
    ):
        r"""
        Compute sparse page indices from hybrid scores.

        Pipeline
        --------
        1) Centroid branch
           - s_cent_full = GeMM(q, cache["centroids"]) -> [S, 1, H_q]
           - Optional softmax along packed S with scale ``softmax_scale``
             if enabled.
           - Aggregate over query axis (Max or Sum) -> [S, 1, 1]

        2) Envelope (QUEST) branch
           - s_max = q * cache["max"], s_min = q * cache["min"]
           - s_env = max(s_max, s_min)
           - score_env = sum over feature dim D -> [S, H_q, 1]
           - Aggregate over query axis (Max or Sum) -> [S, 1, 1]
           - Optional softmax along packed S if enabled.

        3) Combine the two per-page scalars with weights (alpha, 1-alpha)
           and run topK to produce packed sparse indices ``o``.
        """
        # Centroid similarity branch
        s_cent_full = self.gemm(q, cache["centroids"], ctx=ctx)  # [S, 1, H_q]
        if self.softmax_cent is not None:
            self.softmax_cent(s_cent_full, ctx=ctx)
        # GeMM layout: [S, 1, H_q] -> reduce over query axis (last dim)
        s_cent = (self.max_q_last if self.use_query_max else self.sum_q_last)(s_cent_full, ctx=ctx)  # [S, 1, 1]

        # QUEST-style envelope branch
        s_max = self.mul_max(q, cache["max"], ctx=ctx)
        s_min = self.mul_min(q, cache["min"], ctx=ctx)
        s_env = self.maximum_op(s_max, s_min, ctx=ctx)           # [S, H_q, D]
        score_env = self.sum_feat(s_env, ctx=ctx)                 # [S, H_q, 1]
        # Envelope layout: [S, H_q, 1] -> reduce over query axis (dim=1)
        s_env_aggr = (self.max_q_first if self.use_query_max else self.sum_q_first)(score_env, ctx=ctx)  # [S, 1, 1]
        if self.softmax_env is not None:
            self.softmax_env(s_env_aggr, ctx=ctx)

        # Weighted fusion and selection
        fused = self.combine(s_cent, s_env_aggr, ctx=ctx)  # [S, 1, 1]
        self.output_func(fused, o, ctx=ctx)

    def forward_cache(
        self,
        cache: Dict[str, torch.Tensor],
        loc: torch.Tensor,
        ctx: ContextBase,
    ):
        r"""
        Maintain per-request summaries used by the indexer.

        - cache["centroids"] <- mean over page tokens of cache["k"].
        - cache["max"] <- elementwise max envelope over cache["k"].
        - cache["min"] <- elementwise min envelope over cache["k"].
        """
        self.centroid_reduction(cache["k"], cache["centroids"], loc=loc, ctx=ctx)
        self.env_max_reduction(cache["k"], cache["max"], loc=loc, ctx=ctx)
        self.env_min_reduction(cache["k"], cache["min"], loc=loc, ctx=ctx)

    def create_cache(self, page_size: int, head_dim: int):
        r"""
        Declare inner shapes for custom cache tensors.

        Returns
        -------
        Dict[str, Tuple[int, int]]
            - "centroids": (1, head_dim)
            - "max": (1, head_dim)
            - "min": (1, head_dim)
        """
        return {
            "centroids": (1, head_dim),
            "max": (1, head_dim),
            "min": (1, head_dim),
        }


# For agent developers!
# The ops are not reusable, even if they have the same semantic meaning. Internally, they will initialize different memory buffer.
# For example, in Quest attention, we need to define two multiply operators.

# In forward indexer, q can be viewed as [1, H_q, D] or [B, H_q, D] (B=1) and cache["xxx"] can be viewed as [S, r, c] (r, c defined in create_cache) logically.
# In forward cache, the cache["xxx"] is viewed as [B, r, c]  (r, c defined in create_cache) logically.
# Thus, all the tensors have 3 dimensions. Reduce operators (Mean, Max, Min, etc) will always keep the dims.
# GeMM(x, y) = yx^t, which might be different from typical definitions.

