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


@register("block_sparse_attention_sub")
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
        self.gemm = GeMM()
        self.mean = Mean(dim=1)
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
        q_mean = self.mean(q, ctx=ctx)
        score = self.gemm(q_mean, cache["centroids"], ctx=ctx)
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

    def create_cache(self, block_size: int, head_dim: int):
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


