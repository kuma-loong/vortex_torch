import torch
from typing import Dict

from .flow_mla import vFlowMLA
from .registry import register
from ..indexer import topK, GeMM, Mean, Max, MaskSlice, Multiply
from ..cache import Mean as CMean, MeanInterleave as CMeanInterleave
from ..abs import ContextBase

# MLA flows (DeepSeek-V2/V3-style latent attention). The cache stores a single
# shared, fused latent per token: cache["latent"] = [ kv_c | k_pe ] of inner dim
# (kv_lora_rank + qk_rope_head_dim), auto-provided. forward_indexer receives the
# fused absorbed query q = [ q_nope_out | q_pe ] (the backend concatenates), so a
# single dot ⟨q, centroid(latent)⟩ is the FULL decode logit
# ⟨q_nope,kv_c⟩ + ⟨q_pe,k_pe⟩. See vortex_torch/flow/flow_mla.py.


@register("rope_aware_block_sparse_mla")
class RopeAwareBlockSparseMLA(vFlowMLA):
    r"""
    Block-sparse routing on the **fused MLA latent** — the full decode logit
    (latent + RoPE terms) from one centroid and one dot product.

    **Cache.** :meth:`forward_cache` keeps one centroid per page: the mean of the
    page's fused latents

    .. math::

        c_p \;=\; \frac{1}{|p|}\sum_{t\in p}\text{latent}_t \;\in\; \mathbb{R}^{d_c+d_r},
        \qquad \text{latent}_t = [\,\text{kv\_c}_t \,|\, \text{k\_pe}_t\,].

    **Routing.** With the head-averaged fused query
    :math:`\bar q = \tfrac1H\sum_h [\,q\_nope\_out_h \,|\, q\_pe_h\,]`, each page
    is scored by the **complete** logit (both terms fall out of the single dot,
    since the latent and query are the concatenations), then top-:math:`k`:

    .. math::

        \operatorname{score}(p) = \langle \bar q,\, c_p\rangle
        = \langle \bar q_{\text{nope}}, c^{\text{kv}}_p\rangle
        + \langle \bar q_{\text{pe}}, c^{\text{pe}}_p\rangle .
    """

    def __init__(self):
        super().__init__()
        # Indexer-side ops (run every decode step)
        self.mean = Mean(dim=1)        # average the fused query over its H heads
        self.gemm = GeMM()             # GeMM(x, y) = y @ xᵀ  → per-page score
        self.output_func = topK()      # terminal: write selected page ids to o

        # Cache-side op (run once per finished page)
        self.reduction = CMean(dim=1)  # centroid = mean of the fused latent over the block

    def forward_indexer(
        self,
        q: torch.Tensor,               # [B, H, latent_dim]  ([q_nope_out | q_pe])
        o: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        ctx: ContextBase,
    ):
        q_mean = self.mean(q, ctx=ctx)                              # [B, 1, latent_dim]
        score = self.gemm(q_mean, cache["centroids"], ctx=ctx)     # [S, 1, 1]
        self.output_func(score, o, ctx=ctx)

    def forward_cache(
        self,
        cache: Dict[str, torch.Tensor],
        loc: torch.Tensor,
        ctx: ContextBase,
    ):
        self.reduction(cache["latent"], cache["centroids"], loc=loc, ctx=ctx)

    def create_cache(self, block_size: int, kv_lora_rank: int, qk_rope_head_dim: int):
        # "latent" is auto-provided — declare only the aux centroid (full width).
        return {
            "centroids": (1, kv_lora_rank + qk_rope_head_dim),
        }


@register("lserve_centroid_mla")
class LServeCentroidMLA(vFlowMLA):
    r"""
    LServe-style **sub-block** centroid routing on the fused MLA latent.

    Each block is split into consecutive sub-blocks of :attr:`SUB_BLOCK_SIZE`
    tokens, and one fused-latent centroid is kept **per sub-block**. A block is
    ranked by the query's best match against any of its sub-block centroids, so
    a single relevant sub-region is enough to select the block — sharper than a
    single block-mean when the relevant keys are concentrated in part of a
    block.

    **Cache.** :meth:`forward_cache` stores, for each of the
    :math:`n_b = \text{block\_size} / \text{SUB\_BLOCK\_SIZE}` sub-blocks
    :math:`b` of block :math:`p`, a fused-latent centroid via
    :class:`MeanInterleave`:

    .. math::

        c_{p,b} = \frac{1}{|b|}\sum_{t\in b}\text{latent}_t,
        \qquad \text{latent}_t = [\,\text{kv\_c}_t \,|\, \text{k\_pe}_t\,].

    **Routing.** With the head-averaged fused query
    :math:`\bar q = \tfrac1H\sum_h [\,q\_nope\_out_h \,|\, q\_pe_h\,]`, each
    sub-block carries the **complete** decode logit (latent + RoPE terms fall
    out of the one dot, since both query and latent are the concatenations);
    the block score is the max over its sub-blocks:

    .. math::

        \operatorname{score}(p)
        = \max_{b}\,\langle \bar q,\, c_{p,b}\rangle .

    **Shapes.** ``q`` is ``[B, H, d_c+d_r]``; ``cache["centroids"]`` is
    ``[S, n_b, d_c+d_r]`` (indexer) / ``[B, n_b, d_c+d_r]`` (cache). With
    ``block_size == SUB_BLOCK_SIZE`` there is one sub-block per block
    (:math:`n_b = 1`), recovering :class:`RopeAwareBlockSparseMLA`.
    """
    SUB_BLOCK_SIZE = 16

    def __init__(self):
        super().__init__()
        # Indexer-side ops (run every decode step)
        self.mean = Mean(dim=1)        # average the fused query over its H heads
        self.gemm = GeMM()             # GeMM(x, y) = y @ xᵀ  → per-sub-block score
        self.max_sub = Max(dim=1)      # max over the block's sub-block centroids
        self.output_func = topK()      # terminal: write selected block ids to o

        # Cache-side op (run once per finished block): per-sub-block latent mean
        self.reduction = CMeanInterleave(dim=1, k=self.SUB_BLOCK_SIZE)

    def forward_indexer(
        self,
        q: torch.Tensor,               # [B, H, latent_dim]  ([q_nope_out | q_pe])
        o: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        ctx: ContextBase,
    ):
        q_mean = self.mean(q, ctx=ctx)                              # [B, 1, latent_dim]
        score = self.gemm(q_mean, cache["centroids"], ctx=ctx)     # [S, n_b, 1]
        page_score = self.max_sub(score, ctx=ctx)                  # [S, 1, 1]
        self.output_func(page_score, o, ctx=ctx)

    def forward_cache(
        self,
        cache: Dict[str, torch.Tensor],
        loc: torch.Tensor,
        ctx: ContextBase,
    ):
        self.reduction(cache["latent"], cache["centroids"], loc=loc, ctx=ctx)

    def create_cache(self, block_size: int, kv_lora_rank: int, qk_rope_head_dim: int):
        # "latent" is auto-provided — declare only the per-sub-block centroids.
        return {
            "centroids": (block_size // self.SUB_BLOCK_SIZE, kv_lora_rank + qk_rope_head_dim),
        }


# ---------------------------------------------------------------------------
# RoPE-UNAWARE variants
# ---------------------------------------------------------------------------
# The fused latent is ``[ kv_c (dims 0:kv_lora_rank) | k_pe (dims kv_lora_rank:) ]``
# and the fused query is ``[ q_nope_out | q_pe ]``. The rope_aware flows above
# score with the FULL dot ⟨q,c⟩ = ⟨q_nope,c_kv⟩ + ⟨q_pe,c_pe⟩, so the RoPE
# (positional) term participates in routing.
#
# The rope_unaware flows below DROP the RoPE term: they route on the latent
# (NoPE) component only,
#
#     score(p) = ⟨q_nope, c_kv_p⟩ ,
#
# i.e. the landmark/centroid is effectively built from the first ``kv_lora_rank``
# dims and the last ``qk_rope_head_dim`` (= 64) dims are masked out. We realise
# this by zeroing the query's RoPE dims before the GeMM: ``MaskSlice`` emits the
# constant pattern ``[1]*kv_lora_rank ++ [0]*qk_rope_head_dim`` (it is a pure
# position writer — input values are ignored), and ``Multiply`` applies it to the
# query. With q's RoPE dims zeroed the 576-wide dot collapses to ⟨q_nope,c_kv⟩,
# so the (unchanged, full-width) centroid's RoPE dims never affect the score —
# mathematically identical to masking the centroid, but with GeMM widths still
# aligned and using only cheap indexer ops. Compare against the rope_aware
# baselines to measure what the RoPE term contributes to routing quality.
#
# NoPE width is the framework-fixed kv_lora_rank (512); create_cache asserts it.


@register("rope_unaware_block_sparse_mla")
class RopeUnawareBlockSparseMLA(vFlowMLA):
    r"""
    RoPE-unaware twin of :class:`RopeAwareBlockSparseMLA`: one centroid per page,
    but routing uses only the latent (NoPE) component — the last
    ``qk_rope_head_dim`` (64) query dims are masked to zero, so

    .. math::

        \operatorname{score}(p) = \langle \bar q_{\text{nope}},\, c^{\text{kv}}_p\rangle ,

    dropping the RoPE term :math:`\langle \bar q_{\text{pe}}, c^{\text{pe}}_p\rangle`.
    """
    NOPE_DIM = 512  # kv_lora_rank; RoPE (k_pe) occupies dims [NOPE_DIM, latent_dim)

    def __init__(self):
        super().__init__()
        # Indexer-side ops (run every decode step)
        self.mean = Mean(dim=1)        # average the fused query over its H heads
        # Position pattern [1]*NOPE_DIM ++ [0]*rope over the latent (dim=2) axis.
        self.rope_mask = MaskSlice(start=0, end=self.NOPE_DIM, dim=2, alpha=1.0, beta=0.0)
        self.mul = Multiply()          # zero the query's RoPE dims
        self.gemm = GeMM()             # GeMM(x, y) = y @ xᵀ  → per-page score
        self.output_func = topK()      # terminal: write selected page ids to o

        # Cache-side op (run once per finished page)
        self.reduction = CMean(dim=1)  # centroid = mean of the fused latent over the block

    def forward_indexer(
        self,
        q: torch.Tensor,               # [B, H, latent_dim]  ([q_nope_out | q_pe])
        o: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        ctx: ContextBase,
    ):
        q_mean = self.mean(q, ctx=ctx)                              # [B, 1, latent_dim]
        mask = self.rope_mask(q_mean, ctx=ctx)                     # [B, 1, latent_dim] 1s|0s
        q_nope = self.mul(q_mean, mask, ctx=ctx)                   # RoPE dims zeroed
        score = self.gemm(q_nope, cache["centroids"], ctx=ctx)     # [S, 1, 1]
        self.output_func(score, o, ctx=ctx)

    def forward_cache(
        self,
        cache: Dict[str, torch.Tensor],
        loc: torch.Tensor,
        ctx: ContextBase,
    ):
        self.reduction(cache["latent"], cache["centroids"], loc=loc, ctx=ctx)

    def create_cache(self, block_size: int, kv_lora_rank: int, qk_rope_head_dim: int):
        assert kv_lora_rank == self.NOPE_DIM, (
            f"rope_unaware_block_sparse_mla assumes kv_lora_rank={self.NOPE_DIM}, "
            f"got {kv_lora_rank}"
        )
        # "latent" is auto-provided — declare only the aux centroid (full width).
        return {
            "centroids": (1, kv_lora_rank + qk_rope_head_dim),
        }


@register("rope_unaware_lserve_centroid_mla")
class RopeUnawareLServeCentroidMLA(vFlowMLA):
    r"""
    RoPE-unaware twin of :class:`LServeCentroidMLA`: per-sub-block centroids with
    a max over sub-blocks, but routing uses only the latent (NoPE) component —
    the last ``qk_rope_head_dim`` (64) query dims are masked to zero, so

    .. math::

        \operatorname{score}(p) = \max_{b}\,\langle \bar q_{\text{nope}},\, c^{\text{kv}}_{p,b}\rangle ,

    dropping the per-sub-block RoPE term.
    """
    NOPE_DIM = 512        # kv_lora_rank; RoPE (k_pe) occupies dims [NOPE_DIM, latent_dim)
    SUB_BLOCK_SIZE = 16

    def __init__(self):
        super().__init__()
        # Indexer-side ops (run every decode step)
        self.mean = Mean(dim=1)        # average the fused query over its H heads
        self.rope_mask = MaskSlice(start=0, end=self.NOPE_DIM, dim=2, alpha=1.0, beta=0.0)
        self.mul = Multiply()          # zero the query's RoPE dims
        self.gemm = GeMM()             # GeMM(x, y) = y @ xᵀ  → per-sub-block score
        self.max_sub = Max(dim=1)      # max over the block's sub-block centroids
        self.output_func = topK()      # terminal: write selected block ids to o

        # Cache-side op (run once per finished block): per-sub-block latent mean
        self.reduction = CMeanInterleave(dim=1, k=self.SUB_BLOCK_SIZE)

    def forward_indexer(
        self,
        q: torch.Tensor,               # [B, H, latent_dim]  ([q_nope_out | q_pe])
        o: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        ctx: ContextBase,
    ):
        q_mean = self.mean(q, ctx=ctx)                              # [B, 1, latent_dim]
        mask = self.rope_mask(q_mean, ctx=ctx)                     # [B, 1, latent_dim] 1s|0s
        q_nope = self.mul(q_mean, mask, ctx=ctx)                   # RoPE dims zeroed
        score = self.gemm(q_nope, cache["centroids"], ctx=ctx)     # [S, n_b, 1]
        page_score = self.max_sub(score, ctx=ctx)                  # [S, 1, 1]
        self.output_func(page_score, o, ctx=ctx)

    def forward_cache(
        self,
        cache: Dict[str, torch.Tensor],
        loc: torch.Tensor,
        ctx: ContextBase,
    ):
        self.reduction(cache["latent"], cache["centroids"], loc=loc, ctx=ctx)

    def create_cache(self, block_size: int, kv_lora_rank: int, qk_rope_head_dim: int):
        assert kv_lora_rank == self.NOPE_DIM, (
            f"rope_unaware_lserve_centroid_mla assumes kv_lora_rank={self.NOPE_DIM}, "
            f"got {kv_lora_rank}"
        )
        # "latent" is auto-provided — declare only the per-sub-block centroids.
        return {
            "centroids": (block_size // self.SUB_BLOCK_SIZE, kv_lora_rank + qk_rope_head_dim),
        }


# ---------------------------------------------------------------------------
# WEIGHTED NoPE/RoPE block-sparse routing
# ---------------------------------------------------------------------------
# rope_aware scores with the full dot ⟨q,c⟩ = ⟨q_nope,c_kv⟩ + ⟨q_pe,c_pe⟩
# (the NoPE and RoPE terms enter at a fixed 1:1 ratio); rope_unaware drops the
# RoPE term. This flow makes the split a tunable convex blend:
#
#     score(p) = a · ⟨q_nope, c_kv_p⟩ + b · ⟨q_pe, c_pe_p⟩ ,     a + b = 1.
#
# Realised by scaling the query per-dimension before the (unchanged, full-width)
# GeMM: a single MaskSlice emits the weight vector [a]*kv_lora_rank ++ [b]*rope
# (alpha on [0, NOPE_DIM), beta elsewhere — MaskSlice is a pure position writer),
# Multiply applies it to the head-averaged query, and
# ⟨[a·q_nope | b·q_pe], c⟩ = a·⟨q_nope,c_kv⟩ + b·⟨q_pe,c_pe⟩ falls straight out.
#
# Because top-k ranks pages by score for one query, only the a:b RATIO affects
# selection (a global scale is irrelevant). Hence the b knob interpolates:
#   b = 0.0  -> NoPE only            (== rope_unaware_block_sparse_mla)
#   b = 0.5  -> 1:1 NoPE:RoPE        (same ranking as rope_aware_block_sparse_mla)
#   b -> 1.0 -> RoPE-dominated routing
# Sweep b to recover the RoPE signal that pure-NoPE routing loses, without
# letting the 64-d RoPE term swamp the 512-d latent term.


class _WeightedRopeBlockSparseMLA(vFlowMLA):
    r"""Base for the weighted NoPE/RoPE block-sparse flow; subclasses set the
    blend weights :attr:`A` (NoPE) and :attr:`B` (RoPE), with ``A + B == 1``.
    Not registered directly — use a ``rope_weighted_block_sparse_mla_b*`` name."""
    NOPE_DIM = 512  # kv_lora_rank; RoPE (k_pe) occupies dims [NOPE_DIM, latent_dim)
    A = 0.5         # NoPE-term weight
    B = 0.5         # RoPE-term weight   (A + B must == 1)

    def __init__(self):
        super().__init__()
        assert abs((self.A + self.B) - 1.0) < 1e-6, (
            f"{type(self).__name__}: require A + B == 1, got A={self.A}, B={self.B}"
        )
        self.mean = Mean(dim=1)
        # Per-dim blend weights: A on the NoPE dims [0, NOPE_DIM), B on the RoPE
        # dims [NOPE_DIM, latent_dim). Pure position writer (q values unused).
        self.weight = MaskSlice(start=0, end=self.NOPE_DIM, dim=2, alpha=self.A, beta=self.B)
        self.mul = Multiply()          # scale q: a·q_nope | b·q_pe
        self.gemm = GeMM()             # GeMM(x, y) = y @ xᵀ  → per-page score
        self.output_func = topK()      # terminal: selected page ids
        self.reduction = CMean(dim=1)  # centroid = mean of the fused latent

    def forward_indexer(
        self,
        q: torch.Tensor,               # [B, H, latent_dim]  ([q_nope_out | q_pe])
        o: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        ctx: ContextBase,
    ):
        q_mean = self.mean(q, ctx=ctx)                              # [B, 1, latent_dim]
        w = self.weight(q_mean, ctx=ctx)                           # [a]*nope ++ [b]*rope
        q_w = self.mul(q_mean, w, ctx=ctx)                         # a·q_nope | b·q_pe
        score = self.gemm(q_w, cache["centroids"], ctx=ctx)        # a·⟨q_nope,c_kv⟩ + b·⟨q_pe,c_pe⟩
        self.output_func(score, o, ctx=ctx)

    def forward_cache(
        self,
        cache: Dict[str, torch.Tensor],
        loc: torch.Tensor,
        ctx: ContextBase,
    ):
        self.reduction(cache["latent"], cache["centroids"], loc=loc, ctx=ctx)

    def create_cache(self, block_size: int, kv_lora_rank: int, qk_rope_head_dim: int):
        assert kv_lora_rank == self.NOPE_DIM, (
            f"{type(self).__name__} assumes kv_lora_rank={self.NOPE_DIM}, got {kv_lora_rank}"
        )
        return {"centroids": (1, kv_lora_rank + qk_rope_head_dim)}


@register("rope_weighted_block_sparse_mla_b10")
class RopeWeightedBlockSparseMLA_B10(_WeightedRopeBlockSparseMLA):
    A, B = 0.90, 0.10


@register("rope_weighted_block_sparse_mla_b25")
class RopeWeightedBlockSparseMLA_B25(_WeightedRopeBlockSparseMLA):
    A, B = 0.75, 0.25


@register("rope_weighted_block_sparse_mla_b50")
class RopeWeightedBlockSparseMLA_B50(_WeightedRopeBlockSparseMLA):
    # 1:1 NoPE:RoPE — same page ranking as rope_aware_block_sparse_mla.
    A, B = 0.50, 0.50


@register("rope_weighted_block_sparse_mla_b75")
class RopeWeightedBlockSparseMLA_B75(_WeightedRopeBlockSparseMLA):
    A, B = 0.25, 0.75


@register("rope_weighted_block_sparse_mla_b90")
class RopeWeightedBlockSparseMLA_B90(_WeightedRopeBlockSparseMLA):
    A, B = 0.10, 0.90


@register("rope_weighted_block_sparse_mla_b100")
class RopeWeightedBlockSparseMLA_B100(_WeightedRopeBlockSparseMLA):
    # Pure RoPE routing: drops the 512-d latent term entirely (mirror of unaware).
    A, B = 0.00, 1.00
