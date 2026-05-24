import torch
from typing import FrozenSet
from ..abs import vOp, vTensor, FORMAT
from .context import Context
from ..utils import Schedule

class topK(vOp):
    r"""
    Per-request top-k page selector with reserved BOS/EOS pages.

    The terminal op of a ``forward_indexer``: it turns per-page scores into the
    sparse set of pages each request attends to.

    :Math:
        For a request's per-page scores :math:`X_p` (:math:`p=0,\dots,S-1`),
        with reserved prefix :math:`\mathcal{B}=\{0,\dots,n_{\mathrm{bos}}-1\}`
        and suffix :math:`\mathcal{E}=\{S-n_{\mathrm{eos}},\dots,S-1\}`, the
        selected page set is

        .. math::

            \mathcal{S} = \mathcal{B}\,\cup\,\mathcal{E}\,\cup\,
            \operatorname*{top\text{-}k}_{\,p\,\notin\,\mathcal{B}\cup\mathcal{E}} X_p,
            \qquad k = \texttt{topk\_val}.
    :__init__: ``topK()`` — no arguments; the budget ``topk_val`` and the
        reserved BOS/EOS counts are read from :class:`Context` at runtime.
    :__call__: ``op(score, o, ctx=ctx)`` — ``score`` is ``[S, 1, 1]`` (one
        scalar per page; must be ``RAGGED``); the selected page ids are written
        **in place** into ``o``. Returns nothing.
    :Note: every flow's ``forward_indexer`` must end in this op (or
        :class:`approxTopK`).
    """

    # Supported input formats; only RAGGED is supported for now.
    _supported_formats: FrozenSet[FORMAT] = frozenset({FORMAT.RAGGED})

    def __init__(self):
        super().__init__()
        self.schedule = Schedule.S

    # ---------------- profile ----------------
    def profile(self, x: vTensor, o: vTensor, ctx: Context) -> None:
        r"""Trace-time: validate ``x`` ``[S, 1, 1]`` and ``o`` (both ``RAGGED``
        ``vTensor`` on the same device) and register the op. Allocates nothing
        and returns nothing — ``o`` is filled in place at execute time."""
        prefix = self._prefix()

        # ---- type checks ----
        assert isinstance(x, vTensor), f"{prefix}profile expects x to be vTensor, got {type(x)}"
        assert isinstance(o, vTensor), f"{prefix}profile expects o to be vTensor, got {type(o)}"

        # ---- rank checks ----
        assert x.dim() == 3, (
            f"{prefix}expected x to be 3D, "
            f"got ndim={x.dim()} shape={tuple(x.shape)}"
        )
        assert o.dim() == 3, (
            f"{prefix}expected o to be 3D, "
            f"got ndim={o.dim()} shape={tuple(o.shape)}"
        )

        # ---- shape checks for x ----
        # x is expected to carry per-page scalars at dims (1,2)
        assert x.shape[1] == 1 and x.shape[2] == 1, (
            f"{prefix}expected x.shape[1] == x.shape[2] == 1, got {tuple(x.shape)}"
        )

        # ---- implementation availability ----
        x_fmt = x._format
        assert x_fmt in self._supported_formats, (
            f"{prefix}no implementation for x._format={x_fmt}. "
            f"Supported: {sorted(self._supported_formats, key=lambda f: f.value)}"
        )

        # ---- optional sanity checks on `o` ----
        # We only assert device consistency and leave exact (S_pack, D0, D1)
        # to the upstream contract and the implementation.
        assert x.device == o.device, (
            f"{prefix}x and o must be on the same device "
            f"(x.device={x.device}, o.device={o.device})"
        )

        # Track auxiliary memory and graph structure in the context
        ctx.output_tensor_to_op_list[o.tensor_id] = len(ctx.op_list)   # Map the output tensor to this operation
        ctx.op_list.append(self)  # Track this operation in the context
        ctx.op_to_input_tensor_list.append([x.tensor_id])  # Map this op to its input tensors
        ctx.op_to_output_tensor_list.append([o.tensor_id])  # Map this op to its output tensor


class approxTopK(topK):
    r"""
    Approximate :class:`topK` — faster adaptive 8-bit radix selection.

    :Math:
        The selected set :math:`\mathcal{S}` matches :class:`topK`, but the
        top-:math:`k` search over the non-reserved pages stops at the first
        radix round whose count :math:`r` still owed by the threshold bin
        satisfies

        .. math::

            r \;\le\; \texttt{tolerate\_ratio}\cdot k,

        with any remaining slots filled in arrival order.
    :__init__: ``approxTopK(tolerate_ratio=0.0)`` — approximation budget in
        ``[0, 1]``: ``0.0`` = exact (all radix rounds run), higher = cheaper
        but looser (typical throughput sweet spot ``0.05–0.15``).
    :__call__: ``op(score, o, ctx=ctx)`` — same signature as :class:`topK`
        (``score`` ``[S, 1, 1]``, ``RAGGED``; writes page ids into ``o``).
    :Note: indices are **unsorted within each request**; downstream consumers
        must not assume sorted order.
    """

    def __init__(self, tolerate_ratio: float = 0.0):
        super().__init__()
        try:
            tol = float(tolerate_ratio)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"approxTopK: tolerate_ratio must be a number in [0.0, 1.0], "
                f"got {tolerate_ratio!r}"
            ) from e
        if not (0.0 <= tol <= 1.0):
            raise ValueError(
                f"approxTopK: tolerate_ratio={tol} out of range [0.0, 1.0]. "
                f"0.0 = exact (4 rounds); 1.0 = cheapest single-round; "
                f"typical sweet spot 0.05 - 0.15."
            )
        self.tolerate_ratio = tol


class Union(vOp):
    r"""Merge two ``(block_table, seqlens)`` pairs into the final sparse output.

    Output function for the trtllm backend that takes two
    ``(block_table_i, seqlens_i)`` tuples (typically produced by
    :class:`vortex_torch.indexer.TopK` calls), computes the per-row
    deduplicated union of their block ids, and overwrites the final
    ``sparse_block_tables`` + ``sparse_seqlens`` buffers so the pair can
    be fed straight to ``trtllm_batch_decode_with_kv_cache``.

    Per-row behavior, with
    ``dense_block_len = ceil(dense_seqlens[i] / block_size)`` and
    ``last_block_len = dense_seqlens[i] mod block_size`` (treated as
    ``block_size`` when the remainder is zero):

      * Compute the deduped union of block ids drawn from
        ``block_table_0[i, :blocks_0]`` and ``block_table_1[i, :blocks_1]``,
        where ``blocks_j = ceil(seqlens_j[i] / block_size)``. The dense
        path's true last block id (``dense_block_tables[i, dense_block_len-1]``)
        is excluded from the dedup walk and re-inserted at the **tail**
        of the union so trtllm decode reads the partial token count from
        the right slot.
      * Write the deduped tail-canonical union into ``o[i, :u+1]`` and
        set ``sparse_seqlens[i] = u * block_size + last_block_len``
        (where ``u`` is the dedup count, excluding the appended last
        block).

    Invocation
    ----------
    ::

        self.output_func = Union()
        ...
        self.output_func(i_0, i_1, o, ctx=ctx)
        # i_0 = (block_table_0, seqlens_0)
        # i_1 = (block_table_1, seqlens_1)

    ``o`` is the framework-provided final block-table output (the trtllm
    backend wires it to ``ctx.metadata.sparse_block_tables``). The op
    additionally writes ``ctx.metadata.sparse_seqlens`` as a side effect
    so the seqlens buffer the trtllm decode call reads is up to date.
    """

    _supported_formats: FrozenSet[FORMAT] = frozenset({FORMAT.RAGGED})

    def __init__(self):
        super().__init__()
        self.schedule = Schedule.S

    def profile(self, i_0, i_1, o: vTensor, ctx: Context):
        prefix = self._prefix()

        assert isinstance(i_0, tuple) and len(i_0) == 2, (
            f"{prefix}i_0 must be a (block_table, seqlens) tuple; got {type(i_0).__name__}"
        )
        assert isinstance(i_1, tuple) and len(i_1) == 2, (
            f"{prefix}i_1 must be a (block_table, seqlens) tuple; got {type(i_1).__name__}"
        )
        bt_0, sl_0 = i_0
        bt_1, sl_1 = i_1
        for name, t in (("bt_0", bt_0), ("sl_0", sl_0),
                        ("bt_1", bt_1), ("sl_1", sl_1),
                        ("o",    o)):
            assert isinstance(t, vTensor), (
                f"{prefix}{name} must be a vTensor; got {type(t).__name__}"
            )
        assert bt_0._format in self._supported_formats, (
            f"{prefix}bt_0._format={bt_0._format} not supported"
        )
        assert bt_1._format in self._supported_formats, (
            f"{prefix}bt_1._format={bt_1._format} not supported"
        )
        assert o._format in self._supported_formats, (
            f"{prefix}o._format={o._format} not supported"
        )

        backend = (
            getattr(ctx, "vortex_attention_backend", None) or "flashinfer"
        ).lower()
        assert backend == "trtllm", (
            f"{prefix}Union() is only supported under the trtllm attention "
            f"backend; got vortex_attention_backend={backend!r}."
        )

        # Claim ``o`` as the op's single output.
        ctx.output_tensor_to_op_list[o.tensor_id] = len(ctx.op_list)
        ctx.op_list.append(self)
        ctx.op_to_input_tensor_list.append([
            bt_0.tensor_id, sl_0.tensor_id,
            bt_1.tensor_id, sl_1.tensor_id,
        ])
        ctx.op_to_output_tensor_list.append([o.tensor_id])


