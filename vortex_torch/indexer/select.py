"""Selection-style indexer ops.

Ops in this file *produce intermediate* block-table / seqlens tensors —
they don't write into the framework-provided ``o`` (which the existing
:class:`vortex_torch.indexer.topK` / "TopKOut" does). Use a follow-up op
(forthcoming) to copy the intermediates produced here into the final
``o`` / ``ctx.metadata.sparse_seqlens`` buffers consumed by
``trtllm_batch_decode_with_kv_cache``.
"""
from __future__ import annotations

from typing import FrozenSet

import torch

from ..abs import FORMAT, vOp, vTensor
from ..utils import Schedule
from .context import Context


class TopK(vOp):
    r"""
    Explicit-``k`` block-table top-k (trtllm backend, two outputs).

    :Math:
        For request row :math:`i` with score :math:`s` over its dense blocks,
        ``bos`` / ``eos`` reserved blocks and
        :math:`L_i = \lceil \text{seqlen}_i / \text{block\_size}\rceil`:

        .. math::

            \mathcal{B}_i = \begin{cases}
                \{0,\dots,L_i-1\}, & L_i \le \text{bos}+k+\text{eos}, \\[2pt]
                [0,\text{bos}) \;\cup\; \operatorname*{top\text{-}k}_{\,\text{bos}\le j<L_i-\text{eos}} s_j
                \;\cup\; [L_i-\text{eos}, L_i), & \text{otherwise}.
            \end{cases}

        Rows that already fit the budget are copied dense; larger rows keep
        the leading ``bos`` and trailing ``eos`` blocks and fill the middle
        with the top-``k`` by score.
    :__init__: ``TopK(k)`` — number of *selected* blocks (excludes the
        reserved BOS/EOS); per-row sparse block count is ``bos + k + eos``.
    :__call__: ``block_tables, seqlens = op(score, ctx=ctx)`` — ``score`` RAGGED
        ``[S, 1, 1]`` → ``block_tables`` (RAGGED int32) and ``seqlens``
        (BATCHED int32), both auto-allocated. Feed the pair to
        :class:`Union` before ``trtllm_batch_decode_with_kv_cache``.
    :Note: **trtllm only** — asserts under flashinfer; use :func:`topK` for
        flashinfer / CSR layouts.
    """

    _supported_formats: FrozenSet[FORMAT] = frozenset({FORMAT.RAGGED})

    def __init__(self, k: int):
        super().__init__()
        try:
            k_int = int(k)
        except (TypeError, ValueError) as e:
            raise ValueError(f"TopK: k must be an integer, got {k!r}") from e
        if k_int < 1:
            raise ValueError(f"TopK: k must be >= 1, got {k_int}")
        self.k = k_int
        self.schedule = Schedule.S
        self.block_tables_buffer: vTensor = None  # filled in profile()
        self.seqlens_buffer: vTensor = None

    def profile(self, x: vTensor, ctx: Context):
        prefix = self._prefix()

        # ---- input validation (mirrors topK) ----
        assert isinstance(x, vTensor), (
            f"{prefix}profile expects x to be vTensor, got {type(x)}"
        )
        assert x.dim() == 3, (
            f"{prefix}expected x to be 3D, got ndim={x.dim()} shape={tuple(x.shape)}"
        )
        assert x.shape[1] == 1 and x.shape[2] == 1, (
            f"{prefix}expected x.shape[1] == x.shape[2] == 1, got {tuple(x.shape)}"
        )
        assert x._format in self._supported_formats, (
            f"{prefix}no implementation for x._format={x._format}. "
            f"Supported: {sorted(self._supported_formats, key=lambda f: f.value)}"
        )

        # ---- trtllm-only ----
        backend = (
            getattr(ctx, "vortex_attention_backend", None) or "flashinfer"
        ).lower()
        assert backend == "trtllm", (
            f"{prefix}TopK(k) is only supported under the trtllm attention "
            f"backend; got vortex_attention_backend={backend!r}. Use the "
            f"regular ``topK()`` op for flashinfer / CSR layouts."
        )

        # ---- allocate the two intermediate outputs ----
        # block_tables: RAGGED int32. memory_init picks ``leading =
        # ctx.max_num_blocks`` (= eff_bs * max_blocks_per_seq) for RAGGED
        # intermediates, which is exactly the byte count the CUDA kernel
        # addresses as ``[eff_bs, max_blocks_per_seq]`` contiguous memory.
        self.block_tables_buffer = vTensor(
            shape=(0, 1, 1),
            dtype=torch.int32,
            device=x.device,
            _format=FORMAT.RAGGED,
            tensor_id=len(ctx.tensor_list),
        )
        ctx.tensor_list.append(self.block_tables_buffer)
        ctx.output_tensor_to_op_list.append(len(ctx.op_list))

        # seqlens: BATCHED int32. memory_init picks ``leading = ctx.max_bs *
        # ctx.num_kv_heads`` for BATCHED intermediates.
        self.seqlens_buffer = vTensor(
            shape=(0, 1, 1),
            dtype=torch.int32,
            device=x.device,
            _format=FORMAT.BATCHED,
            tensor_id=len(ctx.tensor_list),
        )
        ctx.tensor_list.append(self.seqlens_buffer)
        ctx.output_tensor_to_op_list.append(len(ctx.op_list))

        # Two-output op — multi-output is supported by Graph since the
        # migration in ``compiler/graph.py``.
        ctx.op_list.append(self)
        ctx.op_to_input_tensor_list.append([x.tensor_id])
        ctx.op_to_output_tensor_list.append([
            self.block_tables_buffer.tensor_id,
            self.seqlens_buffer.tensor_id,
        ])

        return self.block_tables_buffer, self.seqlens_buffer


__all__ = ["TopK"]
