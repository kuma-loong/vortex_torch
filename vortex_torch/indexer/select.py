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
    r"""Block-table top-k for the trtllm backend (k explicit, multi-output).

    Distinct from :class:`vortex_torch.indexer.topK` (lowercase ``t``,
    a.k.a. *TopKOut*): :class:`topK` writes selected block ids into a
    caller-supplied ``o`` and has a *predetermined* k (sourced from
    ``ctx.topk_val`` / ``ctx.sparse_kv_indptr`` diff at runtime). This
    :class:`TopK` is **trtllm-only**, takes the top-k budget ``k`` as an
    *explicit* constructor argument, and **returns its own
    ``(block_tables, seqlens)`` tuple** — both auto-allocated as
    intermediate buffers by the compiler's ``memory_initiazation_str``.
    A separate op (developed elsewhere) is responsible for copying these
    intermediates into the final output buffers consumed by
    ``trtllm_batch_decode_with_kv_cache``.

    Per-row behavior, with ``bos = ctx.block_reserved_bos``,
    ``eos = ctx.block_reserved_eos``, ``block_len = ceil(dense_seqlens[i] / block_size)``:

      * If ``block_len <= bos + k + eos`` → copy the entire dense row
        into ``block_tables[i, :block_len]`` and set ``seqlens[i] =
        dense_seqlens[i]``. No selection — the row is already small
        enough to fit the budget.
      * Otherwise → write
          - ``[0:bos)``           ← first ``bos`` dense blocks
          - ``[bos:bos+k)``       ← top-``k`` blocks by score over
                                    ``[bos, block_len-eos)`` of the dense row
          - ``[bos+k:bos+k+eos)`` ← last ``eos`` dense blocks
        and set ``seqlens[i] = (bos+k+eos-1) * block_size + last_block_len``
        where ``last_block_len`` is the per-row trailing block's token
        count (so the last block — the dense path's last — contributes
        the exact token count trtllm decode expects).

    Constructor
    -----------
    ``k`` : int — number of *selected* blocks (excludes the reserved
    BOS+EOS slots). The total per-row sparse block count is
    ``bos + k + eos``.

    Invocation
    ----------
    ::

        block_tables, seqlens = self.topk(score, ctx=ctx)

    ``score`` is the per-block RAGGED scores ``[S_pack, 1, 1]``. The op
    returns two new intermediate vTensors:

      * ``block_tables`` — RAGGED int32, leading dim ``ctx.max_num_blocks``
        (= ``eff_bs * max_blocks_per_seq``). The CUDA kernel addresses it
        as the contiguous 2D ``[eff_bs, max_blocks_per_seq]`` memory it is.
      * ``seqlens`` — BATCHED int32, leading dim ``ctx.max_bs * num_kv_heads``.

    Both are allocated by ``interface.generate_entry_point``'s
    ``memory_initiazation_str`` (no aliasing with ``ctx.metadata``).
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
