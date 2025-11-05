import torch
from .triton_kernels.mv_impl import triton_mv
from typing import Tuple, Dict, Callable
from .context import Context
from ..abs import vTensor, as_vtensor, FORMAT, vOp


class GeMV(vOp):
    
    """
    General matrix–vector multiplication (GEMV) operator with format-aware dispatch.

    This operator multiplies a matrix by a vector under the two-phase ``vOp``
    contract: a lightweight **profiling** phase that produces a metadata tensor,
    and an **execution** phase that performs the actual computation.
    
    .. math::

        o = y_i x_i^T, \quad y_i \in \mathbb{R}^{S \times 1 \times D}, \quad x_i \in \mathbb{R}^{1 \times 1 \times D}.

    Implements:
        - ``profile(x: :class:`~vortex_torch.vTensor`, y: :class:`~vortex_torch.vTensor`, ctx) -> :class:`~vortex_torch.vTensor``
        - ``execute(x: torch.Tensor, y: torch.Tensor, ctx) -> torch.Tensor``

    Args:
        x (:class:`~vortex_torch.vTensor` | torch.Tensor):
            Vector operand. In the profiling phase, a :class:`~vortex_torch.vTensor`;
            in the execution phase, a ``torch.Tensor`` with shape ``(B, 1, D)``.
        y (:class:`~vortex_torch.vTensor` | torch.Tensor):
            Matrix operand. In the profiling phase, a :class:`~vortex_torch.vTensor`;
            in the execution phase, a ``torch.Tensor`` with shape ``(S_{pack}, 1, D)``.

    Returns:
        :class:`~vortex_torch.vTensor` | torch.Tensor:
            A vector tensor of shape ``(S_{pack}, 1, 1)`` representing the GEMV
            result, depending on the phase.

    Notes:
        * No dtype or device dispatch is performed; the caller must ensure that
          operands share compatible dtype and device.

    """

    # Implementation dispatch table: keyed only by (x_format, y_format).
    _impl_map: Dict[Tuple[FORMAT, FORMAT], Tuple[Callable, FORMAT]] = {
        (FORMAT.BATCHED, FORMAT.PAGED): (triton_mv, FORMAT.RAGGED),
        # Add more (x_fmt, y_fmt) -> (impl, out_fmt) pairs as needed.
    }

    def __init__(self):
        super().__init__()
        self.impl: Callable = None
        self.output_format: FORMAT | None = None
        self.output_buffer: torch.Tensor | None = None

    # ---------------- profile ----------------
    def profile(self, x: vTensor, y: vTensor, ctx: Context) -> vTensor:
        # Inputs must be vTensor per the system rule (no mixing with plain tensors).
        assert isinstance(x, vTensor), f"GeMV.profile expects x to be vTensor, got {type(x)}"
        assert isinstance(y, vTensor), f"GeMV.profile expects y to be vTensor, got {type(y)}"

        # Dimensionality checks (expect [*, 1, *] with both tensors being 3D).
        assert len(x.shape) == len(y.shape) == 3, (
            f"Expected 3D inputs (B, 1, D). Got x.ndim={x.dim()}, y.ndim={y.dim()}"
        )
        assert x.shape[1] == 1, f"Expected x.shape[1] == 1, got x.shape={tuple(x.shape)}"
        assert y.shape[1] == 1, f"Expected y.shape[1] == 1, got y.shape={tuple(y.shape)}"
        assert x.shape[2] == y.shape[2], (
            f"Last dimension must match: x.shape[2]={x.shape[2]} vs y.shape[2]={y.shape[2]}"
        )

        # Dispatch by (format_x, format_y).
        x_fmt, y_fmt = x._format, y._format
        assert (x_fmt, y_fmt) in self._impl_map, (
            f"No implementation for (x_fmt={x_fmt}, y_fmt={y_fmt}). "
            f"Available keys: {list(self._impl_map.keys())}"
        )

        self.impl, self.output_format = self._impl_map[(x_fmt, y_fmt)]

        # Allocate output buffer on x.device, with ctx.indexer_dtype (as per your original logic).
        self.output_buffer = torch.empty(
            (ctx.max_num_pages, y.shape[1], x.shape[1]),
            device=x.device,
            dtype=ctx.indexer_dtype,
        )

        # Account auxiliary memory.
        ctx.add_aux_memory(self.output_buffer)

        # Return vTensor view carrying the dispatched output format.
        return as_vtensor(self.output_buffer, self.output_format)

    # ---------------- execute ----------------
    def execute(self, x: torch.Tensor, y: torch.Tensor, ctx: Context) -> torch.Tensor:
        # Must call profile() first to set impl and output buffer.
        assert self.impl is not None, "GeMV.execute called before profile(): impl is None"
        assert self.output_buffer is not None, "GeMV.execute called before profile(): output_buffer is None"

        # Run the selected implementation. Keep the original argument order and return type.
        self.impl(
            x, y, self.output_buffer,
            ctx.dense_kv_indices,
            ctx.winfo_q_indices,
            ctx.winfo_kv_offsets,
            ctx.winfo_kv_lens,
            ctx.winfo_num_workloads,
            ctx.max_chunk_size,
            ctx.num_sms,
        )

        # execute returns a plain torch.Tensor by design.
        return self.output_buffer