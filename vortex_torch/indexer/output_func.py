import torch
from typing import Any, Tuple, Dict, Callable, Optional
from ..abs import vOp
from vortex_torch_C import topk_output
from .context import Context
from ..abs import vTensor, FORMAT


class topK(vOp):
    """
    Top-K dispatcher.
    - Dispatches by input format only (currently supports FORMAT.RAGGED).
    - The op is out-of-place with respect to `o`, but fills `o` in-place.
    - `profile()` validates and selects implementation; no allocation is done here.
    """

    # Dispatch by input format; only RAGGED is supported for now.
    _impl_map: Dict[FORMAT, Callable] = {
        FORMAT.RAGGED: topk_output,
    }

    def __init__(self):
        super().__init__()
        self.impl: Optional[Callable] = None

    # ---------------- profile ----------------
    def profile(self, x: vTensor, o: vTensor, ctx: Context) -> None:
        """
        Validate shapes/formats and select implementation. Does not allocate or return.
        """
        prefix = self._prefix()

        # ---- type checks ----
        assert isinstance(x, vTensor), f"{prefix}profile expects x to be vTensor, got {type(x)}"
        assert isinstance(o, vTensor), f"{prefix}profile expects o to be vTensor, got {type(o)}"

        # ---- rank checks ----
        assert x.dim() == 3, f"{prefix}expected x to be 3D, got ndim={x.dim()} shape={tuple(x.shape)}"
        assert o.dim() == 3, f"{prefix}expected o to be 3D, got ndim={o.dim()} shape={tuple(o.shape)}"

        # ---- shape checks for x ----
        # x is expected to carry per-token scalars at dims (1,2) for top-k selection parameters
        assert x.shape[1] == 1 and x.shape[2] == 1, (
            f"{prefix}expected x.shape[1] == x.shape[2] == 1, got {tuple(x.shape)}"
        )

        # ---- implementation availability ----
        x_fmt = x._format
        assert x_fmt in self._impl_map, (
            f"{prefix}no implementation for x._format={x_fmt}. "
            f"Available: {list(self._impl_map.keys())}"
        )
        self.impl = self._impl_map[x_fmt]

        # ---- optional sanity checks on `o` ----
        # We only assert device consistency and leave exact (N,D) to upstream contract.
        assert x.device == o.device, (
            f"{prefix}x and o must be on the same device "
            f"(x.device={x.device}, o.device={o.device})"
        )
        

    # ---------------- execute ----------------
    def execute(self, x: torch.Tensor, o: torch.Tensor, ctx: Context) -> torch.Tensor:
        """
        Run the selected implementation. Fills `o` in-place and returns it.
        Expected implementation signature:
            impl(x, dense_kv_indptr, sparse_kv_indptr, dense_kv_indices,
                 o, batch_heads, topk_val, page_reserved_bos, page_reserved_eos,
                 max_num_pages_per_request)
        """
        prefix = self._prefix()
        assert self.impl is not None, f"{prefix}execute called before profile() (impl is None)"

        # Call the selected kernel/implementation
        self.impl(
            x,
            ctx.dense_kv_indptr,
            ctx.sparse_kv_indptr,
            ctx.dense_kv_indices,
            o,
            ctx.batch_size * ctx.num_kv_heads,
            ctx.topk_val,
            ctx.page_reserved_bos,
            ctx.page_reserved_eos,
            ctx.max_num_pages_per_request,
        )
