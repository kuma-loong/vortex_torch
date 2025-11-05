import torch
from typing import Any, Tuple, Dict, Callable
from ..abs import vOp
from vortex_torch_C import topk_output
from .context import Context
from ..abs import vTensor, FORMAT


class topK(vOp):
    # Dispatch by input format; only RAGGED is supported for now.
    _impl_map: Dict[FORMAT, Callable] = {
        FORMAT.RAGGED: topk_output,
    }

    def __init__(self):
        super().__init__()
        self.impl: Callable | None = None

    def profile(self, x: vTensor, o: vTensor, ctx: Context):
        """Validate shapes/formats and select implementation. Does not allocate or return."""
        # Type checks: inputs must be vTensor by contract
        assert isinstance(x, vTensor), f"topK.profile expects x to be vTensor, got {type(x)}"
        assert isinstance(o, vTensor), f"topK.profile expects o to be vTensor, got {type(o)}"

        # Rank and shape checks
        assert len(x.shape) == 3, f"Expected x to be 3D, got x.ndim={x.dim()} with shape={tuple(x.shape)}"
        # Expect singleton dims at positions 1 and 2
        assert x.shape[1] == 1 and x.shape[2] == 1, (
            f"Expected x.shape[1] == x.shape[2] == 1, got shape={tuple(x.shape)}"
        )

        # Implementation availability
        assert x._format in self._impl_map, (
            f"No implementation for x._format={x._format}. "
            f"Available: {list(self._impl_map.keys())}"
        )

        # Select implementation
        self.impl = self._impl_map[x._format]


    def execute(self, x: torch.Tensor, o: torch.Tensor, ctx: Context):
        """Run the selected implementation. Returns nothing (in-place fill of `o`)."""
        # Must have run profile() first
        assert self.impl is not None, "topK.execute called before profile(): impl is None"

        

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
