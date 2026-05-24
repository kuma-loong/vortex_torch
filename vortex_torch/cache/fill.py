import torch
from ..abs import vOp, vTensor, FORMAT
from .context import Context
from ..utils import Schedule
from typing import FrozenSet


class Fill(vOp):
    r"""
    In-place page-wise fill with a scalar.

    :Math:
        .. math::

            Y_{b,n,d} = \alpha \quad\text{(over the selected page tiles).}
    :__init__: ``Fill(alpha=0.0)`` — scalar fill value :math:`\alpha`.
    :__call__: ``op(x, loc=loc, ctx=ctx)`` — overwrites the page tiles of ``x``
        (``PAGED``) at ``loc`` with :math:`\alpha`. A **pure producer**: its
        output *is* ``x`` (used e.g. to zero a cache field before
        :class:`vortex_torch.indexer.Save` / :class:`vortex_torch.indexer.Load`).
    :Note: ``PAGED`` input only.
    """

    # Supported input formats; only PAGED is wired up for now.
    _supported_formats: FrozenSet[FORMAT] = frozenset({FORMAT.PAGED})

    def __init__(self, alpha: float = 0.0):
        super().__init__()
        self.alpha = alpha
        # Cache fill fuses into the per-block kernel — see
        # ``cache.compiler.triton_impl.kernel_gen``.
        self.schedule = Schedule.W

    # ---------------- profile ----------------
    def profile(self, x: vTensor, loc: torch.Tensor, ctx: Context) -> vTensor:
        r"""
        Validate inputs and register this op in the cache graph.

        ``Fill`` is a pure producer of ``x``: it overwrites ``x`` with a
        constant and claims ``x``'s producer slot (mirroring
        :class:`indexer.save_load.Save`). The same :class:`vTensor` is
        returned so downstream ops can continue to reference ``x``.
        """
        prefix = self._prefix()

        assert isinstance(x, vTensor), (
            f"{prefix}profile expects x to be vTensor, got {type(x)}"
        )
        assert isinstance(loc, torch.Tensor), (
            f"{prefix}profile expects loc to be torch.Tensor, got {type(loc)}"
        )
        assert x.dim() == 3, (
            f"{prefix}expected 3D input [B, D0, D1], "
            f"got ndim={x.dim()} shape={tuple(x.shape)}"
        )

        assert x._format in self._supported_formats, (
            f"{prefix}no implementation for x_fmt={x._format}. "
            f"Supported: {sorted(self._supported_formats, key=lambda f: f.value)}"
        )

        # Register in the cache graph as a pure producer of ``x``.
        ctx.output_tensor_to_op_list[x.tensor_id] = len(ctx.op_list)
        ctx.op_list.append(self)
        ctx.op_to_input_tensor_list.append([])
        ctx.op_to_output_tensor_list.append([x.tensor_id])

        return x
