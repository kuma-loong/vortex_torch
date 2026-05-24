import torch
from typing import Tuple, Optional, Union, Iterable
from .context import Context
from ..abs import vTensor, FORMAT, vOp
from ..utils import Schedule


class Kron(vOp):
    r"""
    Kronecker product over a configurable subset of the inner axes.

    The chosen inner axes (``dim``) are Kronecker-expanded; any other inner
    axis is multiplied elementwise (with broadcasting), and the leading
    :math:`S` axis is always elementwise.

    :Math:
        For :math:`X\in\mathbb{R}^{S\times x_1\times x_2}` and
        :math:`Y\in\mathbb{R}^{S\times y_1\times y_2}`:

        .. math::

            \begin{aligned}
            \text{dim}=(1,2):\quad & O_{s,\,i\,y_1+j,\,k\,y_2+l} = X_{s,i,k}\,Y_{s,j,l}, \\
            \text{dim}=(1,):\quad  & O_{s,\,i\,y_1+j,\,d} = X_{s,i,d}\,Y_{s,j,d}, \\
            \text{dim}=(2,):\quad  & O_{s,\,c,\,k\,y_2+l} = X_{s,c,k}\,Y_{s,c,l}.
            \end{aligned}
    :__init__: ``Kron(dim=(1, 2))`` — inner axis/axes to expand, each ``1`` or
        ``2`` (non-listed axes must be equal or broadcastable).
    :__call__: ``o = op(x, y, ctx=ctx)`` — ``x`` ``[S, x_1, x_2]``, ``y``
        ``[S, y_1, y_2]``; an expanded axis has output size
        ``x.shape[a]*y.shape[a]``, a broadcast axis ``max(x.shape[a],
        y.shape[a])``. Output is ``BATCHED`` iff both inputs are.
    """

    def __init__(self, dim: Union[int, Iterable[int]] = (1, 2)):
        super().__init__()
        # Normalize ``dim`` to a sorted tuple of unique ints in {1, 2}.
        if isinstance(dim, int):
            dim_tuple: Tuple[int, ...] = (dim,)
        else:
            dim_tuple = tuple(dim)
        cls = self.__class__.__name__
        assert len(set(dim_tuple)) == len(dim_tuple), (
            f"{cls}.__init__: duplicate axes in dim={dim_tuple!r}"
        )
        for a in dim_tuple:
            assert isinstance(a, int) and a in (1, 2), (
                f"{cls}.__init__: dim entries must be 1 or 2, got {a!r}"
            )
        assert len(dim_tuple) >= 1, (
            f"{cls}.__init__: dim must list at least one axis, got empty"
        )
        self.dim: Tuple[int, ...] = tuple(sorted(dim_tuple))

        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[vTensor] = None
        # Fused into the per-workload kernel — the per-block tiles are
        # already loaded as 3D ``(W, C, D)`` blocks and the Kronecker
        # expansion stays tile-local.
        self.schedule = Schedule.W

    # ---------------- profile ----------------
    def profile(self, x: vTensor, y: vTensor, ctx: Context) -> vTensor:
        r"""
        Validate inputs, allocate the output buffer, and return a
        :class:`vTensor` view.

        For each inner axis ``a`` in ``{1, 2}``:

        - if ``a`` is in :attr:`dim`, the output size is the product
          ``x.shape[a] * y.shape[a]`` (Kronecker expansion);
        - otherwise, ``x.shape[a]`` and ``y.shape[a]`` must be equal or
          one must be ``1`` (broadcast), and the output size is
          ``max(x.shape[a], y.shape[a])``.

        Parameters
        ----------
        x : vTensor
            Left-hand input with logical shape ``[S, x1, x2]``.
        y : vTensor
            Right-hand input with logical shape ``[S, y1, y2]``.
        ctx : Context
            Execution context tracking graph structure / aux memory.

        Returns
        -------
        vTensor
            A :class:`vTensor` view wrapping the internally allocated
            output buffer.

        Raises
        ------
        AssertionError
            If types are not :class:`vTensor`, if ranks are not 3, if a
            non-Kron axis is not equal/broadcastable, or if ``x`` and
            ``y`` live on different devices.
        """
        prefix = self._prefix()

        # Type & rank checks
        assert isinstance(x, vTensor), f"{prefix}profile expects x to be vTensor, got {type(x)}"
        assert isinstance(y, vTensor), f"{prefix}profile expects y to be vTensor, got {type(y)}"
        assert x.dim() == 3 and y.dim() == 3, (
            f"{prefix}expected 3D inputs [S, C, D]; got x.ndim={x.dim()}, y.ndim={y.dim()}"
        )

        # Output is BATCHED iff both inputs are BATCHED; otherwise RAGGED.
        self.output_format = (
            FORMAT.BATCHED
            if (x._format == FORMAT.BATCHED and y._format == FORMAT.BATCHED)
            else FORMAT.RAGGED
        )

        # Device consistency
        assert x.device == y.device, (
            f"{prefix}x and y must be on the same device "
            f"(x.device={x.device}, y.device={y.device})"
        )

        # Per-axis output size: Kron-expanded for axes in ``dim``,
        # broadcast-elementwise otherwise.
        out_inner: Tuple[int, ...] = tuple(
            self._resolve_axis(x.shape[a], y.shape[a], a) for a in (1, 2)
        )
        C_out, D_out = out_inner

        # Pure-metadata vTensor — no torch.empty allocation needed.
        self.output_buffer = vTensor(
            shape=(0, C_out, D_out),
            dtype=ctx.vortex_dtype,
            device=x.device,
            _format=self.output_format,
            tensor_id=len(ctx.tensor_list),
        )

        # Track graph structure in the context (mirror Elementwise_Binary).
        ctx.tensor_list.append(self.output_buffer)
        ctx.output_tensor_to_op_list.append(len(ctx.op_list))
        ctx.op_list.append(self)
        ctx.op_to_input_tensor_list.append([x.tensor_id, y.tensor_id])
        ctx.op_to_output_tensor_list.append([self.output_buffer.tensor_id])

        return self.output_buffer

    # ---------------- helpers ----------------
    def _resolve_axis(self, sx: int, sy: int, axis: int) -> int:
        """Compute the output size for one inner axis."""
        if axis in self.dim:
            return sx * sy
        # Elementwise axis: must be equal or one of them must be 1.
        assert (sx == sy) or (sx == 1) or (sy == 1), (
            f"{self._prefix()}dim={self.dim}: axis {axis} not in dim and not "
            f"broadcastable (x.shape[{axis}]={sx}, y.shape[{axis}]={sy})"
        )
        return max(sx, sy)
