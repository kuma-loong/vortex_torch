import torch
from typing import Tuple, Optional, Union, Iterable
from .context import Context
from ..abs import vTensor, FORMAT, vOp
from ..utils import Schedule


class Kron(vOp):
    r"""
    Kronecker product over a configurable subset of the inner axes.

    Given two rank-3 inputs

    .. math::

        X \in \mathbb{R}^{S \times x_1 \times x_2}, \qquad
        Y \in \mathbb{R}^{S \times y_1 \times y_2},

    this operator takes a Kronecker product along the axes listed in
    :attr:`dim` (a subset of ``{1, 2}``), and an elementwise (broadcast)
    product along every other inner axis. The leading axis :math:`S` is
    always elementwise.

    For axis ``a`` in ``{1, 2}``:

    - if ``a`` is in :attr:`dim`, the output size is ``x.shape[a] * y.shape[a]``;
    - otherwise, ``x.shape[a]`` must equal ``y.shape[a]`` or one of them
      must be ``1`` (broadcast), and the output size is
      ``max(x.shape[a], y.shape[a])``.

    Examples
    --------
    All three forms below select different output layouts from the same
    pair of inputs::

        Kron(dim=(1, 2))   # full Kron:    [B, x1*y1, x2*y2]
        Kron(dim=1)        # row-Kron:     [B, x1*y1, D]      (x2==y2 or 1)
        Kron(dim=2)        # col-Kron:     [B, C,     x2*y2]  (x1==y1 or 1)

    With the explicit index formula

    .. math::

        \text{out}[s, \cdot, \cdot]
        \;=\; \mathrm{kron\_along}_{\text{dim}}\!\bigl(X[s], Y[s]\bigr).

    For ``dim == (1, 2)``:

    .. math::

        \text{out}[s,\, i \cdot y_1 + j,\, k \cdot y_2 + l]
        = X[s, i, k] \cdot Y[s, j, l].

    For ``dim == (1,)`` (with ``x_2 = y_2 = D`` after broadcast):

    .. math::

        \text{out}[s,\, i \cdot y_1 + j,\, d]
        = X[s, i, d] \cdot Y[s, j, d].

    For ``dim == (2,)`` (with ``x_1 = y_1 = C`` after broadcast):

    .. math::

        \text{out}[s,\, c,\, k \cdot y_2 + l]
        = X[s, c, k] \cdot Y[s, c, l].

    Output format rule: ``BATCHED`` iff both inputs are ``BATCHED``
    (each tile is one ``(batch, head)`` summary, stored once per
    workload group via the kernel's first-workload gate); otherwise
    ``RAGGED`` (per-workload tile). ``BATCHED`` inputs broadcast along
    the workload axis (leading block dim == 1) so they can pair with
    non-``BATCHED`` partners. Format compatibility is enforced by the
    compiler's per-workload kernel.

    Parameters
    ----------
    dim : int or Iterable[int], optional
        The inner axis (or axes) along which to take the Kronecker
        product. Each entry must be ``1`` or ``2``; duplicates are
        rejected. Default is ``(1, 2)`` (full Kronecker product).

    Attributes
    ----------
    dim : Tuple[int, ...]
        Sorted tuple of axes that get the Kronecker expansion.
    output_format : Optional[FORMAT]
        Resolved output format from :meth:`profile`.
    output_buffer : Optional[vTensor]
        Pure-metadata vTensor descriptor for the output (graph node).
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
