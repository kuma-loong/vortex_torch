import torch
from .context import Context
from ..abs import vOp, vTensor, FORMAT
from ..utils import Schedule
from typing import Tuple, Dict, Optional


class GeMM(vOp):
    r"""
    General matrix-matrix multiplication dispatcher for page/token-tiled layouts.

    This operator computes a batched GEMM of the form

    .. math::

        O_b = Y_b X_b^\top, \qquad b = 0, \dots, B-1,

    where, for each batch index :math:`b`,

    - :math:`X_b \in \mathbb{R}^{N_x \times K}`,
    - :math:`Y_b \in \mathbb{R}^{N_y \times K}`, and
    - :math:`O_b \in \mathbb{R}^{N_y \times N_x}`.

    In the logical 3D layout used by this dispatcher, the tensors have
    shapes

    .. math::

        X &\in \mathbb{R}^{B \times N_x \times K}, \\
        Y &\in \mathbb{R}^{B \times N_y \times K}, \\
        O &\in \mathbb{R}^{B \times N_y \times N_x},

    where the leading dimension :math:`B` is a batch-like axis typically
    derived from the runtime (for example,
    ``max_new_tokens_per_batch * head_num`` in an attention-style kernel).

    Dispatch is based on the triplet of tensor formats
    ``(x_format, y_format, o_format)`` and a registry mapping:

    .. code-block:: text

        (x_format, y_format, o_format) -> (impl, resolved_output_format)

    Policy
    ------
    - If ``output`` is ``None``:

      - :meth:`profile` selects an implementation with
        ``o_format == FORMAT.RAGGED``, i.e. a key
        ``(x_fmt, y_fmt, FORMAT.RAGGED)`` in :attr:`_impl_map`.
      - An internal buffer is allocated with logical shape
        ``[B, N_y, N_x]`` on the same device and with the same dtype as
        ``x``.

    - If ``output`` is provided:

      - :meth:`profile` requires an exact implementation key for
        ``(x_fmt, y_fmt, o_fmt)``.
      - The shape of ``output`` must be rank-3 with last two dimensions
        ``(N_y, N_x)``.
      - Device consistency is enforced across ``x``, ``y`` and ``output``.

    Additionally, the shared inner dimension :math:`K` must match:

    .. math::

        K_x = x.\text{shape}[2], \quad K_y = y.\text{shape}[2], \quad K_x = K_y.

    Attributes
    ----------
    _impl_map : Dict[Tuple[FORMAT, FORMAT, FORMAT], FORMAT]
        Dispatch table keyed by ``(x_format, y_format, o_format)``.
        Each entry maps to the resolved output format.
    output_format : Optional[FORMAT]
        The output tensor format as determined in :meth:`profile`.
    output_buffer : Optional[vTensor]
        Pure-metadata vTensor descriptor for the output (graph node).
    """

    _impl_map: Dict[Tuple[FORMAT, FORMAT, FORMAT], FORMAT] = {
        (FORMAT.PAGED,  FORMAT.PAGED,  FORMAT.PAGED):  FORMAT.PAGED,
        (FORMAT.PAGED,  FORMAT.PAGED,  FORMAT.RAGGED): FORMAT.RAGGED,
        (FORMAT.RAGGED, FORMAT.RAGGED, FORMAT.PAGED):  FORMAT.PAGED,
        (FORMAT.RAGGED, FORMAT.RAGGED, FORMAT.RAGGED): FORMAT.RAGGED,
    }

    def __init__(self):
        super().__init__()
        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[vTensor] = None
        # Cache GeMM fuses into the per-block kernel — see
        # ``cache.compiler.triton_impl.kernel_gen``.
        self.schedule = Schedule.W

    def _infer_output_format_ragged(
        self, x_fmt: FORMAT, y_fmt: FORMAT
    ) -> FORMAT:
        """Pick the RAGGED-output dispatch entry for ``(x_fmt, y_fmt)``."""
        key = (x_fmt, y_fmt, FORMAT.RAGGED)
        assert key in self._impl_map, (
            f"{self._prefix()}no RAGGED-output implementation for "
            f"(x_fmt={x_fmt}, y_fmt={y_fmt}). "
            f"Available keys: {list(self._impl_map.keys())}"
        )
        return self._impl_map[key]

    # --------------------------------------------------------------------- #
    # profile: validate, select impl/format, and optionally allocate output
    # --------------------------------------------------------------------- #
    def profile(
        self, x: vTensor, y: vTensor, output: Optional[vTensor], loc: torch.Tensor, ctx: Context
    ) -> vTensor:
        r"""
        Validate inputs, resolve the GEMM implementation and output format,
        and optionally allocate an internal output buffer.

        The logical shapes are:

        - ``x``: ``[B, N_x, K]``
        - ``y``: ``[B, N_y, K]``
        - ``output`` (if provided): ``[B_out, N_y, N_x]``

        with the constraint that the inner dimension :math:`K` matches:

        .. math::

            x.\text{shape}[2] = y.\text{shape}[2].

        The auxiliary tensor ``loc`` carries per-position or per-tile
        metadata used by the implementation (for example, page indices or
        tiling information); its shape and semantics are kernel-defined.

        Parameters
        ----------
        x : vTensor
            Right-hand operand in ``Y @ X^T``, with logical shape
            ``[B, N_x, K]``.

        y : vTensor
            Left-hand operand in ``Y @ X^T``, with logical shape
            ``[B, N_y, K]``.

        output : Optional[vTensor]
            Optional preallocated output tensor. If ``None``, an internal
            buffer with shape ``[B, N_y, N_x]`` is allocated using
            ``ctx.max_new_tokens_per_batch * ctx.head_num`` for the
            leading dimension and a RAGGED-output implementation is
            selected. If not ``None``, this tensor must have rank 3
            and last two dimensions ``(N_y, N_x)``, with a format
            compatible with :attr:`_impl_map`.

        loc : torch.Tensor
            Auxiliary tensor carrying metadata required by the GEMM
            implementation.

        ctx : Context
            Execution context that provides the runtime value of ``B``
            and is used for auxiliary memory accounting.

        Returns
        -------
        vTensor
            A :class:`vTensor` view representing the resolved output:
            either the provided ``output`` or an internally allocated
            buffer.

        Raises
        ------
        AssertionError
            If types, ranks, inner-dimension match, formats, shapes, or
            devices are incompatible, or if no implementation is found
            in :attr:`_impl_map`.
        """
        prefix = self._prefix()

        # --- type & rank checks ---
        assert isinstance(x, vTensor), f"{prefix}x must be vTensor, got {type(x)}"
        assert isinstance(y, vTensor), f"{prefix}y must be vTensor, got {type(y)}"
        assert isinstance(loc, torch.Tensor), f"{prefix}loc must be torch.Tensor, got {type(loc)}"

        assert x.dim() == 3, f"{prefix}x must be 3D, got ndim={x.dim()} shape={tuple(x.shape)}"
        assert y.dim() == 3, f"{prefix}y must be 3D, got ndim={y.dim()} shape={tuple(y.shape)}"

        # --- K match: x[..., K] == y[..., K] ---
        Kx, Ky = x.shape[2], y.shape[2]
        assert Kx == Ky, f"{prefix}K mismatch: x.shape[2]={Kx} vs y.shape[2]={Ky}"

        # Output logical shape: [B, Ny, Nx]
        Ny, Nx = y.shape[1], x.shape[1]
        x_fmt, y_fmt = x._format, y._format

        # Case A: output not provided -> pick RAGGED impl and build a
        # pure-metadata vTensor, then register it in the cache graph.
        if output is None:
            self.output_format = self._infer_output_format_ragged(x_fmt, y_fmt)

            B = ctx.max_new_tokens_per_batch * ctx.head_num
            self.output_buffer = vTensor(
                shape=(B, Ny, Nx),
                dtype=ctx.vortex_dtype,
                device=x.device,
                _format=self.output_format,
                tensor_id=len(ctx.tensor_list),
            )
            ctx.tensor_list.append(self.output_buffer)
            ctx.output_tensor_to_op_list.append(len(ctx.op_list))
            ctx.op_list.append(self)
            ctx.op_to_input_tensor_list.append([x.tensor_id, y.tensor_id])
            ctx.op_to_output_tensor_list.append([self.output_buffer.tensor_id])
            return self.output_buffer

        # Case B: output provided -> validate and select exact impl
        assert isinstance(output, vTensor), f"{prefix}output must be vTensor, got {type(output)}"
        assert output.dim() == 3, (
            f"{prefix}output must be 3D, got ndim={output.dim()} shape={tuple(output.shape)}"
        )

        o_fmt = output._format
        key = (x_fmt, y_fmt, o_fmt)
        assert key in self._impl_map, (
            f"{prefix}no implementation for (x_fmt={x_fmt}, y_fmt={y_fmt}, o_fmt={o_fmt}). "
            f"Available keys: {list(self._impl_map.keys())}"
        )
        self.output_format = self._impl_map[key]

        # Shape consistency: GEMM yields [*, Ny, Nx]
        assert output.shape[1] == Ny and output.shape[2] == Nx, (
            f"{prefix}output shape mismatch. Expected (*,{Ny},{Nx}), got {tuple(output.shape)}"
        )

        # Device consistency check
        assert x.device == y.device == output.device, (
            f"{prefix}x, y, and output must be on the same device "
            f"(x.device={x.device}, y.device={y.device}, output.device={output.device})"
        )

        # Register in the cache graph (caller-provided output path).
        ctx.output_tensor_to_op_list[output.tensor_id] = len(ctx.op_list)
        ctx.op_list.append(self)
        ctx.op_to_input_tensor_list.append([x.tensor_id, y.tensor_id])
        ctx.op_to_output_tensor_list.append([output.tensor_id])

        return output
