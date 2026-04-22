import torch
from typing import Dict, Optional
from .context import Context
from ..abs import vTensor, as_vtensor, FORMAT, vOp
from ..utils import Schedule

class Save(vOp):
    r"""
    Format-aware save (copy/convert) dispatcher.

    This operator copies or converts data from an input tensor ``x`` into a
    preallocated output tensor ``o``. Both tensors are treated as rank-3
    objects with logical shape

    .. math::

        [S, D_0, D_1],

    where :math:`S` is a generic leading dimension (it may represent a batch,
    a sequence length, or a page count, depending on the surrounding runtime).

    Key properties:

    - Dispatch is keyed **only** by the format of ``x`` (``x._format``).
    - No internal buffer is allocated; the provided output tensor ``o`` is
      validated and written into directly.
    - The output format is determined by the dispatch table and must match
      ``o._format``.

    Attributes
    ----------
    _impl_map : Dict[FORMAT, FORMAT]
        Dispatch table keyed by ``x_format``. Each entry maps to the
        resolved output format.

    output_format : Optional[FORMAT]
        The expected output format for ``o`` as determined in :meth:`profile`.
    """

    # Dispatch table keyed by x_format -> resolved output format.
    _impl_map: Dict[FORMAT, FORMAT] = {
        FORMAT.RAGGED: FORMAT.PAGED,
        # Add more entries if you support other formats.
    }

    def __init__(self):
        super().__init__()
        self.output_format: Optional[FORMAT] = None
        self.schedule = Schedule.W

    # ---------------- profile ----------------
    def profile(self, x: vTensor, o: vTensor, ctx: Context) -> vTensor:
        r"""
        Validate inputs, resolve the implementation, and return ``o`` as
        the output view.

        This method checks:

        - that both ``x`` and ``o`` are rank-3 ``vTensor`` instances with
          logical shape ``[S, D_0, D_1]``,
        - that their inner dimensions ``D_0`` and ``D_1`` match,
        - that a save implementation is registered for ``x._format``, and
        - that ``o._format`` matches the format required by the dispatch.

        No new buffers are allocated; the provided ``o`` will be used as the
        destination.

        Parameters
        ----------
        x : vTensor
            Input tensor to be copied/converted. Expected logical shape
            ``[S, D_0, D_1]``.

        o : vTensor
            Preallocated output tensor. Must have logical shape compatible
            with ``x`` (matching ``D_0`` and ``D_1``) and the format expected
            by the selected implementation.

        ctx : Context
            Execution context (included for API symmetry; not used for buffer
            allocation in this dispatcher).

        Returns
        -------
        vTensor
            The same object as ``o``, returned as the resolved output view.

        Raises
        ------
        AssertionError
            If types, ranks, shapes, formats, or devices are incompatible,
            or if no implementation is registered for ``x._format``.
        """
        prefix = self._prefix()

        # Type & rank checks
        assert isinstance(x, vTensor), f"{prefix}profile expects x to be vTensor, got {type(x)}"
        assert isinstance(o, vTensor), f"{prefix}profile expects o to be vTensor, got {type(o)}"
        assert x.dim() == 3, f"{prefix}expected 3D x [S, D0, D1], got {tuple(x.shape)}"
        assert o.dim() == 3, f"{prefix}expected 3D o [S, D0, D1], got {tuple(o.shape)}"

        # Shape checks: D0/D1 must match (S may differ by layout; implementation handles it)
        assert x.shape[1] == o.shape[1], (
            f"{prefix}expected matching D0: x.shape[1]={x.shape[1]} vs o.shape[1]={o.shape[1]}"
        )
        assert x.shape[2] == o.shape[2], (
            f"{prefix}expected matching D1: x.shape[2]={x.shape[2]} vs o.shape[2]={o.shape[2]}"
        )

        # Dispatch by x format
        x_fmt = x._format
        assert x_fmt in self._impl_map, (
            f"{prefix}no implementation for x_fmt={x_fmt}. "
            f"Available keys: {list(self._impl_map.keys())}"
        )
        self.output_format = self._impl_map[x_fmt]

        # Output format must match the resolved format from dispatch
        assert o._format == self.output_format, (
            f"{prefix}output format mismatch. Expected {self.output_format}, got {o._format}"
        )

        # Device consistency
        assert x.device == o.device, (
            f"{prefix}x and o must be on the same device "
            f"(x.device={x.device}, o.device={o.device})"
        )

        # Track in the context graph. Save claims `o` as its produced tensor
        # (mirrors the topK pattern): we override `o`'s producer slot with
        # this op so that downstream consumers correctly depend on Save.
        ctx.output_tensor_to_op_list[o.tensor_id] = len(ctx.op_list)
        ctx.op_list.append(self)
        ctx.op_to_input_tensor_list.append([x.tensor_id])
        ctx.op_to_output_tensor_list.append([o.tensor_id])

        return o


class Load(vOp):
    r"""
    Format-aware load (copy/convert) dispatcher.

    This operator copies or converts data from an input tensor ``x`` into an
    internally allocated output buffer. Both input and output are treated as
    rank-3 tensors with logical shape

    .. math::

        [S, D_0, D_1],

    where :math:`S` is a generic leading dimension (it may represent a batch
    size, a sequence length, or a page count, depending on the surrounding
    runtime).

    Key properties
    --------------
    - Dispatch is keyed **only** by the format of ``x`` (``x._format``).
    - The output buffer is allocated during :meth:`profile` using the runtime
      value ``ctx.max_num_pages`` for the leading dimension.
    - The output format is determined by the dispatch table and used when
      wrapping the internal buffer as a :class:`vTensor`.

    Attributes
    ----------
    _impl_map : Dict[FORMAT, FORMAT]
        Dispatch table keyed by ``x_format``. Each entry maps to the
        resolved output format.
    output_format : Optional[FORMAT]
        The output tensor format as determined in :meth:`profile`.
    output_buffer : Optional[torch.Tensor]
        Preallocated output tensor buffer with logical shape
        ``[S_out, D_0, D_1]``, where ``S_out`` comes from the runtime context.
    """

    # Dispatch table keyed by x_format -> resolved output format.
    _impl_map: Dict[FORMAT, FORMAT] = {
        FORMAT.PAGED: FORMAT.RAGGED,
        # Add more entries if you support other formats.
    }

    def __init__(self):
        super().__init__()
        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[torch.Tensor] = None
        self.schedule = Schedule.W

    # ---------------- profile ----------------
    def profile(self, x: vTensor, ctx: Context) -> vTensor:
        r"""
        Validate the input, select an implementation, allocate the output
        buffer, and return an :func:`as_vtensor` view with the resolved format.

        The input tensor is expected to have logical shape ``[S_in, D_0, D_1]``.
        The output tensor is allocated with shape

        .. math::

            [S_{\text{out}}, D_0, D_1],

        where :math:`S_{\text{out}}` is taken from ``ctx.max_num_pages`` to
        match the runtime configuration for the leading dimension.

        Parameters
        ----------
        x : vTensor
            Input tensor to be loaded from, with logical shape
            ``[S_in, D_0, D_1]``.

        ctx : Context
            Execution context providing ``ctx.max_num_pages`` for the leading
            dimension and tracking auxiliary memory usage.

        Returns
        -------
        vTensor
            A ``vTensor`` view wrapping the internally allocated output
            buffer with the resolved output format.

        Raises
        ------
        AssertionError
            If ``x`` is not a :class:`vTensor`, if its rank is not 3, or if
            no implementation is registered for ``x._format``.
        """
        prefix = self._prefix()

        # Type & rank checks
        assert isinstance(x, vTensor), f"{prefix}profile expects x to be vTensor, got {type(x)}"
        assert x.dim() == 3, f"{prefix}expected 3D x [S, D0, D1], got {tuple(x.shape)}"

        # Dispatch by x format
        x_fmt = x._format
        assert x_fmt in self._impl_map, (
            f"{prefix}no implementation for x_fmt={x_fmt}. "
            f"Available keys: {list(self._impl_map.keys())}"
        )
        self.output_format = self._impl_map[x_fmt]

        # Allocate output buffer [S_out, D0, D1] as a vTensor with a fresh
        # tensor_id, mirroring the Softmax pattern so the compiler graph
        # can track the new buffer as a node.
        D0, D1 = x.shape[1], x.shape[2]
        self.output_buffer = as_vtensor(
            torch.empty(
                (0, D0, D1),
                device=x.device,
                dtype=x.dtype,
            ),
            self.output_format,
            tensor_id=len(ctx.tensor_list),
        )

        # Track in the context graph (same convention as Softmax / Conv1d).
        ctx.tensor_list.append(self.output_buffer)
        ctx.output_tensor_to_op_list.append(len(ctx.op_list))
        ctx.op_list.append(self)
        ctx.op_to_input_tensor_list.append([x.tensor_id])
        ctx.op_to_output_tensor_list.append([self.output_buffer.tensor_id])

        return self.output_buffer
