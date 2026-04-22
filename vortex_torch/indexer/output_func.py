import torch
from typing import FrozenSet
from ..abs import vOp, vTensor, FORMAT
from .context import Context
from ..utils import Schedule

class topK(vOp):
    r"""
    Piecewise top-k dispatcher for packed sequences with reserved pages.

    The input is treated as a rank-3 tensor

    .. math::

        X \in \mathbb{R}^{S_{\text{pack}} \times 1 \times 1},

    where the leading dimension :math:`S_{\text{pack}}` is a packed
    concatenation of :math:`B` segments:

    .. math::

        S_{\text{pack}} = \sum_{b=0}^{B-1} S_b, \qquad
        X =
        \begin{bmatrix}
            X_0 \\
            X_1 \\
            \vdots \\
            X_{B-1}
        \end{bmatrix},

    with

    .. math::

        X_b \in \mathbb{R}^{S_b \times 1 \times 1}.

    For each segment :math:`b`, the operator selects a subset of pages
    according to scores stored in ``x`` (and additional key/value
    metadata from :class:`Context`), but **always** preserves:

    - the first ``page_reserved_bos`` pages in that segment,
    - the last ``page_reserved_eos`` pages in that segment, and
    - an additional ``topk_val`` pages chosen by top-k over the remaining
      candidates in the segment.

    Let :math:`\mathcal{I}_b \subset [0, S_{\text{pack}})` denote the index
    range of segment :math:`b`. The implementation computes a subset
    :math:`\mathcal{J}_b \subset \mathcal{I}_b` such that:

    - all indices corresponding to the reserved prefix (BOS) and suffix
      (EOS) pages in :math:`\mathcal{I}_b` are included, and
    - up to ``topk_val`` additional indices are selected by score.

    The result is written into a preallocated output tensor ``o``; the
    exact layout of ``o`` is defined by the upstream contract and the
    implementation.

    Key properties
    --------------
    - Dispatch is keyed **only** by the input format ``x._format``.
    - The operation is logically out-of-place, but writes into ``o`` in-place.
    - :meth:`profile` only validates and selects the implementation; it does
      not allocate or return any buffers.
    - :meth:`execute` performs the per-segment selection using context
      metadata (indptr arrays, indices, reserved-page counts, and ``topk_val``).

    Attributes
    ----------
    _supported_formats : FrozenSet[FORMAT]
        Set of input formats for which a top-k codegen implementation
        exists. Used by :meth:`profile` for dispatch validation.
    """

    # Supported input formats; only RAGGED is supported for now.
    _supported_formats: FrozenSet[FORMAT] = frozenset({FORMAT.RAGGED})

    def __init__(self):
        super().__init__()
        self.schedule = Schedule.S

    # ---------------- profile ----------------
    def profile(self, x: vTensor, o: vTensor, ctx: Context) -> None:
        r"""
        Validate input/output tensors and select the implementation.

        This method checks:

        - that ``x`` and ``o`` are both rank-3 :class:`vTensor` objects,
        - that ``x`` has shape ``[S_pack, 1, 1]`` (one scalar score per page),
        - that a top-k implementation is registered for ``x._format``, and
        - that ``x`` and ``o`` reside on the same device.

        No buffers are allocated here and nothing is returned; this call
        simply validates that ``x._format`` has a registered codegen path.

        Parameters
        ----------
        x : vTensor
            Input tensor carrying per-page scalar scores, with logical shape
            ``[S_pack, 1, 1]``.

        o : vTensor
            Preallocated output tensor that will be filled in-place by the
            top-k implementation. Its shape and semantics are defined by
            the upstream contract and the implementation.

        ctx : Context
            Execution context providing:

            - ``dense_kv_indptr`` and ``sparse_kv_indptr``: segment
              boundaries in the packed axis,
            - ``dense_kv_indices``: indices into underlying storage, and
            - scalar configuration such as ``batch_size``, ``num_kv_heads``,
              ``topk_val``, ``page_reserved_bos``, and ``page_reserved_eos``.

        Raises
        ------
        AssertionError
            If types, ranks, shapes, formats, or devices are incompatible,
            or if no implementation is registered for ``x._format``.
        """
        prefix = self._prefix()

        # ---- type checks ----
        assert isinstance(x, vTensor), f"{prefix}profile expects x to be vTensor, got {type(x)}"
        assert isinstance(o, vTensor), f"{prefix}profile expects o to be vTensor, got {type(o)}"

        # ---- rank checks ----
        assert x.dim() == 3, (
            f"{prefix}expected x to be 3D, "
            f"got ndim={x.dim()} shape={tuple(x.shape)}"
        )
        assert o.dim() == 3, (
            f"{prefix}expected o to be 3D, "
            f"got ndim={o.dim()} shape={tuple(o.shape)}"
        )

        # ---- shape checks for x ----
        # x is expected to carry per-page scalars at dims (1,2)
        assert x.shape[1] == 1 and x.shape[2] == 1, (
            f"{prefix}expected x.shape[1] == x.shape[2] == 1, got {tuple(x.shape)}"
        )

        # ---- implementation availability ----
        x_fmt = x._format
        assert x_fmt in self._supported_formats, (
            f"{prefix}no implementation for x._format={x_fmt}. "
            f"Supported: {sorted(self._supported_formats, key=lambda f: f.value)}"
        )

        # ---- optional sanity checks on `o` ----
        # We only assert device consistency and leave exact (S_pack, D0, D1)
        # to the upstream contract and the implementation.
        assert x.device == o.device, (
            f"{prefix}x and o must be on the same device "
            f"(x.device={x.device}, o.device={o.device})"
        )

        # Track auxiliary memory and graph structure in the context
        ctx.output_tensor_to_op_list[o.tensor_id] = len(ctx.op_list)   # Map the output tensor to this operation
        ctx.op_list.append(self)  # Track this operation in the context
        ctx.op_to_input_tensor_list.append([x.tensor_id])  # Map this op to its input tensors
        ctx.op_to_output_tensor_list.append([o.tensor_id])  # Map this op to its output tensor
