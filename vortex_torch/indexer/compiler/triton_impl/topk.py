from ..graph import Graph
from typing import Dict, Tuple, Callable
from ...context import Context
from ....abs import FORMAT
from ....utils import INDENT


def _check_topk_io(graph: Graph, op_id: int) -> tuple[int, int]:
    input_tensor_id  = graph.op_to_input_tensor_list[op_id][0]
    output_tensor_id = graph.op_to_output_tensor_list[op_id]
    t_i = graph.tensor_list[input_tensor_id]
    t_o = graph.tensor_list[output_tensor_id]
    assert t_i._format == FORMAT.RAGGED, f"Expected ragged input tensor for topk, got {t_i._format}"
    assert t_o._format == FORMAT.RAGGED, f"Expected ragged output tensor for topk, got {t_o._format}"
    return input_tensor_id, output_tensor_id


def generate_topk_impl(graph: Graph, op_id: int, ctx: Context) -> str:
    # Pick the best-known top-k kernel at runtime via the dispatcher in
    # ``vortex_torch/kernels/topk/dispatcher.py``. The dispatch key is
    # ``(max_topk_val, vortex_attention_backend)`` — neither is known at
    # codegen time, so we emit a tiny per-module memo and look up on the
    # hot path.
    #
    # In trtllm mode the 4th / 5th tensor arguments and the last int
    # argument carry different semantics: ``dense_block_tables`` and
    # ``sparse_block_tables`` (2D, [eff_bs, max_blocks_per_seq]) instead
    # of CSR ``dense_kv_indices`` / ``sparse_kv_indices``, and the row
    # stride ``max_blocks_per_seq`` instead of ``max_num_pages``. The
    # selector below picks the right ctx attribute names accordingly.
    ctx.compilation_header_lines.extend([
        "from vortex_torch.kernels.topk.dispatcher import dispatch as _vortex_topk_dispatch",
        "_vortex_topk_kernel_cache = {}",
        "def _vortex_topk_kernel_for(max_topk, attn_backend):",
        "    key = (max_topk, attn_backend)",
        "    fn = _vortex_topk_kernel_cache.get(key)",
        "    if fn is None:",
        "        fn = _vortex_topk_kernel_cache[key] = _vortex_topk_dispatch("
        "max_topk, attention_backend=attn_backend)",
        "    return fn",
    ])
    input_tensor_id, output_tensor_id = _check_topk_io(graph, op_id)

    # Topk already reads from dense_block_tables in trtllm mode, so it does
    # not depend on dense_kv_indices. The other indexer ops (notably the
    # GeMV / score-gather path) still read CSR ``dense_kv_indices`` in both
    # modes; the trtllm planner therefore writes that buffer too.
    # TODO(opt-b): migrate the score-gather codegen to read
    # ``ctx.dense_block_tables`` in trtllm mode and drop the duplicate CSR
    # write from the planner (see vortex_torch/indexer/planner_sglang.py
    # and utils_sglang.get_decode_planner_trtllm for the matching TODOs).
    backend = (getattr(ctx, "vortex_attention_backend", None) or "flashinfer").lower()
    if backend == "trtllm":
        dense_src = "ctx.dense_block_tables"
        last_arg = "ctx.dense_block_tables.shape[1]"  # row stride
    else:
        dense_src = "ctx.dense_kv_indices"
        last_arg = "ctx.max_num_blocks_per_request"

    impl_lines = [
        f"_vortex_topk_kernel_for(getattr(ctx, 'max_topk_val', None), "
        f"getattr(ctx, 'vortex_attention_backend', 'flashinfer'))(",
        f"{INDENT * 2}tensor_{input_tensor_id},",
        f"{INDENT * 2}ctx.dense_kv_indptr,",
        f"{INDENT * 2}ctx.sparse_kv_indptr,",
        f"{INDENT * 2}{dense_src},",
        f"{INDENT * 2}tensor_{output_tensor_id},",
        f"{INDENT * 2}ctx.batch_size * ctx.num_kv_heads,",
        f"{INDENT * 2}ctx.block_reserved_bos,",
        f"{INDENT * 2}ctx.block_reserved_eos,",
        f"{INDENT * 2}{last_arg},",
        f")",
    ]
    return "\n".join(impl_lines)


def generate_approx_topk_impl(graph: Graph, op_id: int, ctx: Context) -> str:
    """Codegen for `approxTopK(tolerate_ratio=...)` — the adaptive 8-bit
    radix variant. Reads `tolerate_ratio` off the op instance at codegen
    time and bakes it into a compile-time substitution for the
    JIT-built ``approx_topk.cu`` kernel under
    ``vortex_torch/kernels/topk/``."""
    input_tensor_id, output_tensor_id = _check_topk_io(graph, op_id)

    op = graph.op_list[op_id]
    # Defensive: profile() runs first and the registry already pinned
    # the op type via (approxTopK, Schedule.S). Still guard the attribute.
    tolerate_ratio = float(getattr(op, "tolerate_ratio", 0.0))

    # tolerate_ratio is a compile-time knob; each op-id pins its own
    # callable. The dispatcher de-dupes by (file, substitutions), so
    # two ops with the same tolerate_ratio share a compiled module.
    callable_name = f"_vortex_approx_topk_{op_id}"
    ctx.compilation_header_lines.extend([
        "from vortex_torch.kernels.topk.dispatcher import get_kernel as _vortex_topk_get_kernel",
        f"{callable_name} = _vortex_topk_get_kernel(",
        f'{INDENT}"approx_topk.cu",',
        f"{INDENT}substitutions={{",
        f'{INDENT * 2}"__THREADS_PER_BLOCK__": 1024,',
        f'{INDENT * 2}"__VORTEX_MAX_TOPK__": 2048,',
        f'{INDENT * 2}"__TOLERATE_RATIO__": {tolerate_ratio!r},',
        f"{INDENT}}},",
        f").topk",
    ])

    backend = (getattr(ctx, "vortex_attention_backend", None) or "flashinfer").lower()
    if backend == "trtllm":
        dense_src = "ctx.dense_block_tables"
        last_arg = "ctx.dense_block_tables.shape[1]"
    else:
        dense_src = "ctx.dense_kv_indices"
        last_arg = "ctx.max_num_blocks_per_request"

    impl_lines = [
        f"{callable_name}(",
        f"{INDENT * 2}tensor_{input_tensor_id},",
        f"{INDENT * 2}ctx.dense_kv_indptr,",
        f"{INDENT * 2}ctx.sparse_kv_indptr,",
        f"{INDENT * 2}{dense_src},",
        f"{INDENT * 2}tensor_{output_tensor_id},",
        f"{INDENT * 2}ctx.batch_size * ctx.num_kv_heads,",
        f"{INDENT * 2}ctx.block_reserved_bos,",
        f"{INDENT * 2}ctx.block_reserved_eos,",
        f"{INDENT * 2}{last_arg},",
        f")",
    ]
    return "\n".join(impl_lines)
