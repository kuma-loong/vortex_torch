from ..graph import Graph
from typing import Dict, List, Tuple, Callable
from ...context import Context
from ....abs import FORMAT
from ....utils import Schedule, INDENT
from .register import get_impl_func
from .dtype_cast import is_fp8 as _is_fp8, load_cast_expr as _load_cast_expr, store_cast_expr as _store_cast_expr
import os
import torch


def _has_batched_output(sub_graph) -> bool:
    """True iff the subgraph stores any BATCHED tensor.

    Used to conditionally include ``winfo_is_first_workload_per_batch`` in
    the kernel signature/launcher and to decide whether the store body
    needs the per-workload "first" gate at all.
    """
    return any(
        sub_graph.tensor_list[tid]._format == FORMAT.BATCHED
        for tid in sub_graph.output_tensor_ids
    )


def indent_block(text: str, level: int = 1) -> str:
    """
    Indent a multi-line text block by the given indentation level.
    """
    prefix = INDENT * level
    lines = text.splitlines()
    return "\n".join(prefix + line if line.strip() else line for line in lines)


def generate_initialization_str(sub_graph: Graph, ctx:Context) -> str:
    """
    Generate the initialization code string for a given sub-graph.
    This may include declarations of intermediate variables and pre-computations.
    """
    lines = []

    for local_tensor_id in sub_graph.input_tensor_ids:
        t = sub_graph.tensor_list[local_tensor_id]
        if t._format == FORMAT.BATCHED:
            lines.append(f"# Declare variables for tensor_{local_tensor_id}")
            lines.append(
                f"tensor_{local_tensor_id}_block = tl.zeros((1, {t.shape[1]}, {t.shape[2]}), dtype=tl.float32)"
            )

    
    for local_tensor_id in sub_graph.output_tensor_ids:
        t = sub_graph.tensor_list[local_tensor_id]
        if t._format in (FORMAT.RAGGED, FORMAT.BATCHED):
            lines.append(f"tensor_{local_tensor_id}_dim1_ptr = tl.arange(0, {t.shape[1]})")
            lines.append(f"tensor_{local_tensor_id}_dim2_ptr = tl.arange(0, {t.shape[2]})")

    if ctx.num_pages_per_workload > 1:
        lines.append(f"page_idx_i32_ptr = tl.arange(0, {ctx.num_pages_per_workload})")
        lines.append(f"block_i32_ptr = tl.arange(0, {ctx.num_blocks_per_page})")
        for local_tensor_id in sub_graph.input_tensor_ids:
            t = sub_graph.tensor_list[local_tensor_id]
            if t._format == FORMAT.PAGED:     
                lines.append(f"tensor_{local_tensor_id}_flat_ptr = tl.arange(0, {ctx.num_blocks_per_page * t.shape[1] * t.shape[2]})")
        for local_tensor_id in sub_graph.output_tensor_ids:
            t = sub_graph.tensor_list[local_tensor_id]
            if t._format == FORMAT.PAGED:     
                lines.append(f"tensor_{local_tensor_id}_dim1_ptr = tl.arange(0, {t.shape[1]})")
                lines.append(f"tensor_{local_tensor_id}_dim2_ptr = tl.arange(0, {t.shape[2]})")
                
    return "\n".join(lines) if lines else "# No initialization required"


def generate_load_tensor_str(sub_graph: Graph, ctx: Context) -> str:
    """
    Generate the tensor loading code string for a given sub-graph.
    """
    batched_lines = []
    paged_lines = []
    ragged_lines = []
    for local_tensor_id in sub_graph.input_tensor_ids:
        t = sub_graph.tensor_list[local_tensor_id]
        if t._format == FORMAT.BATCHED:
            batched_lines.append(
                f"tensor_{local_tensor_id}_ptr_row_start = new_batch_idx_i32 * {t.shape[1]}"
            )
            batched_lines.append(
                "\n".join([
                    f"tensor_{local_tensor_id}_block_ptr = tl.make_block_ptr(",
                    f"{INDENT}base=tensor_{local_tensor_id}_ptr,",
                    f"{INDENT}shape=(tensor_{local_tensor_id}_dim0 * {t.shape[1]}, {t.shape[2]}),",
                    f"{INDENT}strides=({t.shape[2]}, 1),",
                    f"{INDENT}offsets=(tensor_{local_tensor_id}_ptr_row_start, 0),",
                    f"{INDENT}block_shape=({t.shape[1]}, {t.shape[2]}),",
                    f"{INDENT}order=(1, 0),",
                    f")",
                ])
            )
            batched_reshape = f"tl.reshape(tl.load(tensor_{local_tensor_id}_block_ptr, boundary_check=(0, 1), padding_option=\"zero\", cache_modifier=\".ca\"), (1, {t.shape[1]}, {t.shape[2]}))"
            batched_lines.append(f"tensor_{local_tensor_id}_block = {_load_cast_expr(batched_reshape, t)}")
        elif t._format == FORMAT.PAGED:

            if ctx.num_pages_per_workload == 1:
                paged_lines.append(
                    f"tensor_{local_tensor_id}_ptr_row_start = page_idx_i32 * {t.shape[1]}"
                )
                paged_lines.append(
                    "\n".join([
                        f"tensor_{local_tensor_id}_block_ptr = tl.make_block_ptr(",
                        f"{INDENT}base=tensor_{local_tensor_id}_ptr,",
                        f"{INDENT}shape=(tensor_{local_tensor_id}_dim0 * {t.shape[1]}, {t.shape[2]}),",
                        f"{INDENT}strides=({t.shape[2]}, 1),",
                        f"{INDENT}offsets=(tensor_{local_tensor_id}_ptr_row_start, 0),",
                        f"{INDENT}block_shape=({t.shape[1] * ctx.workload_chunk_size}, {t.shape[2]}),",
                        f"{INDENT}order=(1, 0),",
                        f")",
                    ])
                )
                paged_reshape = f"tl.reshape(tl.load(tensor_{local_tensor_id}_block_ptr, boundary_check=(0, 1), padding_option=\"zero\", cache_modifier=\".cv\"), ({ctx.workload_chunk_size}, {t.shape[1]}, {t.shape[2]}))"
                paged_lines.append(f"tensor_{local_tensor_id}_block = {_load_cast_expr(paged_reshape, t)}")
            else:
                paged_lines.append(f"_tensor_{local_tensor_id}_block_ptr = page_indices_i32[:,None] * {t.shape[1] * t.shape[2]} + tensor_{local_tensor_id}_flat_ptr[None,:]")
                paged_lines.append(f"tensor_{local_tensor_id}_block_ptr = tensor_{local_tensor_id}_ptr + _tensor_{local_tensor_id}_block_ptr")
                paged_reshape = f"tl.reshape(tl.load(tensor_{local_tensor_id}_block_ptr, mask=page_valid[:, None], other=0.0, cache_modifier=\".cv\"), ({ctx.workload_chunk_size}, {t.shape[1]}, {t.shape[2]}))"
                paged_lines.append(f"tensor_{local_tensor_id}_block = {_load_cast_expr(paged_reshape, t)}")
            
        elif t._format == FORMAT.RAGGED:
            ragged_lines.append(
                f"tensor_{local_tensor_id}_ptr_row_start = ragged_idx_i32 * {t.shape[1]}"
            )
            ragged_lines.append(
                "\n".join([
                    f"tensor_{local_tensor_id}_block_ptr = tl.make_block_ptr(",
                    f"{INDENT}base=tensor_{local_tensor_id}_ptr,",
                    f"{INDENT}shape=(tensor_{local_tensor_id}_dim0 * {t.shape[1]}, {t.shape[2]}),",
                    f"{INDENT}strides=({t.shape[2]}, 1),",
                    f"{INDENT}offsets=(tensor_{local_tensor_id}_ptr_row_start, 0),",
                    f"{INDENT}block_shape=({t.shape[1] * ctx.workload_chunk_size}, {t.shape[2]}),",
                    f"{INDENT}order=(1, 0),",
                    f")",
                ])
            )
            ragged_reshape = f"tl.reshape(tl.load(tensor_{local_tensor_id}_block_ptr, boundary_check=(0, 1), padding_option=\"zero\",), ({ctx.workload_chunk_size}, {t.shape[1]}, {t.shape[2]}))"
            ragged_lines.append(f"tensor_{local_tensor_id}_block = {_load_cast_expr(ragged_reshape, t)}")

    if batched_lines:
        batched_lines = [
            "new_batch_idx_i32 = tl.load(winfo_x_indices + i).to(tl.int32)",
            indent_block("\n".join(batched_lines), 0),
        ]
        batched_lines = "\n".join(batched_lines)

    paged_lines = "\n".join(paged_lines) if paged_lines else "# No paged tensor loading required"
    ragged_lines = "\n".join(ragged_lines) if ragged_lines else "# No ragged tensor loading required"
    addressing_lines = [
        "ragged_idx_i32 = tl.load(winfo_y_offsets + i).to(tl.int32)", 
        "page_idx_i32 = tl.load(indices + ragged_idx_i32).to(tl.int32)" if ctx.num_pages_per_workload == 1 else f"page_indices_i32 = tl.load(indices + ragged_idx_i32 + page_idx_i32_ptr * {ctx.num_blocks_per_page}, mask=page_valid, other=0).to(tl.int32)",
    ]
    addressing_lines = "\n".join(addressing_lines)

    return "\n\n".join(filter(None, [batched_lines, addressing_lines, paged_lines, ragged_lines])) if (batched_lines or addressing_lines or paged_lines or ragged_lines) else "# No tensor loading required"


def generate_store_tensor_str(sub_graph: Graph, ctx: Context) -> str:
    """
    Generate the tensor storing code string for a given sub-graph.

    BATCHED outputs are written once per workload at the
    ``new_batch_idx_i32`` slot. Multiple workloads sharing the same
    ``(batch, head)`` write the same value (BATCHED outputs come from
    page-independent compute, e.g. ``GeMM(BATCHED, BATCHED)``), so
    redundant stores are safe.
    """
    paged_lines = []
    ragged_lines = []
    batched_lines = []
    for local_tensor_id in sub_graph.output_tensor_ids:
        t = sub_graph.tensor_list[local_tensor_id]
        if t._format == FORMAT.PAGED:
            if ctx.num_pages_per_workload == 1:
                paged_lines.append(
                    f"tensor_{local_tensor_id}_ptr_row_start = page_idx_i32 * {t.shape[1]}"
                )
                paged_lines.append(
                    "\n".join([
                        f"tensor_{local_tensor_id}_block_ptr = tl.make_block_ptr(",
                        f"{INDENT}base=tensor_{local_tensor_id}_ptr,",
                        f"{INDENT}shape=(tensor_{local_tensor_id}_dim0 * {t.shape[1]}, {t.shape[2]}),",
                        f"{INDENT}strides=({t.shape[2]}, 1),",
                        f"{INDENT}offsets=(tensor_{local_tensor_id}_ptr_row_start, 0),",
                        f"{INDENT}block_shape=({t.shape[1] * ctx.workload_chunk_size}, {t.shape[2]}),",
                        f"{INDENT}order=(1, 0),",
                        f")",
                    ])
                )
                paged_reshape = f"tl.reshape(tensor_{local_tensor_id}_block, ({t.shape[1] * ctx.workload_chunk_size}, {t.shape[2]}))"
                paged_lines.append(f"tl.store(tensor_{local_tensor_id}_block_ptr, {_store_cast_expr(paged_reshape, t.dtype)})")
            else:
                paged_lines.append(f"tensor_{local_tensor_id}_block_ptr = tensor_{local_tensor_id}_ptr + page_indices_i32[:,None,None,None] * {t.shape[1] * t.shape[2]} + block_i32_ptr[None,:,None,None] * {t.shape[1] * t.shape[2]} + tensor_{local_tensor_id}_dim1_ptr[None,None,:,None] * {t.shape[2]} + tensor_{local_tensor_id}_dim2_ptr[None,None,None,:]")
                paged_reshape = f"tl.reshape(tensor_{local_tensor_id}_block, ({ctx.num_pages_per_workload}, {ctx.num_blocks_per_page}, {t.shape[1]}, {t.shape[2]}))"
                paged_lines.append(f"tl.store(tensor_{local_tensor_id}_block_ptr, {_store_cast_expr(paged_reshape, t.dtype)}, mask=page_valid[:, None, None, None])")

        elif t._format == FORMAT.RAGGED:
            ragged_lines.append(
                "\n".join([
                    f"tensor_{local_tensor_id}_block_ptr = tensor_{local_tensor_id}_ptr + ragged_idx_i32 * {t.shape[1] * t.shape[2]} + workload_ptr[:,None,None] * {t.shape[1] * t.shape[2]} + tensor_{local_tensor_id}_dim1_ptr[None,:,None] * {t.shape[2]} + tensor_{local_tensor_id}_dim2_ptr[None,None,:]",
                ])
            )

            ragged_lines.append(f"tl.store(tensor_{local_tensor_id}_block_ptr, {_store_cast_expr(f'tensor_{local_tensor_id}_block', t.dtype)}, mask=valid[:, None, None])")

        elif t._format == FORMAT.BATCHED:
            # One slot per (batch, head). Multiple workloads can share the
            # same (batch, head) when its KV span exceeds workload_chunk_size,
            # but only the *first* such workload should write the slot —
            # ``winfo_is_first_workload_per_batch[i]`` is the gate computed
            # by the planner (see ``planner_sglang.py``). The load itself
            # is hoisted out of the per-tensor loop below so it happens
            # exactly once per workload regardless of how many BATCHED
            # outputs the subgraph has.
            batched_lines.append(
                f"tensor_{local_tensor_id}_block_ptr = tensor_{local_tensor_id}_ptr "
                f"+ new_batch_idx_i32 * {t.shape[1] * t.shape[2]} "
                f"+ tensor_{local_tensor_id}_dim1_ptr[None, :, None] * {t.shape[2]} "
                f"+ tensor_{local_tensor_id}_dim2_ptr[None, None, :]"
            )
            batched_lines.append(
                f"if _is_first_workload != 0:"
            )
            batched_lines.append(
                f"    tl.store(tensor_{local_tensor_id}_block_ptr, "
                f"{_store_cast_expr(f'tensor_{local_tensor_id}_block', t.dtype)})"
            )


    if paged_lines:
        paged_lines = "\n".join(paged_lines)

    if ragged_lines:
        ragged_lines = [
            f"_len = tl.load(winfo_y_lens + i)",
            f"valid = workload_ptr < _len",
            "\n".join(ragged_lines)
        ] if ctx.num_pages_per_workload == 1 else ragged_lines
        ragged_lines = "\n".join(ragged_lines)

    if batched_lines:
        # Hoist the gate load — exactly one ``tl.load`` per workload, then
        # every BATCHED store guards on the same scalar.
        batched_lines = [
            "_is_first_workload = tl.load(winfo_is_first_workload_per_batch + i)",
            "\n".join(batched_lines),
        ]
        batched_lines = "\n".join(batched_lines)
    return "\n\n".join(filter(None, [paged_lines, ragged_lines, batched_lines])) if (paged_lines or ragged_lines or batched_lines) else "# No tensor loading required"


def generate_computation_str(sub_graph: Graph, ctx: Context) -> str:
    
    lines = []
    for op_id, op in enumerate(sub_graph.op_list):
        op_impl_func = get_impl_func(op)
        impl_str = op_impl_func(sub_graph, op_id, ctx)
        lines.append(impl_str)
    return "\n\n".join(lines) if lines else "# No computation required"

def generate_triton_kernel(
    sub_graph: Graph,
    sub_graph_id: int,
    ctx: Context
) -> str:
    """
    Generate a Triton kernel definition for a given sub-graph.
    """
    kernel_arg_list = [
        "indices,                            # int32",
        "winfo_x_indices,                    # int32",
    ]
    if _has_batched_output(sub_graph):
        kernel_arg_list.append("winfo_is_first_workload_per_batch,  # uint8")
    kernel_arg_list += [
        "winfo_y_offsets,                    # int32",
        "winfo_y_lens,                       # int32",
        "winfo_num_workloads,                # int32",
    ]

    # Collect input tensor kernel arguments. A tensor that's *both* an
    # input (e.g. ``Load`` reads it) and an output (e.g. ``Save`` writes
    # it) in the same subgraph must only appear once in the kernel
    # signature; we bind it in the output-arg loop below.
    output_set = set(sub_graph.output_tensor_ids)
    for local_tensor_id in sub_graph.input_tensor_ids:
        if local_tensor_id in output_set:
            continue
        tensor_name = f"tensor_{local_tensor_id}"
        kernel_arg_list.extend([
            f"{tensor_name}_ptr,",
            f"{tensor_name}_dim0: tl.constexpr,",
        ])

    # Collect output tensor kernel arguments
    for local_tensor_id in sub_graph.output_tensor_ids:
        tensor_name = f"tensor_{local_tensor_id}"
        kernel_arg_list.extend([
            f"{tensor_name}_ptr,",
            f"{tensor_name}_dim0: tl.constexpr,",
        ])

    kernel_args = "\n".join(f"{INDENT}{arg}" for arg in kernel_arg_list)
    prepare_workload_lines = []
    if ctx.num_pages_per_workload > 1:
        prepare_workload_lines = [
        f"_len = tl.load(winfo_y_lens + i)",
        f"valid = workload_ptr < _len",
        f"_page = (_len + {ctx.num_blocks_per_page - 1}) // {ctx.num_blocks_per_page}",
        f"page_valid = page_idx_i32_ptr < _page",
     ]
    prepare_workload_str = "\n".join(prepare_workload_lines) if prepare_workload_lines else "# No workload preparation required"

    initialization_str = indent_block(generate_initialization_str(sub_graph,ctx), 1)
    prepare_workload_str = indent_block(prepare_workload_str, 2)
    load_tensor_str = indent_block(generate_load_tensor_str(sub_graph, ctx), 2)
    store_tensor_str = indent_block(generate_store_tensor_str(sub_graph, ctx), 2)
    computation_str = indent_block(generate_computation_str(sub_graph, ctx), 2)
    kernel_str = f"""
@triton.jit
def {ctx.sparse_attention_name}_subgraph_{sub_graph_id}_kernel(
{kernel_args}
):
    # ------------------------------------------------------------
    # Program-level partitioning of workloads
    # ------------------------------------------------------------
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    n_workloads = tl.load(winfo_num_workloads)

    per = n_workloads // num_progs
    r = n_workloads % num_progs
    start = pid * per + tl.minimum(pid, r)
    end = start + per + (pid < r)
    workload_ptr = tl.arange(0, {ctx.workload_chunk_size})

{initialization_str}

    for i in range(start, end):
{prepare_workload_str}
{load_tensor_str}
{computation_str}
{store_tensor_str}
"""
    return kernel_str.strip()


def generate_triton_impl(
    sub_graph: Graph,
    sub_graph_id: int,
    ctx: Context
) -> str:
    """
    Generate a Triton kernel and its Python wrapper implementation.
    """
    ctx.compilation_header_lines.extend([
        "import torch",
        "import triton",
        "import triton.language as tl",
    ])
    if sub_graph.schedule == Schedule.W:
        kernel_str = generate_triton_kernel(sub_graph, sub_graph_id, ctx)

        arg_list = []
        kernel_input_list = [
            "ctx.dense_kv_indices",
            "ctx.winfo_q_indices",
        ]
        if _has_batched_output(sub_graph):
            kernel_input_list.append("ctx.winfo_is_first_workload_per_batch")
        kernel_input_list += [
            "ctx.winfo_kv_offsets",
            "ctx.winfo_kv_lens",
            "ctx.winfo_num_workloads",
        ]

        # FP8-aware wrapper: any FP8 tensor (input or output) is reinterpreted
        # as ``uint8`` before being handed to the kernel, which bitcasts it
        # back to the matching ``tl.float8eX`` dtype on load and stores raw
        # fp8 bits on write. Mirrors the cache-side convention in
        # ``cache.compiler.triton_impl.kernel_gen``.
        fp8_rebind_lines: List[str] = []

        # A tensor that's both an input and an output (e.g. a cache field
        # both Load'd and Save'd in the same subgraph) must be passed
        # exactly once — we bind it in the output-arg loop below.
        output_set = set(sub_graph.output_tensor_ids)

        # Collect input tensor arguments
        for local_tensor_id in sub_graph.input_tensor_ids:
            if local_tensor_id in output_set:
                continue
            tensor_name = f"tensor_{local_tensor_id}"
            arg_list.append(tensor_name)
            kernel_input_list.extend([
                tensor_name,
                f"{tensor_name}.shape[0]",
            ])
            if _is_fp8(sub_graph.tensor_list[local_tensor_id]):
                fp8_rebind_lines.append(
                    f"{tensor_name} = {tensor_name}.view(torch.uint8)"
                )

        # Collect output tensor arguments
        for local_tensor_id in sub_graph.output_tensor_ids:
            tensor_name = f"tensor_{local_tensor_id}"
            arg_list.append(tensor_name)
            kernel_input_list.extend([
                tensor_name,
                f"{tensor_name}.shape[0]",
            ])
            if _is_fp8(sub_graph.tensor_list[local_tensor_id]):
                fp8_rebind_lines.append(
                    f"{tensor_name} = {tensor_name}.view(torch.uint8)"
                )

        # Append context argument
        arg_list.append("ctx")
        kernel_input_list.append("num_warps=8")
        kernel_input_list.append("num_stages=2")
        args_def = ",\n".join(f"{INDENT}{arg}" for arg in arg_list)
        kernel_inputs = ",\n".join(f"{INDENT * 2}{arg}" for arg in kernel_input_list)
        fp8_rebind_str = (
            indent_block("\n".join(fp8_rebind_lines), 1) if fp8_rebind_lines else ""
        )

        impl_str = f"""
{kernel_str}

def {ctx.sparse_attention_name}_subgraph_{sub_graph_id}_impl(
{args_def}
):
{fp8_rebind_str}
    {ctx.sparse_attention_name}_subgraph_{sub_graph_id}_kernel[({ctx.num_sms * 4},)](
{kernel_inputs}
    )
"""
        return impl_str.strip()
    
    else:
        arg_list = []
        for local_tensor_id in sub_graph.input_tensor_ids:
            arg_list.append(f"tensor_{local_tensor_id}")
        for local_tensor_id in sub_graph.output_tensor_ids:
            arg_list.append(f"tensor_{local_tensor_id}")
        arg_list.append("ctx")
        args_def = ",\n".join(f"{INDENT}{arg}" for arg in arg_list)
        assert len(sub_graph.op_list) == 1, "Expected exactly one operation in non-workload-scheduled sub-graph for direct implementation."
        op_impl_func = get_impl_func(sub_graph.op_list[0])
        op_impl_str = indent_block(op_impl_func(sub_graph, 0, ctx), 1)
        impl_str = f"""
def {ctx.sparse_attention_name}_subgraph_{sub_graph_id}_impl(
{args_def}
):
{op_impl_str}
""" 
        return impl_str.strip()