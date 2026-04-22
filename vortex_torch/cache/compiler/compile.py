"""Top-level cache compiler entry point.

Mirrors :mod:`vortex_torch.indexer.compiler.compile`: build the graph
from ``ctx``, generate the source file, import it dynamically, and
return the compiled class.
"""

from .graph import contruct_graph, Graph
from .interface import generate_interface
from typing import List, Optional
from ..context import Context
import os
import importlib.util
import sys


def compile(ctx: Context, final_output_tensor_ids: Optional[List[int]] = None):
    """Compile the cache pipeline registered on ``ctx`` and return the class."""

    full_graph, sub_graphs = contruct_graph(ctx, final_output_tensor_ids=final_output_tensor_ids)
    file_path, cls_name = generate_interface(full_graph, sub_graphs, ctx=ctx)

    module_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return getattr(module, cls_name)
