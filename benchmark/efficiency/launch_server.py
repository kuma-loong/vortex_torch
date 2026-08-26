"""Launch the matched GLM-4.7-Flash Quest server for SM90 experiments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from benchmark.efficiency.gpu_guard import query_gpu_states


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument(
        "--attention-backend",
        choices=["triton", "cuda_mla_sm90"],
        required=True,
    )
    parser.add_argument(
        "--model-path", default="/data2/pretrain_models/GLM-4.7-Flash"
    )
    parser.add_argument("--module-name", default="quest_mla")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--topk", type=int, default=61)
    parser.add_argument("--context-length", type=int, default=33792)
    parser.add_argument("--mem-fraction-static", type=float, default=0.90)
    parser.add_argument("--max-running-requests", type=int, default=64)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument(
        "--vortex-impl-backend",
        choices=["triton", "cuda"],
        default="triton",
        help="Schedule.W indexer implementation used by Vortex.",
    )
    parser.add_argument(
        "--no-vortex-tensor-core",
        action="store_true",
        help="Disable Triton tensor-core indexer codegen (required for CUDA).",
    )
    return parser.parse_args()


def _validate(args: argparse.Namespace) -> None:
    if args.physical_gpu not in {4, 5, 6, 7}:
        raise ValueError("SM90 experiments may only use physical GPUs 4-7.")
    state = query_gpu_states([args.physical_gpu])[0]
    if not state.idle:
        raise RuntimeError(
            f"GPU {args.physical_gpu} is not idle: "
            f"util={state.utilization_pct:.0f}%, memory={state.memory_used_mib:.0f}MiB, "
            f"processes={state.compute_processes}."
        )
    if args.block_size not in {16, 32, 64}:
        raise ValueError("--block-size must be 16, 32, or 64.")
    if args.topk <= 0 or args.context_length <= 0 or args.max_running_requests <= 0:
        raise ValueError("topk, context length, and max running requests must be positive.")
    if not 0.0 < args.mem_fraction_static < 1.0:
        raise ValueError("--mem-fraction-static must be in (0, 1).")
    if not (Path(args.model_path) / "config.json").is_file():
        raise FileNotFoundError(f"Model config not found under {args.model_path}.")
    if args.vortex_impl_backend == "cuda" and not args.no_vortex_tensor_core:
        raise ValueError(
            "--vortex-impl-backend cuda requires --no-vortex-tensor-core."
        )


def main() -> None:
    args = parse_args()
    _validate(args)

    # Calling ``.venv/bin/python`` does not activate the virtual environment.
    # Torch JIT extensions discover the ``ninja`` executable through PATH, so
    # explicitly inherit the bin directory of the interpreter that owns this
    # process.  This keeps all compiler helpers inside the requested root .venv
    # and makes detached/resumed launches behave like interactive ones.
    # Do not resolve the Python symlink: uv venvs point it at the shared base
    # interpreter, while companion executables (notably ninja) live beside the
    # symlink in the project-local .venv/bin directory.
    interpreter_bin = str(Path(sys.executable).parent)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    os.environ["PATH"] = os.pathsep.join(
        [interpreter_bin, *[entry for entry in path_entries if entry != interpreter_bin]]
    )

    # Pin immediately after the physical-device idle check. Child scheduler
    # processes inherit this single-device view.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu)
    os.environ.setdefault("SGLANG_ENABLE_TORCH_COMPILE", "0")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault(
        "VORTEX_CUDA_MLA_SM90_BUILD_DIR",
        str(REPO_ROOT / ".sm90_mla_work" / "build" / "cuda_mla_sm90"),
    )

    import vortex_torch  # noqa: F401 - installs ServerArgs and backend hooks first
    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    vortex_config = {
        "impl_backend": args.vortex_impl_backend,
        "use_tensor_core": not args.no_vortex_tensor_core,
        "attention_backend": "trtllm",
        "layers_skip": [],
        "block_reserved_eos": 2,
        "block_reserved_bos": 1,
        "topk_val": args.topk,
        "topk_ratio": 0.0,
        "block_size": args.block_size,
        "workload_chunk_size": 64,
        "module_name": args.module_name,
        "max_seq_lens": args.context_length,
        "max_topk_val": 256,
        "dtype": "bfloat16",
        "compilation_cache_dir": str(
            REPO_ROOT / ".sm90_mla_work" / "build" / "vortex_compilation_cache"
        ),
    }
    server_cli = [
        "--model-path",
        args.model_path,
        "--page-size",
        str(args.block_size),
        "--attention-backend",
        args.attention_backend,
        "--vortex-config",
        json.dumps(vortex_config),
        "--context-length",
        str(args.context_length),
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--max-running-requests",
        str(args.max_running_requests),
        "--disable-radix-cache",
        "--trust-remote-code",
        "--tp-size",
        "1",
        "--port",
        str(args.port),
        "--host",
        args.host,
    ]
    if args.disable_cuda_graph:
        server_cli.append("--disable-cuda-graph")

    server_args = prepare_server_args(server_cli)
    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
