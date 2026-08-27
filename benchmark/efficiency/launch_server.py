"""Launch a guarded GLM-4.7-Flash Quest server for efficiency probes."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.efficiency.gpu_guard import query_gpu_states
from benchmark.efficiency.runtime_env import prepend_interpreter_bin_to_path


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
    parser.add_argument("--max-topk", type=int, default=256)
    parser.add_argument("--layers-skip", default="")
    parser.add_argument("--block-reserved-bos", type=int, default=1)
    parser.add_argument("--block-reserved-eos", type=int, default=2)
    parser.add_argument("--context-length", type=int, default=33792)
    parser.add_argument("--chunked-prefill-size", type=int)
    parser.add_argument("--max-prefill-tokens", type=int)
    parser.add_argument("--mem-fraction-static", type=float, default=0.90)
    parser.add_argument("--max-running-requests", type=int, default=64)
    parser.add_argument("--cuda-graph-max-bs", type=int)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--disable-cuda-graph", action="store_true")
    return parser.parse_args()


def _validate(args: argparse.Namespace) -> None:
    if args.physical_gpu not in {4, 5, 6, 7}:
        raise ValueError("SM90 probes may only use physical GPUs 4-7.")
    state = query_gpu_states([args.physical_gpu])[0]
    if not state.idle:
        raise RuntimeError(
            f"GPU {args.physical_gpu} is not idle: "
            f"util={state.utilization_pct:.0f}%, memory={state.memory_used_mib:.0f}MiB, "
            f"processes={state.compute_processes}."
        )
    if args.block_size not in {16, 32, 64}:
        raise ValueError("--block-size must be 16, 32, or 64.")
    args.layers_skip = [
        int(layer.strip()) for layer in args.layers_skip.split(",") if layer.strip()
    ]
    if len(args.layers_skip) != len(set(args.layers_skip)) or any(
        layer < 0 for layer in args.layers_skip
    ):
        raise ValueError("--layers-skip must contain unique non-negative layer IDs.")
    positive = [
        args.topk,
        args.max_topk,
        args.context_length,
        args.max_running_requests,
    ]
    positive.extend(
        value
        for value in (
            args.chunked_prefill_size,
            args.max_prefill_tokens,
            args.cuda_graph_max_bs,
        )
        if value is not None
    )
    if any(value <= 0 for value in positive):
        raise ValueError("Top-k, context, prefill, request, and graph limits must be positive.")
    if args.max_topk < args.topk:
        raise ValueError("--max-topk must be greater than or equal to --topk.")
    if args.block_reserved_bos < 0 or args.block_reserved_eos < 1:
        raise ValueError("Reserved BOS must be non-negative and reserved EOS at least one.")
    if (
        args.chunked_prefill_size is not None
        and args.chunked_prefill_size % args.block_size
    ):
        raise ValueError("--chunked-prefill-size must be divisible by --block-size.")
    if not 0.0 < args.mem_fraction_static < 1.0:
        raise ValueError("--mem-fraction-static must be in (0, 1).")
    if not (Path(args.model_path) / "config.json").is_file():
        raise FileNotFoundError(f"Model config not found under {args.model_path}.")


def main() -> None:
    args = parse_args()
    _validate(args)

    prepend_interpreter_bin_to_path()

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
        "impl_backend": "triton",
        "use_tensor_core": True,
        "attention_backend": "trtllm",
        "layers_skip": args.layers_skip,
        "block_reserved_eos": args.block_reserved_eos,
        "block_reserved_bos": args.block_reserved_bos,
        "topk_val": args.topk,
        "topk_ratio": 0.0,
        "block_size": args.block_size,
        "workload_chunk_size": 64,
        "module_name": args.module_name,
        "max_seq_lens": args.context_length,
        "max_topk_val": args.max_topk,
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
    optional_limits = (
        ("--chunked-prefill-size", args.chunked_prefill_size),
        ("--max-prefill-tokens", args.max_prefill_tokens),
        ("--cuda-graph-max-bs", args.cuda_graph_max_bs),
    )
    for flag, value in optional_limits:
        if value is not None:
            server_cli.extend((flag, str(value)))
    if args.disable_cuda_graph:
        server_cli.append("--disable-cuda-graph")

    server_args = prepare_server_args(server_cli)
    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
