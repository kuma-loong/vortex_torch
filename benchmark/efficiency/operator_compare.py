"""Numerical and latency comparison for Triton versus SM90 CUDA MLA decode."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
from pathlib import Path
from typing import Any, Callable

from benchmark.efficiency.gpu_guard import query_gpu_states


KV_DIM = 576
KV_LORA_RANK = 512
NUM_HEADS = 20
SM_SCALE = 1.0 / math.sqrt(KV_DIM)


def _parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _validate_gpu(physical_gpu: int) -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(physical_gpu):
        raise RuntimeError(
            "Set CUDA_VISIBLE_DEVICES to exactly the physical GPU passed through "
            f"--physical-gpu; got CUDA_VISIBLE_DEVICES={visible!r}, "
            f"physical_gpu={physical_gpu}."
        )
    state = query_gpu_states([physical_gpu])[0]
    if not state.idle:
        raise RuntimeError(
            f"GPU {physical_gpu} is not idle before the operator test: "
            f"util={state.utilization_pct:.0f}%, memory={state.memory_used_mib:.0f}MiB, "
            f"processes={state.compute_processes}."
        )


def _suite(args: argparse.Namespace) -> tuple[list[int], list[int], list[int]]:
    if args.suite == "quick":
        return [1, 16], [32], [29, 253]
    return args.batch_sizes, args.block_sizes, args.selected_blocks


def _make_inputs(
    torch,
    *,
    batch_size: int,
    block_size: int,
    selected_blocks: int,
    pattern: str,
    seed: int,
):
    rng = random.Random(seed)
    max_tokens = block_size * selected_blocks
    if pattern == "uniform":
        lengths = [max_tokens] * batch_size
    elif pattern == "ragged":
        choices = sorted(
            {
                block_size,
                max(block_size, max_tokens // 4),
                max(block_size, max_tokens // 2),
                max_tokens,
            }
        )
        lengths = [rng.choice(choices) for _ in range(batch_size)]
        if batch_size > 1:
            lengths[0], lengths[-1] = block_size, max_tokens
    else:
        raise ValueError(f"Unknown sequence pattern: {pattern}")

    num_pages = batch_size * selected_blocks
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    latent = torch.randn(
        num_pages * block_size,
        KV_DIM,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    block_table = torch.randperm(
        num_pages, device="cuda", dtype=torch.int32, generator=generator
    ).view(batch_size, selected_blocks)
    query = torch.randn(
        batch_size,
        NUM_HEADS,
        KV_DIM,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    sequence_lengths = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    return query, latent, block_table.contiguous(), sequence_lengths


def _latency_us(torch, call: Callable[[], Any], *, warmups: int, iterations: int) -> float:
    for _ in range(warmups):
        call()
    torch.cuda.synchronize()
    values = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            call()
        end.record()
        torch.cuda.synchronize()
        values.append(start.elapsed_time(end) * 1000.0 / iterations)
    return statistics.median(values)


def _error_metrics(torch, reference, candidate) -> dict[str, Any]:
    reference_f32 = reference.float()
    candidate_f32 = candidate.float()
    delta = (candidate_f32 - reference_f32).abs()
    finite = bool(torch.isfinite(candidate_f32).all().item())
    return {
        "all_finite": finite,
        "max_abs_diff": float(delta.max().item()),
        "mean_abs_diff": float(delta.mean().item()),
        "rmse": float(torch.sqrt(torch.mean(delta * delta)).item()),
    }


def _run_case(
    torch,
    triton_decode,
    sm90_kernel,
    args: argparse.Namespace,
    *,
    batch_size: int,
    block_size: int,
    selected_blocks: int,
    pattern: str,
    mode: str,
    seed: int,
) -> dict[str, Any]:
    query, latent, block_table, sequence_lengths = _make_inputs(
        torch,
        batch_size=batch_size,
        block_size=block_size,
        selected_blocks=selected_blocks,
        pattern=pattern,
        seed=seed,
    )
    reference = query.new_empty((batch_size, NUM_HEADS, KV_LORA_RANK))
    candidate = torch.empty_like(reference)

    def run_reference():
        return triton_decode(
            query,
            latent,
            block_table,
            sequence_lengths,
            SM_SCALE,
            block_size,
            KV_LORA_RANK,
            reference,
        )

    decoder = None
    if mode.startswith("stateless"):
        splits = 1 if mode == "stateless_s1" else 4

        def run_candidate():
            return sm90_kernel.decode_blocktable_mla_cuda(
                query,
                latent,
                block_table,
                sequence_lengths,
                SM_SCALE,
                block_size,
                KV_LORA_RANK,
                candidate,
                splits=splits,
            )

        run_candidate_with_plan = run_candidate
    else:
        buffers = sm90_kernel.allocate_mla_buffers(
            batch_size,
            NUM_HEADS,
            block_size,
            selected_blocks,
            query.device,
        )
        decoder = sm90_kernel.make_mla_decoder(
            batch_size,
            NUM_HEADS,
            block_size,
            selected_blocks,
            buffers,
        )
        decoder.plan(sequence_lengths)

        def run_candidate():
            decoder.run(query, latent, block_table, candidate, SM_SCALE)

        def run_candidate_with_plan():
            decoder.plan(sequence_lengths)
            decoder.run(query, latent, block_table, candidate, SM_SCALE)

    run_reference()
    run_candidate()
    torch.cuda.synchronize()
    metrics = _error_metrics(torch, reference, candidate)

    graph_latency_us = None
    if mode == "cuda_graph":
        for _ in range(3):
            run_candidate_with_plan()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run_candidate_with_plan()
        graph.replay()
        torch.cuda.synchronize()
        metrics = _error_metrics(torch, reference, candidate)
        graph_latency_us = _latency_us(
            torch, graph.replay, warmups=args.warmups, iterations=args.iterations
        )

    reference_us = _latency_us(
        torch, run_reference, warmups=args.warmups, iterations=args.iterations
    )
    candidate_run_us = _latency_us(
        torch, run_candidate, warmups=args.warmups, iterations=args.iterations
    )
    candidate_plan_run_us = _latency_us(
        torch,
        run_candidate_with_plan,
        warmups=args.warmups,
        iterations=args.iterations,
    )
    passed = (
        metrics["all_finite"]
        and metrics["max_abs_diff"] <= args.max_abs_threshold
        and metrics["mean_abs_diff"] <= args.mean_abs_threshold
    )
    return {
        "batch_size": batch_size,
        "block_size": block_size,
        "selected_blocks": selected_blocks,
        "selected_tokens_max": selected_blocks * block_size,
        "pattern": pattern,
        "mode": mode,
        "seed": seed,
        **metrics,
        "reference_us": reference_us,
        "candidate_run_us": candidate_run_us,
        "candidate_plan_run_us": candidate_plan_run_us,
        "cuda_graph_replay_us": graph_latency_us,
        "candidate_run_speedup": reference_us / candidate_run_us,
        "candidate_plan_run_speedup": reference_us / candidate_plan_run_us,
        "passed": passed,
        "decoder_target_ctas": getattr(decoder, "target_ctas", None),
        "decoder_minb": getattr(decoder, "minb", None),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--suite", choices=["quick", "full"], default="quick")
    parser.add_argument("--batch-sizes", type=_parse_ints, default=[1, 8, 16, 32, 64, 128])
    parser.add_argument("--block-sizes", type=_parse_ints, default=[16, 32, 64])
    parser.add_argument("--selected-blocks", type=_parse_ints, default=[29, 61, 125, 253])
    parser.add_argument(
        "--modes",
        default="stateless_s1,stateless_s4,work_queue,cuda_graph",
    )
    parser.add_argument("--patterns", default="uniform,ragged")
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-abs-threshold", type=float, default=8e-3)
    parser.add_argument("--mean-abs-threshold", type=float, default=5e-4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_gpu(args.physical_gpu)
    os.environ.setdefault("SGLANG_ENABLE_TORCH_COMPILE", "0")
    import torch

    from vortex_torch.engine.sgl.attention_backend import cuda_mla_sm90_kernel
    from vortex_torch.engine.sgl.attention_backend.triton_mla_kernel import (
        decode_blocktable_mla,
    )

    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Expected exactly one visible GPU, got {torch.cuda.device_count()}."
        )
    capability = torch.cuda.get_device_capability(0)
    if capability != (9, 0):
        raise RuntimeError(f"SM90 comparison requires capability (9, 0), got {capability}.")

    batch_sizes, block_sizes, selected_blocks_values = _suite(args)
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    patterns = [item.strip() for item in args.patterns.split(",") if item.strip()]
    valid_modes = {"stateless_s1", "stateless_s4", "work_queue", "cuda_graph"}
    if not modes or not set(modes) <= valid_modes:
        raise ValueError(f"Invalid modes {modes}; expected subset of {sorted(valid_modes)}.")
    if not patterns or not set(patterns) <= {"uniform", "ragged"}:
        raise ValueError(f"Invalid patterns: {patterns}.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    case_index = 0
    for block_size in block_sizes:
        for selected_blocks in selected_blocks_values:
            for batch_size in batch_sizes:
                for pattern in patterns:
                    for mode in modes:
                        case_index += 1
                        row = _run_case(
                            torch,
                            decode_blocktable_mla,
                            cuda_mla_sm90_kernel,
                            args,
                            batch_size=batch_size,
                            block_size=block_size,
                            selected_blocks=selected_blocks,
                            pattern=pattern,
                            mode=mode,
                            seed=args.seed + case_index,
                        )
                        rows.append(row)
                        payload = {
                            "status": "running",
                            "suite": args.suite,
                            "physical_gpu": args.physical_gpu,
                            "thresholds": {
                                "max_abs_diff": args.max_abs_threshold,
                                "mean_abs_diff": args.mean_abs_threshold,
                            },
                            "cases": rows,
                        }
                        args.output.write_text(
                            json.dumps(payload, indent=2), encoding="utf-8"
                        )
                        print(
                            f"[{case_index}] bs={batch_size} block={block_size} "
                            f"selected={selected_blocks} {pattern} {mode}: "
                            f"max={row['max_abs_diff']:.3e} mean={row['mean_abs_diff']:.3e} "
                            f"speedup={row['candidate_run_speedup']:.3f}x "
                            f"{'PASS' if row['passed'] else 'FAIL'}",
                            flush=True,
                        )
                        del row
                        torch.cuda.empty_cache()

    failed = sum(not row["passed"] for row in rows)
    payload["status"] = "success" if failed == 0 else "correctness_failed"
    payload["failed_cases"] = failed
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if failed:
        raise SystemExit(f"{failed} operator comparison cases failed.")


if __name__ == "__main__":
    main()
