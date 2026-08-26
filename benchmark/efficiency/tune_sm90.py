"""Sweep SM90 MLA work-queue geometry on representative GLM-4.7 cases."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from pathlib import Path

from benchmark.efficiency.gpu_guard import query_gpu_states
from benchmark.efficiency.operator_compare import (
    KV_LORA_RANK,
    NUM_HEADS,
    SM_SCALE,
    _error_metrics,
    _latency_us,
    _make_inputs,
    _parse_ints,
)


DEFAULT_CASES = (
    (1, 29, "uniform"),
    (1, 253, "uniform"),
    (16, 61, "ragged"),
    (64, 61, "ragged"),
    (64, 253, "uniform"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--minb", type=_parse_ints, default=[2, 3, 4])
    parser.add_argument("--chunk-min", type=_parse_ints, default=[32, 64, 128, 256])
    parser.add_argument("--max-split-cap", type=_parse_ints, default=[4, 8, 16, 32])
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--layers", type=int, default=46)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _validate_gpu(args: argparse.Namespace) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must match --physical-gpu exactly.")
    state = query_gpu_states([args.physical_gpu])[0]
    if not state.idle:
        raise RuntimeError(f"GPU {args.physical_gpu} is not idle: {state}.")


def main() -> None:
    args = parse_args()
    _validate_gpu(args)
    os.environ.setdefault("SGLANG_ENABLE_TORCH_COMPILE", "0")

    import torch

    from vortex_torch.engine.sgl.attention_backend import cuda_mla_sm90_kernel as sm90
    from vortex_torch.engine.sgl.attention_backend.triton_mla_kernel import (
        decode_blocktable_mla,
    )

    if torch.cuda.get_device_capability(0) != (9, 0):
        raise RuntimeError("SM90 tuning requires an H100-class device.")

    inputs = []
    for case_index, (bs, selected, pattern) in enumerate(DEFAULT_CASES):
        tensors = _make_inputs(
            torch,
            batch_size=bs,
            block_size=args.block_size,
            selected_blocks=selected,
            pattern=pattern,
            seed=args.seed + case_index,
        )
        query, latent, block_table, seqlens = tensors
        reference = query.new_empty((bs, NUM_HEADS, KV_LORA_RANK))
        decode_blocktable_mla(
            query,
            latent,
            block_table,
            seqlens,
            SM_SCALE,
            args.block_size,
            KV_LORA_RANK,
            reference,
        )
        inputs.append((bs, selected, pattern, tensors, reference))
    torch.cuda.synchronize()

    rows = []
    configs = itertools.product(args.minb, args.chunk_min, args.max_split_cap)
    for config_index, (minb, chunk_min, max_split_cap) in enumerate(configs, start=1):
        cases = []
        valid = True
        for bs, selected, pattern, tensors, reference in inputs:
            query, latent, block_table, seqlens = tensors
            candidate = torch.empty_like(reference)
            try:
                buffers = sm90.allocate_mla_buffers(
                    bs,
                    NUM_HEADS,
                    args.block_size,
                    selected,
                    query.device,
                    max_split_cap=max_split_cap,
                    chunk_min=chunk_min,
                    minb=minb,
                )
                decoder = sm90.make_mla_decoder(
                    bs,
                    NUM_HEADS,
                    args.block_size,
                    selected,
                    buffers,
                    max_split_cap=max_split_cap,
                    chunk_min=chunk_min,
                    minb=minb,
                )
                decoder.plan(seqlens)

                def run():
                    decoder.run(query, latent, block_table, candidate, SM_SCALE)

                def plan_run():
                    decoder.plan(seqlens)
                    run()

                run()
                torch.cuda.synchronize()
                error = _error_metrics(torch, reference, candidate)
                run_us = _latency_us(
                    torch, run, warmups=args.warmups, iterations=args.iterations
                )
                plan_run_us = _latency_us(
                    torch, plan_run, warmups=args.warmups, iterations=args.iterations
                )
                amortized_us = run_us + (plan_run_us - run_us) / args.layers
                passed = (
                    error["all_finite"]
                    and error["max_abs_diff"] <= 8e-3
                    and error["mean_abs_diff"] <= 5e-4
                )
                valid &= passed
                cases.append(
                    {
                        "batch_size": bs,
                        "selected_blocks": selected,
                        "pattern": pattern,
                        **error,
                        "passed": passed,
                        "run_us": run_us,
                        "plan_run_us": plan_run_us,
                        "amortized_per_layer_us": amortized_us,
                        "target_ctas": decoder.target_ctas,
                        "decoder_minb": decoder.minb,
                    }
                )
            except RuntimeError as exc:
                valid = False
                cases.append(
                    {
                        "batch_size": bs,
                        "selected_blocks": selected,
                        "pattern": pattern,
                        "passed": False,
                        "error": str(exc),
                    }
                )
                torch.cuda.synchronize()

        measured = [c["amortized_per_layer_us"] for c in cases if "amortized_per_layer_us" in c]
        geomean_us = (
            math.exp(sum(math.log(value) for value in measured) / len(measured))
            if measured
            else math.inf
        )
        row = {
            "minb": minb,
            "chunk_min": chunk_min,
            "max_split_cap": max_split_cap,
            "valid": valid,
            "geomean_amortized_us": geomean_us,
            "cases": cases,
        }
        rows.append(row)
        print(
            f"[{config_index}] minb={minb} chunk={chunk_min} cap={max_split_cap} "
            f"geomean={geomean_us:.3f}us {'PASS' if valid else 'FAIL'}",
            flush=True,
        )

    valid_rows = [row for row in rows if row["valid"]]
    valid_rows.sort(key=lambda row: row["geomean_amortized_us"])
    payload = {
        "status": "success" if valid_rows else "no_valid_configuration",
        "physical_gpu": args.physical_gpu,
        "block_size": args.block_size,
        "layers_for_plan_amortization": args.layers,
        "best": valid_rows[0] if valid_rows else None,
        "ranked_valid": valid_rows,
        "all_configs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not valid_rows:
        raise SystemExit("No tuning configuration passed correctness thresholds.")


if __name__ == "__main__":
    main()
