# SPDX-License-Identifier: Apache-2.0
"""Matched fixed-batch and churn efficiency probe for a Vortex SGLang server.

The metric protocol is matched to Sparse-vLLM's current efficiency probe; the
trace generator retains the original migrated workload contract.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.efficiency.hardware_monitor import GPUHardwareMonitor
from benchmark.efficiency.sglang_adapter import check_server, run_trace
from benchmark.efficiency.workload import (
    TRACE_GENERATOR_VERSION,
    build_request_trace,
    derive_trace_seed,
    trace_metadata,
)


SOURCE_PROVENANCE = {
    "repository": "Sparse-vLLM",
    "probe_commit": "f09ef4cb796ea69d59d970894429c8c185269bfa",
    "workload_commit": "eb00c1d313b1dc7856cb838a994174f7b1624d55",
}


class HardwareMetricError(RuntimeError):
    pass


def _parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile from an empty list.")
    ordered = sorted(float(value) for value in values)
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _installed_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_metadata() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
            )
            return result.stdout.strip() or None
        except Exception:
            return None

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _monitor_gpu_ids(value: str) -> list[int]:
    gpu_ids = _parse_ints(value)
    if not gpu_ids or len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError(f"--monitor-gpus must contain unique physical IDs, got {gpu_ids}.")
    return gpu_ids


def _gpu_metadata(gpu_ids: list[int]) -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            ",".join(map(str, gpu_ids)),
            "--query-gpu=index,uuid,name,compute_cap,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6:
            raise RuntimeError(f"Unexpected nvidia-smi metadata row: {line!r}")
        rows.append(
            {
                "physical_device_index": int(fields[0]),
                "uuid": fields[1],
                "name": fields[2],
                "compute_capability": fields[3],
                "total_memory_mib": int(fields[4]),
                "driver_version": fields[5],
            }
        )
    if {row["physical_device_index"] for row in rows} != set(gpu_ids):
        raise RuntimeError("GPU metadata does not cover the requested physical IDs.")
    return rows


def _vocab_size(model_path: str) -> int:
    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Local model config is required: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    vocab_size = int(config["vocab_size"])
    if vocab_size <= 1:
        raise ValueError(f"Invalid vocab_size={vocab_size} in {config_path}.")
    return vocab_size


def _scenario_label(scenario: str) -> str:
    return {
        "fixed": "fixed_batch",
        "churn": "oversubscribed_churn",
    }[scenario]


def _request_count(scenario: str, concurrency: int, churn_multiplier: int) -> int:
    return concurrency if scenario == "fixed" else concurrency * churn_multiplier


def _warmup_request_count(
    scenario: str,
    concurrency: int,
    churn_multiplier: int,
) -> int:
    measured_count = _request_count(scenario, concurrency, churn_multiplier)
    return (
        concurrency
        if scenario == "fixed"
        else min(measured_count, max(concurrency, 2 * concurrency))
    )


def _trace(
    args: argparse.Namespace,
    *,
    scenario: str,
    phase: str,
    prompt_len: int,
    output_len: int,
    concurrency: int,
    iteration: int,
    request_count: int,
):
    scenario_label = _scenario_label(scenario)
    seed = derive_trace_seed(
        args.seed,
        scenario=scenario_label,
        phase=phase,
        nominal_prompt_len=prompt_len,
        nominal_output_len=output_len,
        concurrency=concurrency,
        iteration=iteration,
    )
    return build_request_trace(
        seed=seed,
        request_count=request_count,
        nominal_prompt_len=prompt_len,
        nominal_output_len=output_len,
        vocab_size=args.vocab_size,
        prompt_jitter_fraction=args.prompt_length_jitter,
        output_jitter_fraction=args.output_length_jitter,
        vary_output_lengths=scenario_label == "oversubscribed_churn",
    )


def _hardware_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    if summary.get("status") != "success":
        raise HardwareMetricError(f"GPU hardware sampling failed: {summary}")
    aggregate = summary["aggregate"]
    gpus = summary["gpus"]
    return {
        "metric_source": "nvidia-smi sampled activity",
        "sample_count": int(summary["total_samples"]),
        "sampling_interval_ms": int(summary["sampling_interval_ms"]),
        "gpu_compute_activity_pct_mean": float(aggregate["mean_compute_util_pct"]),
        "gpu_memory_io_activity_pct_mean": float(
            aggregate["mean_memory_io_activity_pct"]
        ),
        "gpu_active_duty_pct_mean": float(
            aggregate["mean_coarse_gpu_active_duty_pct"]
        ),
        "gpu_power_w_mean_total": float(aggregate["avg_total_power_w"]),
        "peak_vram_gb_max": max(float(row["peak_vram_gb"]) for row in gpus.values()),
        "per_gpu": gpus,
    }


def _summarize_case(
    args: argparse.Namespace,
    *,
    scenario: str,
    prompt_len: int,
    output_len: int,
    concurrency: int,
    iteration_records: list[dict[str, Any]],
    hardware: dict[str, Any],
) -> dict[str, Any]:
    traces = [record["trace"] for record in iteration_records]
    prompt_lengths = [length for trace in traces for length in trace["prompt_lengths"]]
    output_lengths = [length for trace in traces for length in trace["output_lengths"]]

    row: dict[str, Any] = {
        "engine": "vortex",
        "sparse_method": args.sparse_method,
        "backend_label": args.backend_label,
        "protocol_label": args.backend_label,
        "protocol": {
            "trace_generator_version": TRACE_GENERATOR_VERSION,
            "request_arrival": "single_burst",
            "prefix_caching_enabled": False,
        },
        "scenario": "fixed_batch" if scenario == "fixed" else "oversubscribed_churn",
        "prompt_len": prompt_len,
        "output_len": output_len,
        "prompt_len_min": min(prompt_lengths),
        "prompt_len_max": max(prompt_lengths),
        "output_len_min": min(output_lengths),
        "output_len_max": max(output_lengths),
        "concurrency": concurrency,
        "request_count": int(iteration_records[0]["request_count"]),
        "request_throughput_rps": statistics.fmean(
            record["request_throughput_rps"] for record in iteration_records
        ),
        "input_token_throughput_tps": statistics.fmean(
            record["input_token_throughput_tps"] for record in iteration_records
        ),
        "output_token_throughput_tps": statistics.fmean(
            record["output_token_throughput_tps"] for record in iteration_records
        ),
        "total_token_throughput_tps": statistics.fmean(
            record["total_token_throughput_tps"] for record in iteration_records
        ),
        "status": "success",
        "actual_hardware_metrics": hardware,
        **{key: value for key, value in hardware.items() if key != "per_gpu"},
    }
    if scenario == "fixed":
        ttfts = [float(record["ttft_ms"]) for record in iteration_records]
        tpots = [
            float(record["tpot_ms"])
            for record in iteration_records
            if record["tpot_ms"] is not None
        ]
        row.update(
            {
                "sequence_replacements": 0,
                "ttft_ms_mean": round(statistics.fmean(ttfts), 2),
                "ttft_ms_p50": round(_percentile(ttfts, 0.50), 2),
                "ttft_ms_p99": round(_percentile(ttfts, 0.99), 2),
                "tpot_ms_mean": round(statistics.fmean(tpots), 2) if tpots else None,
                "decode_metric_status": "success" if tpots else "skipped_by_policy",
            }
        )
    else:
        requests = [
            request
            for record in iteration_records
            for request in record["request_results"]
        ]
        ttfts = [float(request["ttft_ms"]) for request in requests]
        latencies = [float(request["latency_ms"]) for request in requests]
        tpots = [
            float(request["tpot_ms"])
            for request in requests
            if request["tpot_ms"] is not None
        ]
        request_count = int(iteration_records[0]["request_count"])
        row.update(
            {
                "sequence_replacements": request_count - concurrency,
                "ttft_ms_mean": statistics.fmean(ttfts),
                "ttft_ms_p50": _percentile(ttfts, 0.50),
                "ttft_ms_p99": _percentile(ttfts, 0.99),
                "latency_ms_p50": _percentile(latencies, 0.50),
                "latency_ms_p99": _percentile(latencies, 0.99),
                "tpot_ms_mean": statistics.fmean(tpots) if tpots else None,
            }
        )
    return row


async def _run_case(
    args: argparse.Namespace,
    *,
    scenario: str,
    prompt_len: int,
    output_len: int,
    concurrency: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    request_count = _request_count(
        scenario, concurrency, args.churn_request_multiplier
    )
    for warmup_index in range(args.num_warmups):
        warmup_count = _warmup_request_count(
            scenario, concurrency, args.churn_request_multiplier
        )
        print(
            f"  [Warmup {warmup_index + 1}/{args.num_warmups}] "
            f"requests={warmup_count}",
            flush=True,
        )
        warmup_trace = _trace(
            args,
            scenario=scenario,
            phase="warmup",
            prompt_len=prompt_len,
            output_len=output_len,
            concurrency=concurrency,
            iteration=warmup_index,
            request_count=warmup_count,
        )
        await run_trace(
            args.server_url,
            warmup_trace,
            timeout_s=args.request_timeout_s,
        )

    case_name = f"{scenario}-p{prompt_len}-o{output_len}-c{concurrency}"
    monitor = GPUHardwareMonitor(
        args.monitor_gpu_ids,
        interval_ms=args.hardware_sampling_interval_ms,
        output_file=Path(args.output_dir) / "case_hardware" / f"{case_name}.json",
    )
    records: list[dict[str, Any]] = []
    monitor.start()
    try:
        for iteration in range(args.num_iters):
            traces = _trace(
                args,
                scenario=scenario,
                phase="measure",
                prompt_len=prompt_len,
                output_len=output_len,
                concurrency=concurrency,
                iteration=iteration,
                request_count=request_count,
            )
            trace_result = await run_trace(
                args.server_url,
                traces,
                timeout_s=args.request_timeout_s,
            )
            request_results = list(trace_result.request_results)
            elapsed_s = trace_result.elapsed_s
            total_input = sum(trace.prompt_len for trace in traces)
            total_output = sum(result.generated_tokens for result in request_results)
            protocol = {
                "trace_generator_version": TRACE_GENERATOR_VERSION,
                "request_arrival": "single_burst",
                "prefix_caching_enabled": False,
            }
            record = {
                "engine": "vortex",
                "sparse_method": args.sparse_method,
                "backend_label": args.backend_label,
                "scenario": "fixed_batch" if scenario == "fixed" else "oversubscribed_churn",
                "prompt_len": prompt_len,
                "output_len": output_len,
                "concurrency": concurrency,
                "request_count": request_count,
                "iteration": iteration,
                "status": "success",
                "elapsed_s": elapsed_s,
                "request_throughput_rps": request_count / elapsed_s,
                "input_token_throughput_tps": total_input / elapsed_s,
                "output_token_throughput_tps": total_output / elapsed_s,
                "total_token_throughput_tps": (total_input + total_output) / elapsed_s,
                "trace": trace_metadata(traces),
                "profiler_breakdown": None,
                "profiler_status": "skipped_by_policy",
                "protocol": protocol,
                "protocol_label": args.backend_label,
            }
            if scenario == "fixed":
                batch_tpot_ms = trace_result.batch_tpot_ms
                record.update(
                    {
                        "ttft_ms": round(trace_result.batch_ttft_ms, 2),
                        "tpot_ms": (
                            None
                            if batch_tpot_ms is None
                            else round(batch_tpot_ms, 2)
                        ),
                        "decode_metric_status": (
                            "success"
                            if batch_tpot_ms is not None
                            else "skipped_by_policy"
                        ),
                    }
                )
            else:
                record.update(
                    {
                        "queued_request_count": request_count - concurrency,
                        "step_count": None,
                        "engine_init_s": None,
                        "startup_decode_cuda_graph": None,
                        "decode_cuda_graph_before": None,
                        "decode_cuda_graph_after": None,
                        "decode_cuda_graph_counter_delta": None,
                        "request_results": [
                            result.metadata() for result in request_results
                        ],
                    }
                )
            records.append(record)
            if scenario == "fixed":
                print(
                    f"  Iter {iteration + 1}/{args.num_iters}: "
                    f"TTFT={record['ttft_ms']:.1f}ms | "
                    f"TPOT={record['tpot_ms'] if record['tpot_ms'] is not None else 'n/a'}ms | "
                    f"Output={record['output_token_throughput_tps']:.1f} tok/s",
                    flush=True,
                )
            else:
                print(
                    f"  Iter {iteration + 1}/{args.num_iters}: "
                    f"elapsed={elapsed_s:.3f}s | "
                    f"Output={record['output_token_throughput_tps']:.1f} tok/s",
                    flush=True,
                )
    finally:
        hardware_summary = monitor.stop()
    hardware = _hardware_metrics(hardware_summary)
    summary = _summarize_case(
        args,
        scenario=scenario,
        prompt_len=prompt_len,
        output_len=output_len,
        concurrency=concurrency,
        iteration_records=records,
        hardware=hardware,
    )
    return summary, records


def _attach_churn_comparisons(rows: list[dict[str, Any]]) -> None:
    fixed_by_case: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = (
            row.get("engine"),
            row.get("sparse_method"),
            row.get("protocol_label"),
            row.get("prompt_len"),
            row.get("output_len"),
            row.get("concurrency"),
        )
        if row.get("scenario") == "fixed_batch":
            fixed_by_case[key] = row
    for row in rows:
        if row.get("scenario") != "oversubscribed_churn":
            continue
        key = (
            row.get("engine"),
            row.get("sparse_method"),
            row.get("protocol_label"),
            row.get("prompt_len"),
            row.get("output_len"),
            row.get("concurrency"),
        )
        fixed = fixed_by_case.get(key)
        if fixed is None:
            row["fixed_batch_comparison_status"] = "skipped_by_policy"
            continue
        fixed_output_tps = float(fixed["output_token_throughput_tps"])
        fixed_request_rps = float(fixed["request_throughput_rps"])
        if fixed_output_tps <= 0 or fixed_request_rps <= 0:
            raise ValueError(f"Fixed-batch throughput must be positive: {fixed}")
        row["fixed_batch_comparison_status"] = "success"
        row["churn_output_tps_ratio_vs_fixed_batch"] = (
            float(row["output_token_throughput_tps"]) / fixed_output_tps
        )
        row["churn_request_rps_ratio_vs_fixed_batch"] = (
            float(row["request_throughput_rps"]) / fixed_request_rps
        )
        row["churn_ttft_p99_delta_ms_vs_fixed_batch"] = (
            float(row["ttft_ms_p99"]) - float(fixed["ttft_ms_p99"])
        )


def _attach_saturation_metrics(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row.get("engine"),
            row.get("sparse_method"),
            row.get("protocol_label"),
            row.get("scenario"),
            row.get("prompt_len"),
            row.get("output_len"),
        )
        groups.setdefault(key, []).append(row)

    for group_rows in groups.values():
        ordered = sorted(group_rows, key=lambda row: int(row["concurrency"]))
        concurrencies = [int(row["concurrency"]) for row in ordered]
        if len(concurrencies) != len(set(concurrencies)):
            raise ValueError(
                f"Saturation sweep contains duplicate concurrencies: {concurrencies}."
            )
        rates = [float(row["output_token_throughput_tps"]) for row in ordered]
        if any(rate <= 0 for rate in rates):
            raise ValueError(f"Saturation sweep throughput must be positive: {rates}.")
        observed_peak = max(rates)
        base_concurrency = concurrencies[0]
        base_rate = rates[0]
        saturation_concurrency = next(
            concurrency
            for concurrency, rate in zip(concurrencies, rates)
            if rate >= 0.95 * observed_peak
        )
        analysis_status = "success" if len(ordered) > 1 else "skipped_by_policy"
        for index, (row, concurrency, rate) in enumerate(
            zip(ordered, concurrencies, rates)
        ):
            row["saturation_analysis_status"] = analysis_status
            row["output_tps_pct_of_observed_sweep_peak"] = rate / observed_peak * 100.0
            row["output_tps_scaling_efficiency_pct_vs_min_concurrency"] = (
                (rate / base_rate) / (concurrency / base_concurrency) * 100.0
            )
            row["marginal_output_tps_gain_pct_vs_previous_concurrency"] = (
                None if index == 0 else (rate / rates[index - 1] - 1.0) * 100.0
            )
            row["observed_output_saturation_threshold_pct"] = 95.0
            row["observed_output_saturation_concurrency"] = (
                saturation_concurrency if analysis_status == "success" else None
            )


def _format_report(rows: list[dict[str, Any]], args: argparse.Namespace) -> str:
    lines = [
        "# Standardized LLM Efficiency Benchmark Report",
        "",
        f"- **Model**: `{Path(args.model_path).name}`",
        "- **Tensor parallel size**: `1`",
        "- **GPU metrics**: directly sampled activity; no theoretical MFU/MBU estimates",
        f"- **Timestamp**: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
        "| System / Method | Scenario | Prompt Range | Output Range | Concurrency | Req/s | Output tok/s | Observed peak | Scaling efficiency | TTFT p50/p99 (ms) | GPU compute activity | GPU memory I/O activity | Peak VRAM (GB) | Status |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    def number(value: Any, precision: int) -> str:
        return "n/a" if value is None else f"{float(value):.{precision}f}"

    for row in rows:
        lines.append(
            f"| `{row['protocol_label']}` | {row['scenario']} "
            f"| {row['prompt_len_min']}-{row['prompt_len_max']} "
            f"| {row['output_len_min']}-{row['output_len_max']} "
            f"| {row['concurrency']} | {number(row.get('request_throughput_rps'), 2)} "
            f"| {number(row.get('output_token_throughput_tps'), 2)} "
            f"| {number(row.get('output_tps_pct_of_observed_sweep_peak'), 1)}% "
            f"| {number(row.get('output_tps_scaling_efficiency_pct_vs_min_concurrency'), 1)}% "
            f"| {number(row.get('ttft_ms_p50'), 2)}/{number(row.get('ttft_ms_p99'), 2)} "
            f"| {number(row.get('gpu_compute_activity_pct_mean'), 1)}% "
            f"| {number(row.get('gpu_memory_io_activity_pct_mean'), 1)}% "
            f"| {number(row.get('peak_vram_gb_max'), 2)} | {row['status']} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vortex SGLang matched efficiency probe")
    parser.add_argument("--server-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--backend-label", required=True)
    parser.add_argument("--sparse-method", default="quest")
    parser.add_argument("--prompt-lens", type=_parse_ints, default=[4096, 32768])
    parser.add_argument("--output-lens", type=_parse_ints, default=[64])
    parser.add_argument("--batch-sizes", type=_parse_ints, default=[1, 16])
    parser.add_argument("--scenario", choices=["fixed", "churn", "all"], default="fixed")
    parser.add_argument("--num-warmups", type=int, default=1)
    parser.add_argument("--num-iters", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-length-jitter", type=float, default=0.10)
    parser.add_argument("--output-length-jitter", type=float, default=0.25)
    parser.add_argument("--churn-request-multiplier", type=int, default=4)
    parser.add_argument("--request-timeout-s", type=float, default=3600.0)
    parser.add_argument("--monitor-gpus", required=True)
    parser.add_argument("--hardware-sampling-interval-ms", type=int, default=100)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("prompt_lens", "output_lens", "batch_sizes"):
        values = getattr(args, name)
        if not values or any(value <= 0 for value in values):
            raise ValueError(f"--{name.replace('_', '-')} requires positive integers.")
    if args.num_warmups < 0 or args.num_iters <= 0:
        raise ValueError("Warmups must be non-negative and iterations positive.")
    if args.churn_request_multiplier < 2:
        raise ValueError("--churn-request-multiplier must be at least 2.")
    if args.scenario == "all":
        raise ValueError(
            "Strict Sparse-vLLM lifecycle matching requires separate server instances "
            "for fixed and churn. Run --scenario fixed, restart the Vortex server, "
            "then run --scenario churn into a different output directory."
        )
    if not 0.0 <= args.prompt_length_jitter < 1.0:
        raise ValueError("--prompt-length-jitter must be in [0, 1).")
    if not 0.0 <= args.output_length_jitter < 1.0:
        raise ValueError("--output-length-jitter must be in [0, 1).")


async def _main_async(args: argparse.Namespace) -> None:
    _validate_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = ("run_manifest.json", "raw_samples.jsonl", "request_samples.jsonl", "summary.json", "comparison_report.md", "run_status.json")
    collisions = [name for name in artifacts if (output_dir / name).exists()]
    if collisions:
        raise FileExistsError(
            f"Refusing to mix benchmark runs in {output_dir}: {collisions}."
        )

    args.monitor_gpu_ids = _monitor_gpu_ids(args.monitor_gpus)
    args.vocab_size = _vocab_size(args.model_path)
    await check_server(args.server_url)
    manifest = {
        "manifest_version": "1.0",
        "timestamp": datetime.now().isoformat(),
        "status": "running",
        "command": [sys.executable, *sys.argv],
        "args": {key: value for key, value in vars(args).items() if key not in {"monitor_gpu_ids"}},
        "git": _git_metadata(),
        "source_provenance": SOURCE_PROVENANCE,
        "trace_generator_version": TRACE_GENERATOR_VERSION,
        "workload": {
            "trace_generator_version": TRACE_GENERATOR_VERSION,
            "prefix_caching_enabled": False,
            "cross_engine_trace_contract": "same seed, token IDs, and per-request lengths",
            "iteration_prompt_reuse_allowed": False,
            "request_arrival": "single_burst",
            "fixed_ttft": "batch_start_to_all_requests_first_token",
            "churn_ttft": "request_submit_to_first_token_including_server_queue",
        },
        "environment": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "torch": _installed_version("torch"),
            "triton": _installed_version("triton"),
            "flashinfer-python": _installed_version("flashinfer-python"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "physical_gpus": _gpu_metadata(args.monitor_gpu_ids),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "raw_samples.jsonl").write_text("", encoding="utf-8")
    (output_dir / "request_samples.jsonl").write_text("", encoding="utf-8")

    rows: list[dict[str, Any]] = []
    scenarios = [args.scenario]
    try:
        for scenario in scenarios:
            for prompt_len in args.prompt_lens:
                for output_len in args.output_lens:
                    for concurrency in args.batch_sizes:
                        print(
                            f"[probe] {scenario=} {prompt_len=} {output_len=} {concurrency=}",
                            flush=True,
                        )
                        summary, records = await _run_case(
                            args,
                            scenario=scenario,
                            prompt_len=prompt_len,
                            output_len=output_len,
                            concurrency=concurrency,
                        )
                        rows.append(summary)
                        with (output_dir / "raw_samples.jsonl").open("a", encoding="utf-8") as handle:
                            for record in records:
                                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        with (output_dir / "request_samples.jsonl").open("a", encoding="utf-8") as handle:
                            for record in records:
                                requests = (
                                    record["trace"]["requests"]
                                    if scenario == "fixed"
                                    else record["request_results"]
                                )
                                for request in requests:
                                    handle.write(
                                        json.dumps(
                                            {
                                                "backend_label": args.backend_label,
                                                "engine": "vortex",
                                                "sparse_method": args.sparse_method,
                                                "scenario": (
                                                    "fixed_batch"
                                                    if scenario == "fixed"
                                                    else "oversubscribed_churn"
                                                ),
                                                "nominal_prompt_len": prompt_len,
                                                "nominal_output_len": output_len,
                                                "concurrency": concurrency,
                                                "iteration": record["iteration"],
                                                **request,
                                            },
                                            ensure_ascii=False,
                                        )
                                        + "\n"
                                    )
    except Exception as exc:
        failure = {
            "status": "metric_failed" if isinstance(exc, HardwareMetricError) else "model_failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        manifest.update(failure)
        (output_dir / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (output_dir / "run_status.json").write_text(
            json.dumps(failure, indent=2), encoding="utf-8"
        )
        raise

    _attach_churn_comparisons(rows)
    _attach_saturation_metrics(rows)
    (output_dir / "summary.json").write_text(
        json.dumps({"status": "success", "summary": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report = _format_report(rows, args)
    (output_dir / "comparison_report.md").write_text(report, encoding="utf-8")
    manifest["status"] = "success"
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "run_status.json").write_text(
        json.dumps({"status": "success"}, indent=2), encoding="utf-8"
    )
    print(report)


def main() -> None:
    asyncio.run(_main_async(parse_args()))


if __name__ == "__main__":
    main()
