"""Validate and aggregate the sharded SM90 MLA formal benchmark matrix."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


PROMPTS = (4096, 16384, 32768)
OUTPUT_LEN = 128
CONCURRENCIES = (1, 8, 16, 32, 64)
SCENARIOS = ("fixed", "churn")
STAGES = ("triton_baseline", "sm90_arch_only", "sm90_tuned")
SUMMARY_SCENARIO = {"fixed": "fixed_batch", "churn": "continuous_churn"}
HIGHER_IS_BETTER = (
    "request_throughput_rps",
    "input_token_throughput_tps",
    "output_token_throughput_tps",
    "total_token_throughput_tps",
)
LOWER_IS_BETTER = (
    "ttft_ms_p50",
    "ttft_ms_p95",
    "ttft_ms_p99",
    "tpot_ms_p50",
    "tpot_ms_p95",
    "tpot_ms_p99",
    "latency_ms_p50",
    "latency_ms_p95",
    "latency_ms_p99",
)
CASE_KEY = ("scenario", "prompt_len", "output_len", "concurrency")
TRACE_KEY = (
    "scenario",
    "nominal_prompt_len",
    "nominal_output_len",
    "concurrency",
    "iteration",
    "request_index",
    "prompt_len",
    "requested_output_len",
    "prompt_digest",
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"expected an object in {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read valid JSONL from {path}: {exc}") from exc


def _validate_manifest(
    manifest: dict[str, Any], *, stage: str, scenario: str, prompt: int, path: Path
) -> None:
    args = manifest.get("args", {})
    expected = {
        "model_path": "/data2/pretrain_models/GLM-4.7-Flash",
        "prompt_lens": [prompt],
        "output_lens": [OUTPUT_LEN],
        "batch_sizes": list(CONCURRENCIES),
        "scenario": scenario,
        "num_warmups": 1,
        "num_iters": 2,
        "seed": 42,
        "monitor_gpus": "6",
    }
    for name, value in expected.items():
        if args.get(name) != value:
            raise ValueError(
                f"{path}: {name}={args.get(name)!r}, expected {value!r}"
            )
    if manifest.get("status") != "success":
        raise ValueError(f"{path}: manifest status is not success")
    physical_gpus = manifest.get("physical_gpus", [])
    if len(physical_gpus) != 1 or physical_gpus[0].get("physical_device_index") != 6:
        raise ValueError(f"{path}: expected exactly physical GPU 6")
    if physical_gpus[0].get("compute_capability") != "9.0":
        raise ValueError(f"{path}: expected SM90 compute capability 9.0")
    git = manifest.get("git", {})
    if git.get("branch") != "feat/sm90-support" or git.get("dirty") is not False:
        raise ValueError(f"{path}: expected clean feat/sm90-support provenance")
    expected_label = stage.replace("_", "-")
    if args.get("backend_label") != expected_label:
        raise ValueError(
            f"{path}: backend_label={args.get('backend_label')!r}, expected {expected_label!r}"
        )


def _validate_summary(
    payload: dict[str, Any], *, scenario: str, prompt: int, path: Path
) -> dict[tuple[Any, ...], dict[str, Any]]:
    if payload.get("status") != "success":
        raise ValueError(f"{path}: summary status is not success")
    rows = payload.get("summary")
    if not isinstance(rows, list) or len(rows) != len(CONCURRENCIES):
        raise ValueError(f"{path}: expected {len(CONCURRENCIES)} summary rows")
    indexed: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        if row.get("status") != "success":
            raise ValueError(f"{path}: case status is not success: {row}")
        expected_values = {
            "scenario": SUMMARY_SCENARIO[scenario],
            "prompt_len": prompt,
            "output_len": OUTPUT_LEN,
            "iterations": 2,
        }
        for name, value in expected_values.items():
            if row.get(name) != value:
                raise ValueError(
                    f"{path}: row {name}={row.get(name)!r}, expected {value!r}"
                )
        concurrency = row.get("concurrency")
        if concurrency not in CONCURRENCIES:
            raise ValueError(f"{path}: unexpected concurrency {concurrency!r}")
        for metric in HIGHER_IS_BETTER + LOWER_IS_BETTER:
            value = row.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{path}: invalid {metric}={value!r}")
        key = tuple(row[name] for name in CASE_KEY)
        if key in indexed:
            raise ValueError(f"{path}: duplicate case {key}")
        indexed[key] = row
    expected_concurrency = {row[3] for row in indexed}
    if expected_concurrency != set(CONCURRENCIES):
        raise ValueError(f"{path}: incomplete concurrency set {expected_concurrency}")
    return indexed


def _validate_samples(
    rows: list[dict[str, Any]], *, scenario: str, prompt: int, path: Path
) -> tuple[tuple[Any, ...], ...]:
    expected = sum(
        concurrency * (1 if scenario == "fixed" else 4) * 2
        for concurrency in CONCURRENCIES
    )
    if len(rows) != expected:
        raise ValueError(f"{path}: expected {expected} measured requests, got {len(rows)}")
    signatures = []
    for row in rows:
        if (
            row.get("scenario") != scenario
            or row.get("nominal_prompt_len") != prompt
            or row.get("nominal_output_len") != OUTPUT_LEN
            or row.get("concurrency") not in CONCURRENCIES
            or row.get("iteration") not in (0, 1)
            or row.get("status") != "success"
            or row.get("generated_tokens") != row.get("requested_output_len")
        ):
            raise ValueError(f"{path}: invalid measured request row: {row}")
        signatures.append(tuple(row[name] for name in TRACE_KEY))
    if len(set(signatures)) != len(signatures):
        raise ValueError(f"{path}: duplicate measured request identity")
    return tuple(sorted(signatures))


def load_stage(root: Path, stage: str) -> dict[str, Any]:
    cases: dict[tuple[Any, ...], dict[str, Any]] = {}
    traces: dict[tuple[str, int], tuple[tuple[Any, ...], ...]] = {}
    provenance = []
    for scenario in SCENARIOS:
        for prompt in PROMPTS:
            shard = root / stage / f"{scenario}_p{prompt}"
            manifest_path = shard / "run_manifest.json"
            summary_path = shard / "summary.json"
            samples_path = shard / "request_samples.jsonl"
            manifest = _read_json(manifest_path)
            _validate_manifest(
                manifest, stage=stage, scenario=scenario, prompt=prompt, path=manifest_path
            )
            shard_cases = _validate_summary(
                _read_json(summary_path),
                scenario=scenario,
                prompt=prompt,
                path=summary_path,
            )
            overlap = set(cases).intersection(shard_cases)
            if overlap:
                raise ValueError(f"{stage}: duplicate cross-shard cases {sorted(overlap)}")
            cases.update(shard_cases)
            traces[(scenario, prompt)] = _validate_samples(
                _read_jsonl(samples_path),
                scenario=scenario,
                prompt=prompt,
                path=samples_path,
            )
            provenance.append(
                {
                    "scenario": scenario,
                    "prompt": prompt,
                    "commit": manifest["git"]["commit"],
                    "gpu_uuid": manifest["physical_gpus"][0]["uuid"],
                }
            )
    if len(cases) != len(PROMPTS) * len(CONCURRENCIES) * len(SCENARIOS):
        raise ValueError(f"{stage}: expected 30 unique cases, got {len(cases)}")
    return {"cases": cases, "traces": traces, "provenance": provenance}


def _compare(
    baseline: dict[tuple[Any, ...], dict[str, Any]],
    candidate: dict[tuple[Any, ...], dict[str, Any]],
) -> list[dict[str, Any]]:
    if set(baseline) != set(candidate):
        raise ValueError("stage case sets do not match")
    rows = []
    for key in sorted(baseline):
        base = baseline[key]
        cand = candidate[key]
        row = {name: value for name, value in zip(CASE_KEY, key)}
        for metric in HIGHER_IS_BETTER:
            row[f"{metric}_speedup"] = cand[metric] / base[metric]
        for metric in LOWER_IS_BETTER:
            row[f"{metric}_speedup"] = base[metric] / cand[metric]
        rows.append(row)
    return rows


def aggregate(root: Path) -> dict[str, Any]:
    loaded = {stage: load_stage(root, stage) for stage in STAGES}
    reference_traces = loaded["triton_baseline"]["traces"]
    for stage in STAGES[1:]:
        if loaded[stage]["traces"] != reference_traces:
            raise ValueError(f"{stage}: request trace differs from Triton baseline")

    comparisons = {
        "arch_only_vs_triton": _compare(
            loaded["triton_baseline"]["cases"], loaded["sm90_arch_only"]["cases"]
        ),
        "tuned_vs_arch_only": _compare(
            loaded["sm90_arch_only"]["cases"], loaded["sm90_tuned"]["cases"]
        ),
        "tuned_vs_triton": _compare(
            loaded["triton_baseline"]["cases"], loaded["sm90_tuned"]["cases"]
        ),
    }
    aggregates = {}
    for name, rows in comparisons.items():
        aggregates[name] = {}
        for metric in (
            "output_token_throughput_tps_speedup",
            "ttft_ms_p50_speedup",
            "tpot_ms_p50_speedup",
            "latency_ms_p50_speedup",
        ):
            values = [row[metric] for row in rows]
            aggregates[name][f"{metric}_geomean"] = math.prod(values) ** (1 / len(values))
            aggregates[name][f"{metric}_min"] = min(values)
            aggregates[name][f"{metric}_max"] = max(values)

    return {
        "status": "success",
        "matrix": {
            "prompts": list(PROMPTS),
            "output": OUTPUT_LEN,
            "concurrency": list(CONCURRENCIES),
            "scenarios": list(SCENARIOS),
            "warmups": 1,
            "measured_iterations": 2,
            "cases_per_stage": 30,
        },
        "trace_match": True,
        "stage_aliases": {
            "sm90_migrated": {
                "source": "sm90_arch_only",
                "reason": "remaining CUDA Schedule.W candidate regressed all quick throughput cases",
            }
        },
        "provenance": {stage: loaded[stage]["provenance"] for stage in STAGES},
        "aggregates": aggregates,
        "comparisons": comparisons,
    }


def markdown(payload: dict[str, Any]) -> str:
    matrix = payload["matrix"]
    lines = [
        "# SM90 MLA Formal Matrix Report",
        "",
        f"- Matrix: prompt `{','.join(map(str, matrix['prompts']))}`, output `{matrix['output']}`, "
        f"concurrency `{','.join(map(str, matrix['concurrency']))}`, fixed + churn",
        f"- Repetitions: `{matrix['warmups']}` warmup + `{matrix['measured_iterations']}` measured",
        f"- Cases: `{matrix['cases_per_stage']}` per measured implementation",
        "- Trace audit: all measured request identities and prompt digests match across stages",
        "- `sm90 migrated` aliases `sm90 arch-only`: the remaining CUDA indexer candidate was rejected after regression",
        "",
        "## Aggregate speedups",
        "",
        "| Comparison | Output tok/s gmean [min,max] | TTFT p50 gmean | TPOT p50 gmean | E2E p50 gmean |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = {
        "arch_only_vs_triton": "arch-only / Triton",
        "tuned_vs_arch_only": "tuned / arch-only",
        "tuned_vs_triton": "tuned / Triton",
    }
    for name, label in labels.items():
        row = payload["aggregates"][name]
        prefix = "output_token_throughput_tps_speedup"
        lines.append(
            f"| {label} | {row[prefix + '_geomean']:.3f}x "
            f"[{row[prefix + '_min']:.3f}, {row[prefix + '_max']:.3f}] "
            f"| {row['ttft_ms_p50_speedup_geomean']:.3f}x "
            f"| {row['tpot_ms_p50_speedup_geomean']:.3f}x "
            f"| {row['latency_ms_p50_speedup_geomean']:.3f}x |"
        )

    for name, label in labels.items():
        lines.extend(
            [
                "",
                f"## {label}",
                "",
                "| Scenario | Prompt | C | Output tok/s | TTFT p50 | TPOT p50 | E2E p50 |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in payload["comparisons"][name]:
            lines.append(
                f"| {row['scenario']} | {row['prompt_len']} | {row['concurrency']} "
                f"| {row['output_token_throughput_tps_speedup']:.3f}x "
                f"| {row['ttft_ms_p50_speedup']:.3f}x "
                f"| {row['tpot_ms_p50_speedup']:.3f}x "
                f"| {row['latency_ms_p50_speedup']:.3f}x |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = aggregate(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown(payload), encoding="utf-8")
    args.output.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
