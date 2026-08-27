"""Compare matched Vortex efficiency-probe summary files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CASE_KEYS = ("scenario", "prompt_len", "output_len", "concurrency")
HIGHER_IS_BETTER = (
    "request_throughput_rps",
    "input_token_throughput_tps",
    "output_token_throughput_tps",
    "total_token_throughput_tps",
)
LOWER_IS_BETTER = (
    "ttft_ms_mean",
    "ttft_ms_p50",
    "ttft_ms_p99",
    "tpot_ms_mean",
    "latency_ms_p50",
    "latency_ms_p99",
)


def _load(path: Path) -> dict[tuple[Any, ...], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "success":
        raise ValueError(f"Probe did not succeed: {path}")
    result = {}
    for row in payload["summary"]:
        key = tuple(row[name] for name in CASE_KEYS)
        if key in result:
            raise ValueError(f"Duplicate case {key} in {path}")
        result[key] = row
    return result


def compare(baseline_path: Path, candidate_path: Path) -> dict[str, Any]:
    baseline = _load(baseline_path)
    candidate = _load(candidate_path)
    if set(baseline) != set(candidate):
        raise ValueError(
            "Summary case sets differ: "
            f"missing={sorted(set(baseline) - set(candidate))}, "
            f"extra={sorted(set(candidate) - set(baseline))}"
        )
    rows = []
    for key in sorted(baseline):
        base_row = baseline[key]
        candidate_row = candidate[key]
        row = {name: value for name, value in zip(CASE_KEYS, key)}
        for metric in HIGHER_IS_BETTER:
            base_value = float(base_row[metric])
            candidate_value = float(candidate_row[metric])
            row[f"{metric}_baseline"] = base_value
            row[f"{metric}_candidate"] = candidate_value
            row[f"{metric}_speedup"] = candidate_value / base_value
        for metric in LOWER_IS_BETTER:
            if metric not in base_row and metric not in candidate_row:
                continue
            if metric not in base_row or metric not in candidate_row:
                raise ValueError(
                    f"Metric availability differs for {key}: {metric}"
                )
            if base_row[metric] is None and candidate_row[metric] is None:
                continue
            if base_row[metric] is None or candidate_row[metric] is None:
                raise ValueError(
                    f"Metric status differs for {key}: {metric}"
                )
            base_value = float(base_row[metric])
            candidate_value = float(candidate_row[metric])
            row[f"{metric}_baseline"] = base_value
            row[f"{metric}_candidate"] = candidate_value
            row[f"{metric}_speedup"] = base_value / candidate_value
        rows.append(row)
    return {
        "status": "success",
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
        "cases": rows,
    }


def _markdown(payload: dict[str, Any]) -> str:
    def speedup(row: dict[str, Any], metric: str) -> str:
        value = row.get(f"{metric}_speedup")
        return "n/a" if value is None else f"{float(value):.3f}x"

    lines = [
        "# Vortex Probe Comparison",
        "",
        f"- Baseline: `{payload['baseline']}`",
        f"- Candidate: `{payload['candidate']}`",
        "",
        "| Scenario | Prompt | Output | C | Output tok/s speedup | TTFT p50 speedup | TPOT mean speedup | E2E p50 speedup |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["cases"]:
        lines.append(
            f"| {row['scenario']} | {row['prompt_len']} | {row['output_len']} "
            f"| {row['concurrency']} | {speedup(row, 'output_token_throughput_tps')} "
            f"| {speedup(row, 'ttft_ms_p50')} "
            f"| {speedup(row, 'tpot_ms_mean')} "
            f"| {speedup(row, 'latency_ms_p50')} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = compare(args.baseline, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_markdown(payload), encoding="utf-8")
    args.output.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
