"""CPU-only tests for the migrated efficiency-probe infrastructure."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmark.efficiency.compare_runs import compare
from benchmark.efficiency.workload import (
    build_request_trace,
    derive_trace_seed,
    trace_metadata,
)


class WorkloadTest(unittest.TestCase):
    def test_trace_is_reproducible_and_unique(self) -> None:
        seed = derive_trace_seed(
            42,
            scenario="fixed",
            phase="measure",
            nominal_prompt_len=64,
            nominal_output_len=8,
            concurrency=8,
            iteration=0,
        )
        kwargs = {
            "seed": seed,
            "request_count": 8,
            "nominal_prompt_len": 64,
            "nominal_output_len": 8,
            "vocab_size": 1024,
            "prompt_jitter_fraction": 0.10,
            "output_jitter_fraction": 0.25,
            "vary_output_lengths": False,
        }
        first = build_request_trace(**kwargs)
        second = build_request_trace(**kwargs)
        self.assertEqual(first, second)
        metadata = trace_metadata(first)
        self.assertEqual(metadata["request_count"], 8)
        self.assertEqual(len(set(metadata["prompt_digests"])), 8)
        self.assertGreater(metadata["unique_prompt_lengths"], 1)
        self.assertEqual(metadata["unique_output_lengths"], 1)


class CompareRunsTest(unittest.TestCase):
    def test_matched_case_speedups(self) -> None:
        row = {
            "scenario": "fixed_batch",
            "prompt_len": 4096,
            "output_len": 64,
            "concurrency": 1,
            "request_throughput_rps": 1.0,
            "input_token_throughput_tps": 2.0,
            "output_token_throughput_tps": 3.0,
            "total_token_throughput_tps": 4.0,
            "ttft_ms_p50": 10.0,
            "ttft_ms_p99": 20.0,
            "tpot_ms_p50": 2.0,
            "tpot_ms_p99": 3.0,
            "latency_ms_p50": 100.0,
            "latency_ms_p99": 120.0,
        }
        candidate = dict(row)
        for metric in (
            "request_throughput_rps",
            "input_token_throughput_tps",
            "output_token_throughput_tps",
            "total_token_throughput_tps",
        ):
            candidate[metric] *= 2.0
        for metric in (
            "ttft_ms_p50",
            "ttft_ms_p99",
            "tpot_ms_p50",
            "tpot_ms_p99",
            "latency_ms_p50",
            "latency_ms_p99",
        ):
            candidate[metric] /= 2.0

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline_path = root / "baseline.json"
            candidate_path = root / "candidate.json"
            baseline_path.write_text(
                json.dumps({"status": "success", "summary": [row]}),
                encoding="utf-8",
            )
            candidate_path.write_text(
                json.dumps({"status": "success", "summary": [candidate]}),
                encoding="utf-8",
            )
            result = compare(baseline_path, candidate_path)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["cases"][0]["output_token_throughput_tps_speedup"], 2.0)
        self.assertEqual(result["cases"][0]["ttft_ms_p50_speedup"], 2.0)


if __name__ == "__main__":
    unittest.main()
