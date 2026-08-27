"""CPU-only tests for the migrated efficiency-probe infrastructure."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from benchmark.efficiency.compare_runs import compare
from benchmark.efficiency.formal_report import (
    CONCURRENCIES,
    PROMPTS,
    SCENARIOS,
    STAGES,
    aggregate,
)
from benchmark.efficiency.runtime_env import prepend_interpreter_bin_to_path
from benchmark.efficiency.workload import (
    build_request_trace,
    derive_trace_seed,
    trace_metadata,
)


class RuntimeEnvironmentTest(unittest.TestCase):
    def test_interpreter_bin_is_prepended_to_path(self) -> None:
        with mock.patch.dict(os.environ, {"PATH": "/usr/bin:/bin"}):
            interpreter_bin = prepend_interpreter_bin_to_path()
            self.assertEqual(os.environ["PATH"].split(os.pathsep)[0], interpreter_bin)
            self.assertTrue(interpreter_bin.endswith("/.venv/bin"))


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


class FormalReportTest(unittest.TestCase):
    def _write_stage(self, root: Path, stage: str, multiplier: float) -> None:
        for scenario in SCENARIOS:
            for prompt in PROMPTS:
                shard = root / stage / f"{scenario}_p{prompt}"
                shard.mkdir(parents=True)
                label = stage.replace("_", "-")
                manifest = {
                    "status": "success",
                    "args": {
                        "model_path": "/data2/pretrain_models/GLM-4.7-Flash",
                        "backend_label": label,
                        "prompt_lens": [prompt],
                        "output_lens": [128],
                        "batch_sizes": list(CONCURRENCIES),
                        "scenario": scenario,
                        "num_warmups": 1,
                        "num_iters": 2,
                        "seed": 42,
                        "monitor_gpus": "6",
                    },
                    "git": {
                        "commit": "0" * 40,
                        "branch": "feat/sm90-support",
                        "dirty": False,
                    },
                    "physical_gpus": [
                        {
                            "physical_device_index": 6,
                            "compute_capability": "9.0",
                            "uuid": "GPU-test",
                        }
                    ],
                }
                (shard / "run_manifest.json").write_text(json.dumps(manifest))
                summaries = []
                requests = []
                for concurrency in CONCURRENCIES:
                    summaries.append(
                        {
                            "status": "success",
                            "scenario": "fixed_batch"
                            if scenario == "fixed"
                            else "continuous_churn",
                            "prompt_len": prompt,
                            "output_len": 128,
                            "concurrency": concurrency,
                            "iterations": 2,
                            **{
                                metric: multiplier
                                for metric in (
                                    "request_throughput_rps",
                                    "input_token_throughput_tps",
                                    "output_token_throughput_tps",
                                    "total_token_throughput_tps",
                                )
                            },
                            **{
                                metric: 1.0 / multiplier
                                for metric in (
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
                            },
                        }
                    )
                    count = concurrency if scenario == "fixed" else concurrency * 4
                    for iteration in (0, 1):
                        for request_index in range(count):
                            requests.append(
                                {
                                    "scenario": scenario,
                                    "nominal_prompt_len": prompt,
                                    "nominal_output_len": 128,
                                    "concurrency": concurrency,
                                    "iteration": iteration,
                                    "request_index": request_index,
                                    "prompt_len": prompt - request_index % 7,
                                    "requested_output_len": 128,
                                    "generated_tokens": 128,
                                    "prompt_digest": f"{scenario}-{prompt}-{concurrency}-{iteration}-{request_index}",
                                    "status": "success",
                                }
                            )
                (shard / "summary.json").write_text(
                    json.dumps({"status": "success", "summary": summaries})
                )
                (shard / "request_samples.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in requests)
                )

    def test_full_matrix_and_trace_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for stage, multiplier in zip(STAGES, (1.0, 2.0, 3.0)):
                self._write_stage(root, stage, multiplier)
            result = aggregate(root)
        self.assertTrue(result["trace_match"])
        self.assertEqual(result["matrix"]["cases_per_stage"], 30)
        self.assertAlmostEqual(
            result["aggregates"]["arch_only_vs_triton"][
                "output_token_throughput_tps_speedup_geomean"
            ],
            2.0,
        )
        self.assertAlmostEqual(
            result["aggregates"]["tuned_vs_arch_only"][
                "output_token_throughput_tps_speedup_geomean"
            ],
            1.5,
        )


if __name__ == "__main__":
    unittest.main()
