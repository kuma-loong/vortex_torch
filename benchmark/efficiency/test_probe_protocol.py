# SPDX-License-Identifier: Apache-2.0
"""Offline regression tests for the Sparse-vLLM-matched probe protocol."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from benchmark.efficiency.bench_probe import (
    _request_count,
    _summarize_case,
    _trace,
    _warmup_request_count,
)
from benchmark.efficiency.sglang_adapter import RequestResult, TraceResult


def _request(
    index: int,
    *,
    started: float,
    tokens: tuple[float, ...],
) -> RequestResult:
    first = tokens[0]
    last = tokens[-1]
    return RequestResult(
        request_index=index,
        prompt_len=16,
        requested_output_len=len(tokens),
        generated_tokens=len(tokens),
        prompt_digest=f"digest-{index}",
        started_at_s=started,
        first_token_at_s=first,
        last_token_at_s=last,
        finished_at_s=last + 0.01,
        ttft_ms=(first - started) * 1000.0,
        tpot_ms=(last - first) * 1000.0 / (len(tokens) - 1),
        latency_ms=(last + 0.01 - started) * 1000.0,
        status="success",
        token_at_s=tokens,
    )


class ProbeProtocolTest(unittest.TestCase):
    def test_trace_matches_sparse_vllm_fixed_seed(self) -> None:
        args = SimpleNamespace(
            seed=42,
            vocab_size=154880,
            prompt_length_jitter=0.10,
            output_length_jitter=0.25,
        )
        trace = _trace(
            args,
            scenario="fixed",
            phase="measure",
            prompt_len=1024,
            output_len=128,
            concurrency=32,
            iteration=0,
            request_count=32,
        )
        self.assertEqual([row.prompt_len for row in trace[:4]], [932, 1020, 971, 931])
        self.assertEqual(
            trace[0].prompt_digest,
            "8ad69ad298cbe47bc9b12bea1c682224522e5b971454f85d3506a2784400e212",
        )
        self.assertEqual({row.output_len for row in trace}, {128})

    def test_trace_matches_sparse_vllm_churn_seed(self) -> None:
        args = SimpleNamespace(
            seed=42,
            vocab_size=154880,
            prompt_length_jitter=0.10,
            output_length_jitter=0.25,
        )
        trace = _trace(
            args,
            scenario="churn",
            phase="measure",
            prompt_len=1024,
            output_len=128,
            concurrency=32,
            iteration=0,
            request_count=128,
        )
        self.assertEqual([row.prompt_len for row in trace[:4]], [1022, 975, 958, 948])
        self.assertEqual([row.output_len for row in trace[:4]], [111, 103, 113, 121])
        self.assertEqual(
            trace[0].prompt_digest,
            "cdd1a9fc345f138ab5826b055052629ce26176393f4e039dd6cc81fe9e88ad89",
        )

    def test_sparse_vllm_request_counts(self) -> None:
        self.assertEqual(_request_count("fixed", 32, 4), 32)
        self.assertEqual(_request_count("churn", 32, 4), 128)
        self.assertEqual(_warmup_request_count("fixed", 32, 4), 32)
        self.assertEqual(_warmup_request_count("churn", 32, 4), 64)

    def test_fixed_batch_timing_uses_all_request_token_waves(self) -> None:
        result = TraceResult(
            request_results=(
                _request(0, started=1.01, tokens=(1.20, 1.30, 1.40)),
                _request(1, started=1.02, tokens=(1.25, 1.35, 1.45)),
            ),
            started_at_s=1.0,
            finished_at_s=1.5,
        )
        self.assertAlmostEqual(result.batch_ttft_ms, 250.0)
        self.assertAlmostEqual(result.batch_tpot_ms or 0.0, 100.0)
        self.assertNotIn("token_at_s", result.request_results[0].metadata())

    def test_fixed_summary_aggregates_iteration_level_metrics(self) -> None:
        args = SimpleNamespace(backend_label="vortex-quest", sparse_method="quest")
        trace = {"prompt_lengths": [16, 15], "output_lengths": [3, 3]}
        records = [
            {
                "request_count": 2,
                "ttft_ms": 100.0,
                "tpot_ms": 10.0,
                "request_throughput_rps": 2.0,
                "input_token_throughput_tps": 31.0,
                "output_token_throughput_tps": 6.0,
                "total_token_throughput_tps": 37.0,
                "trace": trace,
            },
            {
                "request_count": 2,
                "ttft_ms": 200.0,
                "tpot_ms": 20.0,
                "request_throughput_rps": 4.0,
                "input_token_throughput_tps": 62.0,
                "output_token_throughput_tps": 12.0,
                "total_token_throughput_tps": 74.0,
                "trace": trace,
            },
        ]
        hardware = {
            "metric_source": "test",
            "sample_count": 1,
            "sampling_interval_ms": 100,
            "gpu_compute_activity_pct_mean": 1.0,
            "gpu_memory_io_activity_pct_mean": 2.0,
            "gpu_active_duty_pct_mean": 3.0,
            "gpu_power_w_mean_total": 4.0,
            "peak_vram_gb_max": 5.0,
            "per_gpu": {},
        }
        summary = _summarize_case(
            args,
            scenario="fixed",
            prompt_len=16,
            output_len=3,
            concurrency=2,
            iteration_records=records,
            hardware=hardware,
        )
        self.assertEqual(summary["scenario"], "fixed_batch")
        self.assertEqual(summary["ttft_ms_mean"], 150.0)
        self.assertEqual(summary["ttft_ms_p50"], 150.0)
        self.assertEqual(summary["ttft_ms_p99"], 199.0)
        self.assertEqual(summary["tpot_ms_mean"], 15.0)
        self.assertNotIn("latency_ms_p50", summary)

    def test_churn_summary_aggregates_request_level_queue_metrics(self) -> None:
        args = SimpleNamespace(backend_label="vortex-quest", sparse_method="quest")
        request_results = [
            {"ttft_ms": 100.0, "latency_ms": 500.0, "tpot_ms": 200.0},
            {"ttft_ms": 300.0, "latency_ms": 900.0, "tpot_ms": 300.0},
        ]
        records = [
            {
                "request_count": 2,
                "request_throughput_rps": 1.0,
                "input_token_throughput_tps": 31.0,
                "output_token_throughput_tps": 6.0,
                "total_token_throughput_tps": 37.0,
                "trace": {
                    "prompt_lengths": [16, 15],
                    "output_lengths": [3, 2],
                },
                "request_results": request_results,
            }
        ]
        hardware = {
            "metric_source": "test",
            "sample_count": 1,
            "sampling_interval_ms": 100,
            "gpu_compute_activity_pct_mean": 1.0,
            "gpu_memory_io_activity_pct_mean": 2.0,
            "gpu_active_duty_pct_mean": 3.0,
            "gpu_power_w_mean_total": 4.0,
            "peak_vram_gb_max": 5.0,
            "per_gpu": {},
        }
        summary = _summarize_case(
            args,
            scenario="churn",
            prompt_len=16,
            output_len=3,
            concurrency=1,
            iteration_records=records,
            hardware=hardware,
        )
        self.assertEqual(summary["scenario"], "oversubscribed_churn")
        self.assertEqual(summary["sequence_replacements"], 1)
        self.assertEqual(summary["ttft_ms_mean"], 200.0)
        self.assertEqual(summary["ttft_ms_p50"], 200.0)
        self.assertEqual(summary["ttft_ms_p99"], 298.0)
        self.assertEqual(summary["latency_ms_p50"], 700.0)
        self.assertEqual(summary["tpot_ms_mean"], 250.0)


if __name__ == "__main__":
    unittest.main()
