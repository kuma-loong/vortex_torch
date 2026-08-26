# SPDX-License-Identifier: Apache-2.0
"""Coarse nvidia-smi GPU activity timeline.

Adapted from Sparse-vLLM commit
6f7b8474c1c5ad4d3eaebe62c51e537a527917a8.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


class GPUHardwareMonitor:
    def __init__(
        self,
        gpus: list[int],
        interval_ms: int = 100,
        output_file: str | Path | None = None,
    ):
        if not gpus or len(gpus) != len(set(gpus)):
            raise ValueError(f"GPU IDs must be non-empty and unique, got {gpus}.")
        self.gpus = sorted(gpus)
        self.interval_ms = max(50, int(interval_ms))
        self.interval_s = self.interval_ms / 1000.0
        self.output_file = Path(output_file) if output_file else None
        self.samples: list[dict[str, Any]] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.start_time = 0.0
        self.end_time = 0.0
        self.collection_errors: list[str] = []

    def start(self) -> None:
        self.samples.clear()
        self.collection_errors.clear()
        self._stop_event.clear()
        self.start_time = time.time()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self.end_time = time.time()
        summary = self._analyze()
        self._save(summary)
        return summary

    def _monitor_loop(self) -> None:
        cmd = [
            "nvidia-smi",
            "-i",
            ",".join(map(str, self.gpus)),
            "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop_event.is_set():
            row: dict[str, Any] = {"time_s": round(time.time() - self.start_time, 3)}
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                for line in result.stdout.splitlines():
                    fields = [field.strip() for field in line.split(",")]
                    if len(fields) != 6:
                        raise RuntimeError(f"Unexpected nvidia-smi sample: {line!r}")
                    idx = int(fields[0])
                    row[f"gpu{idx}_util"] = float(fields[1])
                    row[f"gpu{idx}_mem_util"] = float(fields[2])
                    row[f"gpu{idx}_mem_mb"] = float(fields[3])
                    row[f"gpu{idx}_power_w"] = float(fields[4])
                    row[f"gpu{idx}_temp_c"] = float(fields[5])
                observed = {idx for idx in self.gpus if f"gpu{idx}_util" in row}
                if observed != set(self.gpus):
                    raise RuntimeError(
                        f"Incomplete GPU sample: expected={self.gpus}, got={sorted(observed)}"
                    )
                self.samples.append(row)
            except Exception as exc:  # Preserve the benchmark and fail metrics explicitly.
                self.collection_errors.append(repr(exc))
            self._stop_event.wait(self.interval_s)

    def _analyze(self) -> dict[str, Any]:
        duration = max(0.001, (self.end_time or time.time()) - self.start_time)
        status = "success" if self.samples and not self.collection_errors else "metric_failed"
        summary: dict[str, Any] = {
            "status": status,
            "duration_seconds": round(duration, 3),
            "total_samples": len(self.samples),
            "sampling_interval_ms": self.interval_ms,
            "collection_error_count": len(self.collection_errors),
            "last_collection_error": self.collection_errors[-1] if self.collection_errors else None,
            "gpus": {},
            "aggregate": {},
        }
        if not self.samples:
            summary["error"] = "No complete nvidia-smi samples were collected."
            return summary

        for idx in self.gpus:
            utils = [sample[f"gpu{idx}_util"] for sample in self.samples]
            mem_utils = [sample[f"gpu{idx}_mem_util"] for sample in self.samples]
            mems = [sample[f"gpu{idx}_mem_mb"] for sample in self.samples]
            powers = [sample[f"gpu{idx}_power_w"] for sample in self.samples]
            active = sum(value > 10.0 for value in utils) / len(utils) * 100.0
            summary["gpus"][f"gpu_{idx}"] = {
                "avg_compute_util_pct": round(sum(utils) / len(utils), 2),
                "max_compute_util_pct": round(max(utils), 2),
                "avg_memory_io_activity_pct": round(sum(mem_utils) / len(mem_utils), 2),
                "avg_power_w": round(sum(powers) / len(powers), 2),
                "max_power_w": round(max(powers), 2),
                "energy_joules": round(sum(powers) / len(powers) * duration, 2),
                "peak_vram_gb": round(max(mems) / 1024.0, 2),
                "coarse_gpu_active_duty_pct": round(active, 2),
                "coarse_gpu_idle_duty_pct": round(100.0 - active, 2),
            }

        gpu_rows = list(summary["gpus"].values())
        summary["aggregate"] = {
            "num_gpus": len(gpu_rows),
            "mean_compute_util_pct": round(
                sum(row["avg_compute_util_pct"] for row in gpu_rows) / len(gpu_rows), 2
            ),
            "mean_memory_io_activity_pct": round(
                sum(row["avg_memory_io_activity_pct"] for row in gpu_rows) / len(gpu_rows), 2
            ),
            "avg_total_power_w": round(sum(row["avg_power_w"] for row in gpu_rows), 2),
            "total_energy_joules": round(sum(row["energy_joules"] for row in gpu_rows), 2),
            "mean_coarse_gpu_active_duty_pct": round(
                sum(row["coarse_gpu_active_duty_pct"] for row in gpu_rows) / len(gpu_rows), 2
            ),
        }
        return summary

    def _save(self, summary: dict[str, Any]) -> None:
        if self.output_file is None:
            return
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.output_file.write_text(
            json.dumps({"summary": summary, "timeline": self.samples}, indent=2),
            encoding="utf-8",
        )
