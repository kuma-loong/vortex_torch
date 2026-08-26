"""Select or validate an idle physical GPU from an explicit allow-list."""

from __future__ import annotations

import argparse
import csv
import io
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class GPUState:
    index: int
    uuid: str
    utilization_pct: float
    memory_used_mib: float
    compute_processes: int

    @property
    def idle(self) -> bool:
        return (
            self.compute_processes == 0
            and self.memory_used_mib <= 16.0
            and self.utilization_pct <= 5.0
        )


def _query_csv(arguments: list[str]) -> list[list[str]]:
    result = subprocess.run(
        ["nvidia-smi", *arguments, "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [
        [field.strip() for field in row]
        for row in csv.reader(io.StringIO(result.stdout))
        if row
    ]


def query_gpu_states(allowed: list[int]) -> list[GPUState]:
    if not allowed or len(allowed) != len(set(allowed)):
        raise ValueError(f"Allowed GPU IDs must be non-empty and unique: {allowed}")
    gpu_rows = _query_csv(
        [
            "-i",
            ",".join(map(str, allowed)),
            "--query-gpu=index,uuid,utilization.gpu,memory.used",
        ]
    )
    process_rows = _query_csv(["--query-compute-apps=gpu_uuid,pid"])
    process_counts: dict[str, int] = {}
    for row in process_rows:
        if len(row) >= 2:
            process_counts[row[0]] = process_counts.get(row[0], 0) + 1

    states = []
    for row in gpu_rows:
        if len(row) != 4:
            raise RuntimeError(f"Unexpected GPU state row: {row}")
        states.append(
            GPUState(
                index=int(row[0]),
                uuid=row[1],
                utilization_pct=float(row[2]),
                memory_used_mib=float(row[3]),
                compute_processes=process_counts.get(row[1], 0),
            )
        )
    if {state.index for state in states} != set(allowed):
        raise RuntimeError(
            f"nvidia-smi did not return the full allow-list: expected={allowed}, "
            f"actual={[state.index for state in states]}"
        )
    return sorted(states, key=lambda state: state.index)


def select_idle_gpu(allowed: list[int]) -> GPUState:
    states = query_gpu_states(allowed)
    for state in states:
        if state.idle:
            return state
    detail = "; ".join(
        f"gpu{state.index}: util={state.utilization_pct:.0f}%, "
        f"memory={state.memory_used_mib:.0f}MiB, processes={state.compute_processes}"
        for state in states
    )
    raise RuntimeError(f"No idle GPU in allow-list {allowed}. {detail}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allowed", default="4,5,6,7")
    parser.add_argument("--require", type=int, default=None)
    args = parser.parse_args()
    allowed = [int(item.strip()) for item in args.allowed.split(",") if item.strip()]
    if args.require is not None:
        if args.require not in allowed:
            raise SystemExit(f"GPU {args.require} is not in allow-list {allowed}.")
        state = next(state for state in query_gpu_states(allowed) if state.index == args.require)
        if not state.idle:
            raise SystemExit(
                f"GPU {state.index} is not idle: util={state.utilization_pct:.0f}%, "
                f"memory={state.memory_used_mib:.0f}MiB, processes={state.compute_processes}."
            )
    else:
        state = select_idle_gpu(allowed)
    print(state.index)


if __name__ == "__main__":
    main()
