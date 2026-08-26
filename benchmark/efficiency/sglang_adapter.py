# SPDX-License-Identifier: Apache-2.0
"""SGLang native ``/generate`` streaming adapter for efficiency probes."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass
from typing import Any

import httpx

from benchmark.efficiency.workload import RequestTrace


@dataclass(frozen=True)
class RequestResult:
    request_index: int
    prompt_len: int
    requested_output_len: int
    generated_tokens: int
    prompt_digest: str
    started_at_s: float
    first_token_at_s: float
    last_token_at_s: float
    finished_at_s: float
    ttft_ms: float
    tpot_ms: float | None
    latency_ms: float
    status: str

    def metadata(self) -> dict[str, Any]:
        return asdict(self)


async def check_server(base_url: str, timeout_s: float = 10.0) -> None:
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        response = await client.get(f"{base_url.rstrip('/')}/health")
        response.raise_for_status()


async def _run_one(
    client: httpx.AsyncClient,
    generate_url: str,
    trace: RequestTrace,
) -> RequestResult:
    payload = {
        "input_ids": trace.prompt_token_ids,
        "sampling_params": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "ignore_eos": True,
            "max_new_tokens": trace.output_len,
        },
        "stream": True,
    }
    started = time.perf_counter()
    first_token_at: float | None = None
    last_token_at: float | None = None
    generated_tokens = 0
    last_payload: dict[str, Any] | None = None

    async with client.stream("POST", generate_url, json=payload) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            line = line.strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(f"SGLang request failed: {event['error']}")
            last_payload = event
            meta = event.get("meta_info") or {}
            completion_tokens = int(meta.get("completion_tokens", generated_tokens))
            if completion_tokens > generated_tokens:
                now = time.perf_counter()
                if first_token_at is None:
                    first_token_at = now
                last_token_at = now
                generated_tokens = completion_tokens

    finished = time.perf_counter()
    if last_payload is None or first_token_at is None or last_token_at is None:
        raise RuntimeError(
            f"SGLang returned no generated token for request {trace.request_index}."
        )
    if generated_tokens != trace.output_len:
        raise RuntimeError(
            f"Request {trace.request_index} generated {generated_tokens} tokens, "
            f"expected {trace.output_len}."
        )
    tpot_ms = (
        (last_token_at - first_token_at) * 1000.0 / (generated_tokens - 1)
        if generated_tokens > 1
        else None
    )
    return RequestResult(
        request_index=trace.request_index,
        prompt_len=trace.prompt_len,
        requested_output_len=trace.output_len,
        generated_tokens=generated_tokens,
        prompt_digest=trace.prompt_digest,
        started_at_s=started,
        first_token_at_s=first_token_at,
        last_token_at_s=last_token_at,
        finished_at_s=finished,
        ttft_ms=(first_token_at - started) * 1000.0,
        tpot_ms=tpot_ms,
        latency_ms=(finished - started) * 1000.0,
        status="success",
    )


async def run_trace(
    base_url: str,
    traces: list[RequestTrace],
    *,
    timeout_s: float,
    max_in_flight: int | None = None,
) -> tuple[list[RequestResult], float]:
    """Run a trace and return request metrics plus wall time.

    ``max_in_flight`` implements closed-loop churn: each completion is replaced
    until the trace is exhausted while keeping at most that many client requests
    active. With ``None``, all requests are submitted together.
    """
    if not traces:
        raise ValueError("run_trace requires at least one request.")
    if max_in_flight is not None and max_in_flight <= 0:
        raise ValueError(f"max_in_flight must be positive, got {max_in_flight}.")
    worker_count = min(len(traces), max_in_flight or len(traces))
    limits = httpx.Limits(
        max_connections=max(16, len(traces)),
        max_keepalive_connections=max(16, len(traces)),
    )
    timeout = httpx.Timeout(timeout_s, connect=min(timeout_s, 30.0))
    generate_url = f"{base_url.rstrip('/')}/generate"
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        queue: asyncio.Queue[RequestTrace] = asyncio.Queue()
        for trace in traces:
            queue.put_nowait(trace)
        results: list[RequestResult] = []

        async def worker() -> None:
            while True:
                try:
                    trace = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    results.append(await _run_one(client, generate_url, trace))
                finally:
                    queue.task_done()

        started = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(worker_count)))
        elapsed_s = time.perf_counter() - started
    return sorted(results, key=lambda result: result.request_index), elapsed_s
