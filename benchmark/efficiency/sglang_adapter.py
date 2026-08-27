# SPDX-License-Identifier: Apache-2.0
"""SGLang native ``/generate`` streaming adapter for efficiency probes."""

from __future__ import annotations

import asyncio
import json
import statistics
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
    token_at_s: tuple[float, ...]

    def metadata(self) -> dict[str, Any]:
        result = asdict(self)
        # Per-token client timestamps are retained only long enough to rebuild
        # Sparse-vLLM's fixed-batch decode-step metric.  They are deliberately
        # omitted from artifacts: request-level TTFT/TPOT/latency are the
        # stable public contract, and serializing every token timestamp makes
        # long probe artifacts needlessly large.
        result.pop("token_at_s")
        return result


@dataclass(frozen=True)
class TraceResult:
    """One submitted batch, timed at the same boundaries as Sparse-vLLM.

    Every request is submitted at once.  ``batch_ttft_ms`` is therefore the
    time from immediately before submission until *all* requests have emitted
    their first token.  This matches Sparse-vLLM's fixed-batch TTFT instead of
    aggregating per-request HTTP TTFT samples.
    """

    request_results: tuple[RequestResult, ...]
    started_at_s: float
    finished_at_s: float

    @property
    def elapsed_s(self) -> float:
        return self.finished_at_s - self.started_at_s

    @property
    def batch_ttft_ms(self) -> float:
        return (
            max(result.first_token_at_s for result in self.request_results)
            - self.started_at_s
        ) * 1000.0

    @property
    def batch_tpot_ms(self) -> float | None:
        """Mean fixed-batch token-wave interval observed by the client.

        Sparse-vLLM measures the mean duration of synchronous decode steps.
        An HTTP client cannot read SGLang's internal step timer, so the matched
        observable is the interval between successive token waves after every
        request in the fixed batch has reached that token ordinal.
        """

        token_counts = {len(result.token_at_s) for result in self.request_results}
        if len(token_counts) != 1:
            raise ValueError(
                "Fixed-batch TPOT requires equal generated token counts, got "
                f"{sorted(token_counts)}."
            )
        token_count = token_counts.pop()
        if token_count <= 1:
            return None
        wave_times = [
            max(result.token_at_s[token_index] for result in self.request_results)
            for token_index in range(token_count)
        ]
        return statistics.fmean(
            (later - earlier) * 1000.0
            for earlier, later in zip(wave_times, wave_times[1:])
        )


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
    token_at: list[float] = []
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
                token_at.extend([now] * (completion_tokens - generated_tokens))
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
        token_at_s=tuple(token_at),
    )


async def run_trace(
    base_url: str,
    traces: list[RequestTrace],
    *,
    timeout_s: float,
) -> TraceResult:
    """Submit the complete trace as one burst and return matched timing data.

    Sparse-vLLM registers every request in an oversubscribed churn trace before
    stepping the engine.  Starting every HTTP request here gives every request
    an arrival timestamp before SGLang's internal queue, so server queueing is
    included in request TTFT and latency.
    """
    if not traces:
        raise ValueError("run_trace requires at least one request.")
    limits = httpx.Limits(
        max_connections=max(16, len(traces)),
        max_keepalive_connections=max(16, len(traces)),
    )
    timeout = httpx.Timeout(timeout_s, connect=min(timeout_s, 30.0))
    generate_url = f"{base_url.rstrip('/')}/generate"
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        started = time.perf_counter()
        results = await asyncio.gather(
            *(_run_one(client, generate_url, trace) for trace in traces)
        )
        finished = time.perf_counter()
    ordered = tuple(sorted(results, key=lambda result: result.request_index))
    if len(ordered) != len(traces):
        raise RuntimeError(
            f"SGLang returned {len(ordered)} results for {len(traces)} requests."
        )
    return TraceResult(
        request_results=ordered,
        started_at_s=started,
        finished_at_s=finished,
    )
