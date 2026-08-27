# Sparse-vLLM-matched Vortex efficiency probe

This probe follows the request and metric protocol in Sparse-vLLM
`benchmark/efficiency/bench_probe.py`.

## Protocol

- Fixed: submit `concurrency` requests as one burst. TTFT is measured once per
  iteration, from batch submission until every request has emitted its first
  token. Summary TTFT percentiles are over iteration-level batch TTFT samples.
- Churn: submit `concurrency * churn_request_multiplier` requests as one burst.
  The server admits at most `concurrency`; queued server time is included in
  each request's TTFT and latency. Summary latency percentiles are over all
  request-level samples.
- Fixed warmup contains `concurrency` requests. Churn warmup contains
  `min(total_requests, 2 * concurrency)` requests.
- Fixed and churn use Sparse-vLLM's exact scenario labels when deriving trace
  seeds, producing identical token IDs and per-request lengths.
- Prefix caching must remain disabled.

Fixed TPOT is reconstructed from client-observed token waves because the HTTP
API cannot expose SGLang's internal decode-step timer. The artifact records
this measurement boundary; internal profiler and CUDA graph counters are
reported as `skipped_by_policy` instead of being estimated.

## Server

The launcher retains the original Vortex probe defaults. Pass every
workload-specific Quest setting explicitly when running the
Sparse-vLLM-matched baseline:

```bash
PYTHONPATH="$PWD:$PWD/src" \
.venv/bin/python benchmark/efficiency/launch_server.py \
  --physical-gpu 7 \
  --attention-backend cuda_mla_sm90 \
  --model-path /data2/pretrain_models/GLM-4.7-Flash \
  --module-name quest_mla \
  --block-size 16 \
  --layers-skip 0,1 \
  --topk 291 \
  --max-topk 291 \
  --block-reserved-bos 0 \
  --block-reserved-eos 1 \
  --context-length 16640 \
  --chunked-prefill-size 8192 \
  --max-prefill-tokens 8192 \
  --mem-fraction-static 0.85 \
  --max-running-requests 32 \
  --cuda-graph-max-bs 32
```

The effective sparse budget is 292 pages: top-291 previous pages plus the
final page. Page size is 16, layers 0 and 1 remain dense, context length is
16640, and both prefill limits are 8192.

## Probe

Sparse-vLLM starts independent engines for fixed and churn. Match that lifecycle
by running one scenario, restarting the server, and then running the other:

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH="$PWD:$PWD/src" \
.venv/bin/python benchmark/efficiency/bench_probe.py \
  --server-url http://127.0.0.1:30000 \
  --model-path /data2/pretrain_models/GLM-4.7-Flash \
  --backend-label vortex-quest-cuda-mla-sm90 \
  --sparse-method quest \
  --prompt-lens 1024,4096,16384 \
  --output-lens 128 \
  --batch-sizes 32 \
  --scenario fixed \
  --num-warmups 1 \
  --num-iters 2 \
  --monitor-gpus 7 \
  --output-dir benchmark/results/glm47_quest_vortex_fixed
```

After restarting the server, change `--scenario fixed` to `--scenario churn`
and use a new output directory. `--scenario all` is rejected because a shared
server would not match Sparse-vLLM's engine lifecycle.
