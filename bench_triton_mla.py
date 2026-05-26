"""Microbenchmark for the vortex block-table MLA decode kernel(s).

Pure kernel efficiency — NO sglang / RULER. Builds synthetic decode inputs
(q, fused latent pool, sparse block_table, seqlens), times each kernel variant
with CUDA events, and reports achieved HBM bandwidth.

Bandwidth model (decode is KV-read bound): the kernel must read, per request,
its `selected_tokens` fused-latent rows (kv_lora_rank + qk_rope = 576 bf16).
Pages are distinct + scattered across requests (no L2 reuse), so

    KV_bytes = bs * selected_tokens * 576 * dtype_size
    achieved_BW = KV_bytes / kernel_time

is the meaningful "useful" HBM bandwidth (a perfect kernel reads each latent row
once per request). B200 HBM3e peak ~= 8 TB/s.

    export HF_HOME=/raid/catalyst/models/
    CUDA_VISIBLE_DEVICES=<gpu> python marks/mla/bench_triton_mla.py
"""
import argparse
import json
import torch

from vortex_torch.engine.sgl.attention_backend.triton_mla_kernel import (
    KERNELS,  # {name: fn(q, latent, block_table, seqlens, sm_scale, block_size, kv_lora_rank, o)}
)

KV_DIM = 576
KV_LORA = 512
PEAK_BW_GBPS = 8000.0  # B200 HBM3e ~8 TB/s


def make_inputs(bs, num_heads, block_size, selected_tokens, device, dtype):
    assert selected_tokens % block_size == 0
    n_blocks = selected_tokens // block_size
    num_pages = bs * n_blocks                      # distinct page per (req, slot)
    latent = torch.randn(num_pages * block_size, KV_DIM, device=device, dtype=dtype)
    pages = torch.randperm(num_pages, device=device, dtype=torch.int32)
    block_table = pages.view(bs, n_blocks).contiguous()
    seqlens = torch.full((bs,), selected_tokens, device=device, dtype=torch.int32)
    q = torch.randn(bs, num_heads, KV_DIM, device=device, dtype=dtype)
    o = torch.empty(bs, num_heads, KV_LORA, device=device, dtype=dtype)
    return q, latent, block_table, seqlens, o


def torch_reference(q, latent, block_table, seqlens, sm_scale, block_size):
    """Dense block-sparse MLA attention reference (fp32) for correctness."""
    bs, H, _ = q.shape
    out = torch.empty(bs, H, KV_LORA, device=q.device, dtype=torch.float32)
    qf = q.float()
    lf = latent.float()
    for b in range(bs):
        sl = int(seqlens[b])
        nb = (sl + block_size - 1) // block_size
        rows = []
        for j in range(nb):
            page = int(block_table[b, j])
            rows.append(torch.arange(page * block_size, page * block_size + block_size,
                                     device=q.device))
        slots = torch.cat(rows)[:sl]
        k = lf[slots]                       # [sl, 576]
        scores = (qf[b] @ k.t()) * sm_scale  # [H, sl]
        p = torch.softmax(scores, dim=-1)
        out[b] = p @ k[:, :KV_LORA]          # [H, 512]
    return out


def bench_one(fn, q, latent, block_table, seqlens, o, block_size, iters=50, warmup=20):
    sm_scale = 1.0 / (KV_DIM ** 0.5)
    for _ in range(warmup):
        fn(q, latent, block_table, seqlens, sm_scale, block_size, KV_LORA, o)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(q, latent, block_table, seqlens, sm_scale, block_size, KV_LORA, o)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--kernels", default="all")
    ap.add_argument("--out", default="marks/mla/bench_results.jsonl")
    args = ap.parse_args()

    device = "cuda"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    names = list(KERNELS) if args.kernels == "all" else args.kernels.split(",")

    blocks = [16, 32, 64]
    token_budgets = [512, 1024, 2048, 4096]
    sm_scale = 1.0 / (KV_DIM ** 0.5)

    print(f"B200 MLA decode kernel bench | bs={args.bs} heads={args.heads} "
          f"dtype={args.dtype} | peak~{PEAK_BW_GBPS/1000:.0f}TB/s")
    print(f"{'kernel':<14}{'blk':>4}{'sel_tok':>8}{'ms':>9}{'GB/s':>9}{'%peak':>7}{'maxerr':>10}")

    results = []
    # correctness on a small case once per kernel
    for name in names:
        fn = KERNELS[name]
        # --- correctness (small) ---
        q, latent, bt, sl, o = make_inputs(4, args.heads, 32, 256, device, dtype)
        ref = torch_reference(q, latent, bt, sl, sm_scale, 32)
        out = fn(q, latent, bt, sl, sm_scale, 32, KV_LORA, o)
        maxerr = (out.float() - ref).abs().max().item()
        # --- sweep ---
        for blk in blocks:
            for tok in token_budgets:
                q, latent, bt, sl, o = make_inputs(args.bs, args.heads, blk, tok, device, dtype)
                ms = bench_one(fn, q, latent, bt, sl, o, blk)
                kv_bytes = args.bs * tok * KV_DIM * (2 if dtype != torch.float32 else 4)
                gbps = kv_bytes / (ms * 1e-3) / 1e9
                pct = gbps / PEAK_BW_GBPS * 100
                print(f"{name:<14}{blk:>4}{tok:>8}{ms:>9.3f}{gbps:>9.0f}{pct:>6.1f}%"
                      f"{maxerr:>10.2e}")
                results.append({"kernel": name, "bs": args.bs, "heads": args.heads,
                                "block": blk, "sel_tok": tok, "ms": round(ms, 4),
                                "gbps": round(gbps, 1), "pct_peak": round(pct, 1),
                                "maxerr": maxerr})
    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"[bench] wrote {len(results)} rows to {args.out}")


if __name__ == "__main__":
    main()

