"""Micro-benchmark: achieved KV-cache bandwidth of two FlashInfer decode APIs
in the **sparse decode** regime that vortex_torch targets.

Each request keeps a *long* full-context KV cache (sized via --total-seq-len)
in HBM — large enough that the working set overflows L2 and forces real HBM
traffic between timing iterations — but attends to only a sparse subset of
its pages (sized via --seq-len). The wrapper sees the selection as a flat
CSR (kv_indptr + kv_indices), trtllm sees it as a 2D block_tables; both
read exactly the same physical pages.

Bandwidth model (decode is dominated by KV reads):

    kv_bytes = batch * selected_seq_len * 2 * num_kv_heads * head_dim
               * sizeof(kv_dtype)

Achieved bandwidth = kv_bytes / kernel_time. Q in and O out are tracked
separately and are typically << kv_bytes.
"""

import argparse
import math

import torch
import triton
import triton.language as tl
import flashinfer


# ---------------------------------------------------------------------------
# CSR (kv_indptr + kv_indices + last_page_len) -> trtllm (block_tables + seq_lens)
# Copied verbatim from vortex_torch.engine.sgl.attention_backend.trtllm so this
# bench can be run standalone.
# ---------------------------------------------------------------------------
@triton.jit
def _csr_to_block_tables_kernel(
    kv_indptr_ptr,          # int32 [B+1]
    kv_indices_ptr,         # int32 [total_blocks]
    last_page_len_ptr,      # int32 [B]
    block_tables_ptr,       # int32 [B, MAX_BLOCKS_PER_SEQ]
    seq_lens_ptr,           # int32 [B]
    block_tables_stride,
    BLOCK_SIZE: tl.constexpr,
    MAX_BLOCKS_PER_SEQ: tl.constexpr,
    TILE: tl.constexpr,
):
    b = tl.program_id(0)
    start = tl.load(kv_indptr_ptr + b)
    end = tl.load(kv_indptr_ptr + b + 1)
    num_blocks = end - start
    last_len = tl.load(last_page_len_ptr + b)

    seq_len = (num_blocks - 1) * BLOCK_SIZE + last_len
    tl.store(seq_lens_ptr + b, seq_len)

    row = block_tables_ptr + b * block_tables_stride
    for tile_start in range(0, MAX_BLOCKS_PER_SEQ, TILE):
        offs = tile_start + tl.arange(0, TILE)
        valid = offs < num_blocks
        in_range = offs < MAX_BLOCKS_PER_SEQ
        v = tl.load(kv_indices_ptr + start + offs, mask=valid, other=0)
        tl.store(row + offs, v, mask=in_range)


def csr_to_block_tables(
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    last_page_len: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
) -> None:
    """Fill ``block_tables`` / ``seq_lens`` in place from the CSR-style page table."""
    B = kv_indptr.shape[0] - 1
    if B == 0:
        return
    max_blocks_per_seq = block_tables.shape[1]
    tile = max(triton.next_power_of_2(max_blocks_per_seq), 16)
    tile = min(tile, 1024)
    _csr_to_block_tables_kernel[(B,)](
        kv_indptr,
        kv_indices,
        last_page_len,
        block_tables,
        seq_lens,
        block_tables.stride(0),
        BLOCK_SIZE=block_size,
        MAX_BLOCKS_PER_SEQ=max_blocks_per_seq,
        TILE=tile,
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--total-seq-len", type=int, default=65536,
                   help="Full per-request KV context (tokens). Sizes the KV "
                        "pool so the working set overflows L2.")
    p.add_argument("--seq-len", type=int, default=4096,
                   help="Selected (attended) tokens per request. Must be "
                        "<= --total-seq-len.")
    p.add_argument("--num-qo-heads", type=int, default=4)
    p.add_argument("--num-kv-heads", type=int, default=1)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--page-size", type=int, default=16)
    p.add_argument("--kv-dtype", choices=["bf16", "fp16", "fp8_e4m3", "fp8_e5m2"],
                   default="bf16")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


_DT = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp8_e4m3": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
}


def build_inputs(args):
    assert args.seq_len <= args.total_seq_len, "selected seq_len > total seq_len"
    torch.manual_seed(0)
    device = args.device
    q_dtype = torch.bfloat16
    kv_dtype = _DT[args.kv_dtype]

    # Per-request pool of pages (full context).
    pool_pages_per_req = math.ceil(args.total_seq_len / args.page_size)

    # Per-request selected pages (sparse subset of the pool).
    sel_pages_per_req = math.ceil(args.seq_len / args.page_size)
    last_page_len = args.seq_len - (sel_pages_per_req - 1) * args.page_size

    pool_pages = args.batch_size * pool_pages_per_req

    # Random sparse selection: for each request, pick `sel_pages_per_req`
    # distinct pages from its own slice of the pool. Using its own slice
    # (rather than the global pool) mirrors how vortex_torch lays out KV.
    gen = torch.Generator(device=device).manual_seed(0)
    sel = torch.empty(
        (args.batch_size, sel_pages_per_req), dtype=torch.int32, device=device
    )
    for b in range(args.batch_size):
        perm = torch.randperm(pool_pages_per_req, generator=gen, device=device)
        sel[b] = (perm[:sel_pages_per_req] + b * pool_pages_per_req).to(torch.int32)

    # Wrapper view: flat CSR
    kv_indptr = torch.arange(
        0, (args.batch_size + 1) * sel_pages_per_req, sel_pages_per_req,
        dtype=torch.int32, device=device,
    )
    kv_indices = sel.reshape(-1).contiguous()
    kv_last_page_len = torch.full(
        (args.batch_size,), last_page_len, dtype=torch.int32, device=device
    )

    # trtllm view: 2D block_tables, scalar seq_lens
    block_tables = sel
    seq_lens = torch.full(
        (args.batch_size,), args.seq_len, dtype=torch.int32, device=device
    )

    # KV pool tensors — sized to the full context so the working set
    # overflows L2 between iterations.
    def _make_cache():
        t = torch.randn(
            pool_pages, args.page_size, args.num_kv_heads, args.head_dim,
            dtype=torch.bfloat16, device=device,
        )
        return t.to(kv_dtype)

    k_cache = _make_cache()
    v_cache = _make_cache()

    q = torch.randn(
        args.batch_size, args.num_qo_heads, args.head_dim,
        dtype=q_dtype, device=device,
    )

    return dict(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        kv_last_page_len=kv_last_page_len,
        block_tables=block_tables,
        seq_lens=seq_lens,
        sel_pages_per_req=sel_pages_per_req,
        pool_pages_per_req=pool_pages_per_req,
        pool_pages=pool_pages,
        kv_dtype=kv_dtype,
        q_dtype=q_dtype,
    )


def bytes_per_call(args, inputs):
    """Logical bytes read/written per decode call given the sparse selection."""
    kv_elem = inputs["k_cache"].element_size()
    q_elem = inputs["q"].element_size()
    selected_kv_tokens = args.batch_size * args.seq_len
    kv_bytes = 2 * selected_kv_tokens * args.num_kv_heads * args.head_dim * kv_elem
    qo_bytes = 2 * args.batch_size * args.num_qo_heads * args.head_dim * q_elem
    return kv_bytes, qo_bytes


def pool_bytes(args, inputs):
    """Resident size of the KV pool (the thing that overflows L2)."""
    kv_elem = inputs["k_cache"].element_size()
    return 2 * inputs["pool_pages"] * args.page_size * args.num_kv_heads * args.head_dim * kv_elem


def time_callable(fn, warmup, iters, device):
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)

    # Time across `iters` calls with a single start/stop pair so we amortise
    # cudaEvent overhead.
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    ms = start.elapsed_time(end) / iters
    return ms


def bench_wrapper(args, inputs):
    workspace = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device=args.device)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD", use_tensor_cores=True)
    wrapper.plan(
        inputs["kv_indptr"],
        inputs["kv_indices"],
        inputs["kv_last_page_len"],
        args.num_qo_heads,
        args.num_kv_heads,
        args.head_dim,
        args.page_size,
        pos_encoding_mode="NONE",
        q_data_type=inputs["q_dtype"],
        kv_data_type=inputs["kv_dtype"],
    )
    out = torch.empty_like(inputs["q"])

    def call():
        wrapper.run(inputs["q"], (inputs["k_cache"], inputs["v_cache"]), out=out)

    return time_callable(call, args.warmup, args.iters, args.device)


def bench_trtllm(args, inputs):
    workspace = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device=args.device)
    sm_scale = 1.0 / math.sqrt(args.head_dim)
    out = torch.empty_like(inputs["q"])

    def call():
        flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            query=inputs["q"],
            kv_cache=(inputs["k_cache"], inputs["v_cache"]),
            workspace_buffer=workspace,
            block_tables=inputs["block_tables"],
            seq_lens=inputs["seq_lens"],
            max_seq_len=args.seq_len,
            bmm1_scale=sm_scale,
            bmm2_scale=1.0,
            kv_layout="NHD",
            out=out,
        )

    return time_callable(call, args.warmup, args.iters, args.device)


def bench_trtllm_with_conv(args, inputs):
    """trtllm decode preceded by the CSR -> block_tables Triton kernel.

    Models what VortexTRTLLMBackend.forward_decode actually does on the sparse
    path: indexer rewrites kv_indices_decode[1], then csr_to_block_tables
    rebuilds block_tables[1] / seq_lens[1], then trtllm reads them.
    """
    workspace = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device=args.device)
    sm_scale = 1.0 / math.sqrt(args.head_dim)
    out = torch.empty_like(inputs["q"])

    # Scratch buffers overwritten by csr_to_block_tables every call.
    bt = torch.zeros_like(inputs["block_tables"])
    sl = torch.zeros_like(inputs["seq_lens"])

    def call():
        csr_to_block_tables(
            inputs["kv_indptr"],
            inputs["kv_indices"],
            inputs["kv_last_page_len"],
            bt,
            sl,
            args.page_size,
        )
        flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            query=inputs["q"],
            kv_cache=(inputs["k_cache"], inputs["v_cache"]),
            workspace_buffer=workspace,
            block_tables=bt,
            seq_lens=sl,
            max_seq_len=args.seq_len,
            bmm1_scale=sm_scale,
            bmm2_scale=1.0,
            kv_layout="NHD",
            out=out,
        )

    return time_callable(call, args.warmup, args.iters, args.device)


def bench_conv_only(args, inputs):
    """Just the CSR -> block_tables Triton kernel, no decode."""
    bt = torch.zeros_like(inputs["block_tables"])
    sl = torch.zeros_like(inputs["seq_lens"])

    def call():
        csr_to_block_tables(
            inputs["kv_indptr"],
            inputs["kv_indices"],
            inputs["kv_last_page_len"],
            bt,
            sl,
            args.page_size,
        )

    return time_callable(call, args.warmup, args.iters, args.device)


def fmt_bw(bytes_, ms):
    gb_s = bytes_ / (ms * 1e-3) / 1e9
    return f"{gb_s:8.1f} GB/s"


def main():
    args = parse_args()
    inputs = build_inputs(args)
    kv_bytes, qo_bytes = bytes_per_call(args, inputs)
    total_bytes = kv_bytes + qo_bytes

    pool_gb = pool_bytes(args, inputs) / 1e9
    sparsity = args.seq_len / args.total_seq_len

    print("config")
    print(f"  batch_size       : {args.batch_size}")
    print(f"  total_seq_len    : {args.total_seq_len}   ({inputs['pool_pages_per_req']} pool pages/req)")
    print(f"  selected_seq_len : {args.seq_len}   ({inputs['sel_pages_per_req']} pages/req)  ({sparsity*100:.2f}%)")
    print(f"  num_qo_heads     : {args.num_qo_heads}")
    print(f"  num_kv_heads     : {args.num_kv_heads}  (GQA group={args.num_qo_heads // args.num_kv_heads})")
    print(f"  head_dim         : {args.head_dim}")
    print(f"  page_size        : {args.page_size}")
    print(f"  q  dtype         : bf16")
    print(f"  kv dtype         : {args.kv_dtype}")
    print(f"  KV pool size     : {pool_gb:.2f} GB   (overflows L2)")
    print(f"  KV bytes/call    : {kv_bytes/1e9:.3f} GB   (only selected pages)")
    print(f"  QO bytes/call    : {qo_bytes/1e9:.3f} GB")
    print()

    ms_wrap = bench_wrapper(args, inputs)
    ms_trt = bench_trtllm(args, inputs)
    ms_trt_conv = bench_trtllm_with_conv(args, inputs)
    ms_conv = bench_conv_only(args, inputs)

    print(f"{'kernel':<18}  {'time/call':>10}  {'BW (KV only)':>14}  {'BW (KV+QO)':>14}")
    print(f"{'-'*18}  {'-'*10}  {'-'*14}  {'-'*14}")
    print(f"{'wrapper':<18}  {ms_wrap*1e3:8.1f} us   {fmt_bw(kv_bytes, ms_wrap)}   {fmt_bw(total_bytes, ms_wrap)}")
    print(f"{'trtllm':<18}  {ms_trt*1e3:8.1f} us   {fmt_bw(kv_bytes, ms_trt)}   {fmt_bw(total_bytes, ms_trt)}")
    print(f"{'trtllm + csr_conv':<18}  {ms_trt_conv*1e3:8.1f} us   {fmt_bw(kv_bytes, ms_trt_conv)}   {fmt_bw(total_bytes, ms_trt_conv)}")
    print(f"{'csr_conv only':<18}  {ms_conv*1e3:8.1f} us   {'-':>14}   {'-':>14}")
    print()
    print(f"speedup (wrapper / trtllm)            : {ms_wrap / ms_trt:.2f}x")
    print(f"speedup (wrapper / trtllm + csr_conv) : {ms_wrap / ms_trt_conv:.2f}x")
    print(f"csr_conv overhead vs trtllm           : {(ms_trt_conv - ms_trt) / ms_trt * 100:+.1f}%  "
          f"({(ms_trt_conv - ms_trt)*1e3:.1f} us / call)")


if __name__ == "__main__":
    main()
