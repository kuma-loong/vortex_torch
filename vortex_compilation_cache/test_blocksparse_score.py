"""
Correctness test: compare Triton vs CUDA blocksparse score kernel.
"""
import torch
import types

# ── Fake ctx with synthetic workload metadata ──────────────────────────
def make_test_ctx(batch_size, num_centroids, items_per_workload, device="cuda:0"):
    """Build a minimal ctx object for testing."""
    n_workloads = batch_size
    total_items = n_workloads * items_per_workload

    # dense_kv_indices: for each workload, pages of 4 consecutive centroid indices
    # We'll tile indices [0, 4, 8, ...] for simplicity
    # Last page's centroid_row = base + (items_per_workload-1 & ~3) + 3
    # must stay strictly below num_centroids.
    max_last_row = ((items_per_workload - 1) & ~3) + 3
    assert max_last_row + 1 <= num_centroids, "num_centroids too small for test"
    max_base = num_centroids - 1 - max_last_row

    all_indices = []
    for w in range(n_workloads):
        # each workload indexes into a different slice of centroids, bounded
        base = (w * 4) % (max_base + 1)
        for j in range(items_per_workload):
            all_indices.append(base + j - (j % 4))  # page-aligned index
        # pad to multiple of 4
        pad = (4 - items_per_workload % 4) % 4
        all_indices.extend([0] * pad)

    padded_per = items_per_workload + (4 - items_per_workload % 4) % 4

    dense_kv_indices = torch.tensor(all_indices, dtype=torch.int32, device=device)
    winfo_q_indices = torch.arange(n_workloads, dtype=torch.int32, device=device)
    winfo_kv_offsets = (torch.arange(n_workloads, dtype=torch.int32, device=device) * padded_per)
    winfo_kv_lens = torch.full((n_workloads,), items_per_workload, dtype=torch.int32, device=device)
    winfo_num_workloads = torch.tensor([n_workloads], dtype=torch.int32, device=device)

    ctx = types.SimpleNamespace(
        dense_kv_indices=dense_kv_indices,
        winfo_q_indices=winfo_q_indices,
        winfo_kv_offsets=winfo_kv_offsets,
        winfo_kv_lens=winfo_kv_lens,
        winfo_num_workloads=winfo_num_workloads,
    )
    return ctx, padded_per


def test_correctness():
    torch.manual_seed(0)
    device = "cuda:0"
    batch_size = 32
    num_centroids = 512
    items_per_workload = 12  # not a multiple of 4 on purpose

    ctx, padded_per = make_test_ctx(batch_size, num_centroids, items_per_workload, device)
    total_slots = batch_size * padded_per

    query = torch.randn(batch_size, 2, 128, dtype=torch.bfloat16, device=device)
    centroids = torch.randn(num_centroids, 128, dtype=torch.bfloat16, device=device)

    out_triton = torch.zeros(total_slots, 1, 1, dtype=torch.bfloat16, device=device)
    out_cuda   = torch.zeros(total_slots, 1, 1, dtype=torch.bfloat16, device=device)

    # --- Triton kernel ---
    from blocksparseattention_9fbd0903_compiled_func import (
        blocksparseattention_9fbd0903_subgraph_0_impl,
    )
    blocksparseattention_9fbd0903_subgraph_0_impl(query, centroids, out_triton, ctx)
    torch.cuda.synchronize()

    # --- CUDA kernel ---
    from blocksparse_score_kernel import blocksparse_score_impl
    blocksparse_score_impl(query, centroids, out_cuda, ctx)
    torch.cuda.synchronize()

    # --- Compare ---
    # Extract only valid items
    valid_mask = torch.zeros(total_slots, dtype=torch.bool, device=device)
    for w in range(batch_size):
        offset = ctx.winfo_kv_offsets[w].item()
        length = ctx.winfo_kv_lens[w].item()
        valid_mask[offset : offset + length] = True

    t = out_triton.view(-1)[valid_mask].float()
    c = out_cuda.view(-1)[valid_mask].float()

    abs_err = (t - c).abs()
    rel_err = abs_err / t.abs().clamp(min=1e-3)

    print(f"Valid items:    {valid_mask.sum().item()}")
    print(f"Max  abs error: {abs_err.max().item():.6e}")
    print(f"Mean abs error: {abs_err.mean().item():.6e}")
    print(f"Max  rel error: {rel_err.max().item():.6e}")
    print(f"Mean rel error: {rel_err.mean().item():.6e}")

    # Hybrid tolerance: atol (≈2-3 bf16 ULP at ~magnitude 16) OR rtol 2%
    ok = ((abs_err < 0.30) | (rel_err < 0.02)).all().item()
    if ok:
        print("PASS ✓  (within bf16 quantisation tolerance)")
    else:
        print("FAIL ✗")
        diffs = abs_err
        top_idx = diffs.topk(min(5, len(diffs))).indices
        for idx in top_idx:
            print(f"  idx={idx.item()}: triton={t[idx].item():.6f}  cuda={c[idx].item():.6f}")


if __name__ == "__main__":
    print("── items_per_workload = 12 (3 pages) ───────────────────────────")
    test_correctness()
    print()

    # Stress: many pages + large batch
    def test_with_params(batch_size, items_per_workload, num_centroids=2048):
        print(f"── batch={batch_size}, items={items_per_workload}, "
              f"centroids={num_centroids} ──────────")
        import test_blocksparse_score as mod
        ctx, padded_per = mod.make_test_ctx(
            batch_size, num_centroids, items_per_workload, "cuda:0")
        total = batch_size * padded_per
        query = torch.randn(batch_size, 2, 128, dtype=torch.bfloat16, device="cuda:0")
        centroids = torch.randn(num_centroids, 128, dtype=torch.bfloat16, device="cuda:0")
        ot = torch.zeros(total, 1, 1, dtype=torch.bfloat16, device="cuda:0")
        oc = torch.zeros(total, 1, 1, dtype=torch.bfloat16, device="cuda:0")
        from blocksparseattention_9fbd0903_compiled_func import (
            blocksparseattention_9fbd0903_subgraph_0_impl,
        )
        from blocksparse_score_kernel import blocksparse_score_impl
        blocksparseattention_9fbd0903_subgraph_0_impl(query, centroids, ot, ctx)
        blocksparse_score_impl(query, centroids, oc, ctx)
        torch.cuda.synchronize()
        mask = torch.zeros(total, dtype=torch.bool, device="cuda:0")
        for w in range(batch_size):
            off = ctx.winfo_kv_offsets[w].item()
            ln = ctx.winfo_kv_lens[w].item()
            mask[off : off + ln] = True
        t = ot.view(-1)[mask].float()
        c = oc.view(-1)[mask].float()
        ae = (t - c).abs()
        re = ae / t.abs().clamp(min=1e-3)
        ok = ((ae < 0.30) | (re < 0.02)).all().item()
        print(f"  max_abs={ae.max().item():.4f}  mean_abs={ae.mean().item():.4f}  "
              f"{'PASS ✓' if ok else 'FAIL ✗'}")

    test_with_params(batch_size=64, items_per_workload=1)     # single page, partial
    test_with_params(batch_size=64, items_per_workload=4)     # single page, full
    test_with_params(batch_size=128, items_per_workload=32)   # 8 full pages
    test_with_params(batch_size=128, items_per_workload=29)   # 8 pages, last partial
    test_with_params(batch_size=568, items_per_workload=16)   # matches num_sms
