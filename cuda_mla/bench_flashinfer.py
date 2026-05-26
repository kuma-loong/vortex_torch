"""Benchmark flashinfer's CUDA/CUTLASS MLA decode on the vortex block-table
workload, vs the fp32 reference. Builds flashinfer's preferred paged layout
(separate contiguous ckv/kpe caches) — same useful bytes (576/token) as the
fused-latent vortex kernels, so GB/s is directly comparable.

    FLASHINFER_DISABLE_VERSION_CHECK=1 PYTHONPATH=.:./flashinfer \
        CUDA_VISIBLE_DEVICES=<gpu> python cuda_mla/bench_flashinfer.py
"""
import statistics
import torch
import flashinfer

KV_DIM, CKV, KPE = 576, 512, 64
PEAK = 8000.0


def make(bs, H, page, sel, dev="cuda", dt=torch.bfloat16):
    assert sel % page == 0
    nb = sel // page
    num_pages = bs * nb
    ckv = torch.randn(num_pages, page, CKV, device=dev, dtype=dt)
    kpe = torch.randn(num_pages, page, KPE, device=dev, dtype=dt)
    pages = torch.randperm(num_pages, device=dev, dtype=torch.int32)
    page_table = pages.view(bs, nb).contiguous()
    kv_indices = page_table.reshape(-1).contiguous()
    kv_indptr = (torch.arange(bs + 1, device=dev, dtype=torch.int32) * nb)
    qo_indptr = torch.arange(bs + 1, device=dev, dtype=torch.int32)
    kv_len = torch.full((bs,), sel, device=dev, dtype=torch.int32)
    q_nope = torch.randn(bs, H, CKV, device=dev, dtype=dt)
    q_pe = torch.randn(bs, H, KPE, device=dev, dtype=dt)
    return dict(ckv=ckv, kpe=kpe, page_table=page_table, kv_indices=kv_indices,
                kv_indptr=kv_indptr, qo_indptr=qo_indptr, kv_len=kv_len,
                q_nope=q_nope, q_pe=q_pe, nb=nb, num_pages=num_pages)


def reference(d, sm_scale, page):
    bs, H, _ = d["q_nope"].shape
    out = torch.empty(bs, H, CKV, device="cuda", dtype=torch.float32)
    qn, qp = d["q_nope"].float(), d["q_pe"].float()
    ckv, kpe = d["ckv"].float(), d["kpe"].float()
    pt, kvlen = d["page_table"], d["kv_len"]
    for b in range(bs):
        sl = int(kvlen[b]); nb = (sl + page - 1) // page
        ck = torch.cat([ckv[int(pt[b, j])] for j in range(nb)])[:sl]   # [sl,512]
        kp = torch.cat([kpe[int(pt[b, j])] for j in range(nb)])[:sl]   # [sl,64]
        score = (qn[b] @ ck.t() + qp[b] @ kp.t()) * sm_scale           # [H,sl]
        out[b] = torch.softmax(score, -1) @ ck
    return out


def bench(bs, H, page, sel, backend="auto", reps=3, check=False):
    d = make(bs, H, page, sel)
    sm_scale = 1.0 / (KV_DIM ** 0.5)
    ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    w = flashinfer.mla.BatchMLAPagedAttentionWrapper(ws, backend=backend)
    w.plan(d["qo_indptr"], d["kv_indptr"], d["kv_indices"], d["kv_len"],
           H, CKV, KPE, page, False, sm_scale, torch.bfloat16, torch.bfloat16)
    o = torch.empty(bs, H, CKV, device="cuda", dtype=torch.bfloat16)

    def run():
        w.run(d["q_nope"], d["q_pe"], d["ckv"], d["kpe"], out=o)
    err = float("nan")
    if check:
        run(); torch.cuda.synchronize()
        err = (o.float() - reference(d, sm_scale, page)).abs().max().item()
    vals = []
    for _ in range(reps):
        for _ in range(15):
            run()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(40):
            run()
        e.record(); torch.cuda.synchronize()
        vals.append(bs * sel * KV_DIM * 2 / ((s.elapsed_time(e) / 40) * 1e-3) / 1e9)
    return statistics.median(vals), err


if __name__ == "__main__":
    import sys
    backend = sys.argv[1] if len(sys.argv) > 1 else "auto"
    # correctness first on a small case
    g, err = bench(4, 16, 64, 256, backend=backend, check=True)
    print(f"[backend={backend}] correctness maxerr={err:.2e}")
    print(f"\n{'H':>3}{'bs':>5}{'page':>5}{'sel':>6}  {'GB/s':>8}{'%peak':>7}")
    for H in (16, 20):
        for page in (32, 64):
            for bs in (8, 32, 64, 128):
                g, _ = bench(bs, H, page, 2048, backend=backend)
                print(f"{H:>3}{bs:>5}{page:>5}{2048:>6}  {g:>8.0f}{g/PEAK*100:>6.1f}%")
