"""Compare FlashInfer's BatchDecodeWithPagedKVCacheWrapper against
trtllm_batch_decode_with_kv_cache on the same paged-KV inputs (NHD layout).

Both paths receive identical Q tensors and identical paged KV caches; only
the page-table representation differs (CSR for the wrapper, block-tables
for the trtllm-gen kernel).
"""

import math

import torch
import flashinfer


def build_inputs():
    torch.manual_seed(0)

    cfg = dict(
        num_layers=32,
        num_qo_heads=64,
        num_kv_heads=8,
        head_dim=128,
        max_num_pages=128,
        page_size=16,
        batch_size=7,
        device="cuda:0",
        dtype=torch.float16,
    )

    device, dtype = cfg["device"], cfg["dtype"]

    kv_page_indices = torch.arange(
        cfg["max_num_pages"], dtype=torch.int32, device=device
    )
    kv_page_indptr = torch.tensor(
        [0, 17, 29, 44, 48, 66, 100, 128], dtype=torch.int32, device=device
    )
    kv_last_page_len = torch.tensor(
        [1, 7, 14, 4, 3, 1, 16], dtype=torch.int32, device=device
    )

    # Split-KV NHD layout, accepted by both the wrapper and trtllm:
    #   K, V each [pages, page_size, num_kv_heads, head_dim]
    kv_cache_at_layer = [
        (
            torch.randn(
                cfg["max_num_pages"], cfg["page_size"],
                cfg["num_kv_heads"], cfg["head_dim"],
                dtype=dtype, device=device,
            ),
            torch.randn(
                cfg["max_num_pages"], cfg["page_size"],
                cfg["num_kv_heads"], cfg["head_dim"],
                dtype=dtype, device=device,
            ),
        )
        for _ in range(cfg["num_layers"])
    ]

    return cfg, kv_page_indptr, kv_page_indices, kv_last_page_len, kv_cache_at_layer


def csr_to_block_tables(indptr, indices, last_page_len, page_size, device):
    """CSR page table -> (block_tables [B, max_blocks], seq_lens [B], max_seq_len)."""
    indptr_cpu = indptr.cpu().tolist()
    last_cpu = last_page_len.cpu().tolist()
    batch_size = len(indptr_cpu) - 1

    num_blocks = [indptr_cpu[b + 1] - indptr_cpu[b] for b in range(batch_size)]
    seq_lens_list = [(num_blocks[b] - 1) * page_size + last_cpu[b] for b in range(batch_size)]
    max_blocks = max(num_blocks)
    max_seq_len = max(seq_lens_list)

    block_tables = torch.zeros((batch_size, max_blocks), dtype=torch.int32, device=device)
    for b in range(batch_size):
        s, e = indptr_cpu[b], indptr_cpu[b + 1]
        block_tables[b, : e - s] = indices[s:e].to(torch.int32)

    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)
    return block_tables, seq_lens, max_seq_len


def main():
    cfg, indptr, indices, last_page_len, kv_cache_at_layer = build_inputs()
    device, dtype = cfg["device"], cfg["dtype"]

    # ---- Wrapper setup ----
    ws_wrapper = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws_wrapper, "NHD")
    wrapper.plan(
        indptr,
        indices,
        last_page_len,
        cfg["num_qo_heads"],
        cfg["num_kv_heads"],
        cfg["head_dim"],
        cfg["page_size"],
        pos_encoding_mode="NONE",
        data_type=dtype,
    )

    # ---- trtllm setup ----
    block_tables, seq_lens, max_seq_len = csr_to_block_tables(
        indptr, indices, last_page_len, cfg["page_size"], device
    )
    ws_trtllm = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    sm_scale = 1.0 / math.sqrt(cfg["head_dim"])

    # ---- Run and compare per layer ----
    max_abs = 0.0
    max_rel = 0.0
    last_shapes = None

    for i in range(cfg["num_layers"]):
        q = torch.randn(
            cfg["batch_size"], cfg["num_qo_heads"], cfg["head_dim"],
            dtype=dtype, device=device,
        )
        k_cache, v_cache = kv_cache_at_layer[i]

        o_wrapper = wrapper.run(q, (k_cache, v_cache))

        o_trtllm = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            query=q,
            kv_cache=(k_cache, v_cache),
            workspace_buffer=ws_trtllm,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            bmm1_scale=sm_scale,
            bmm2_scale=1.0,
            kv_layout="NHD",
        )

        diff = (o_wrapper.float() - o_trtllm.float()).abs()
        max_abs = max(max_abs, diff.max().item())
        rel = diff / (o_wrapper.float().abs() + 1e-3)
        max_rel = max(max_rel, rel.max().item())
        last_shapes = (tuple(o_wrapper.shape), tuple(o_trtllm.shape))

    print(f"layers compared : {cfg['num_layers']}")
    print(f"output shapes   : wrapper={last_shapes[0]}  trtllm={last_shapes[1]}")
    print(f"max abs diff    : {max_abs:.5e}")
    print(f"max rel diff    : {max_rel:.5e}")


if __name__ == "__main__":
    main()
