"""Latency benchmark for ``Union()`` kernel variants (trtllm).

Compares the baseline ``union_trtllm.cu`` (single-threaded O(n^2) dedup),
``union_trtllm_hash.cu`` (parallel open-address hash dedup), and
``union_trtllm_sort.cu`` (CUB BlockRadixSort + adjacent-difference + scan)
across ``k ∈ {64, 128, 256}`` and a sweep of ``(batch_size, seq_len)``.

The Union signature differs from the regular topk ABI (no scores, takes
two ``(block_table, seqlens)`` pairs and writes a single
``(sparse_block_tables, sparse_seqlens)`` pair) so we don't reuse
``benchmark.py``'s harness — instead the dispatch path is reused via
``dispatcher.load_submission(..., cpp_source=_CPP_SOURCE_TRTLLM_UNION)``.

Outputs are correctness-checked against the baseline (sorted-set
equality + identical seqlens) on a tiny config before timing kicks in;
the dedup union is order-invariant up to the last-block-at-tail rule
that every variant honors.

Run::

    python -m vortex_torch.kernels.topk.benchmark_union
    python -m vortex_torch.kernels.topk.benchmark_union --num-samples 50 --ks 64,128,256
"""
from __future__ import annotations

import argparse
import datetime
import math
from collections import defaultdict
from pathlib import Path

import torch
import triton

from .dispatcher import (
    _CPP_SOURCE_TRTLLM_UNION,
    load_submission,
)


# Sweep matches the topk benchmark so cross-comparison is easy.
SEQ_LENS = [1024, 2048, 4096, 8192]
BATCH_SIZES = [16, 32, 64, 128]
DEFAULT_KS = [64, 128, 256]
DEFAULT_BOS = 1
DEFAULT_EOS = 2
DEFAULT_BLOCK_SIZE = 16
NUM_SAMPLES_DEFAULT = 20
DEVICE = "cuda"

_HERE = Path(__file__).resolve().parent
_CONFIGS_DIR = _HERE / "configs"
_REPORTS_DIR = _HERE / "reports"

BASELINE_CONFIG = _CONFIGS_DIR / "union_baseline_trtllm.json"
HASH_CONFIG     = _CONFIGS_DIR / "union_hash_trtllm.json"
SORT_CONFIG     = _CONFIGS_DIR / "union_sort_trtllm.json"


# ---------------------------------------------------------------------------
# Inputs synthesis
# ---------------------------------------------------------------------------

def _build_topk_inputs(
    *,
    eff_bs: int,
    max_blocks_per_seq: int,
    k: int,
    bos: int,
    eos: int,
    block_size: int,
    overlap_ratio: float,
    device: str,
    seed: int,
):
    """Synthesize two TopK-style ``(block_table, seqlens)`` inputs.

    Models the canonical ``TopK(k)`` output:
        block_table[bx] = [BOS … selected_k … EOS]
    with the BOS / EOS regions fixed (first ``bos`` / last ``eos`` dense
    blocks) and the middle drawn with a controllable overlap ratio
    between the two streams so the dedup workload spans the spectrum from
    "perfect overlap" (single TopK collapse) to "fully disjoint" (worst
    case dedup).

    Returns: ``(dense_seqlens, dense_block_tables, bt_0, sl_0, bt_1, sl_1,
                sparse_block_tables_OUT, sparse_seqlens_OUT, last_block_len)``.
    """
    g = torch.Generator(device=device).manual_seed(int(seed))

    dense_seqlens = torch.full(
        (eff_bs,), max_blocks_per_seq * block_size, dtype=torch.int32, device=device,
    )
    # Dense block_tables: each row holds 0..max_blocks_per_seq-1 (per-row).
    dense_block_tables = torch.arange(
        max_blocks_per_seq, dtype=torch.int32, device=device
    ).repeat(eff_bs, 1).contiguous()

    sparse_block_len = bos + k + eos
    sparse_tokens = sparse_block_len * block_size  # last block full in this synthesis

    # Middle selections per row, two streams with controllable overlap.
    middle_lo = bos
    middle_hi = max_blocks_per_seq - eos
    middle_pool = middle_hi - middle_lo
    if middle_pool < k:
        raise ValueError(
            f"k={k} too large for the synthesized middle range "
            f"({middle_pool} < {k}); try a larger seq_len or smaller k."
        )

    bt_a = torch.empty((eff_bs, max_blocks_per_seq), dtype=torch.int32, device=device)
    bt_b = torch.empty((eff_bs, max_blocks_per_seq), dtype=torch.int32, device=device)
    sl_a = torch.full((eff_bs,), sparse_tokens, dtype=torch.int32, device=device)
    sl_b = torch.full((eff_bs,), sparse_tokens, dtype=torch.int32, device=device)

    bt_a[:, :bos] = dense_block_tables[:, :bos]
    bt_b[:, :bos] = dense_block_tables[:, :bos]
    bt_a[:, bos + k:bos + k + eos] = dense_block_tables[:, middle_hi:middle_hi + eos]
    bt_b[:, bos + k:bos + k + eos] = dense_block_tables[:, middle_hi:middle_hi + eos]

    overlap_k = int(round(k * overlap_ratio))
    overlap_k = max(0, min(k, overlap_k))
    overlap_k = min(overlap_k, middle_pool)
    # Build per-row middles on CPU then move — keeps this synthesis simple.
    middle_a_cpu = torch.empty((eff_bs, k), dtype=torch.int32)
    middle_b_cpu = torch.empty((eff_bs, k), dtype=torch.int32)
    for row in range(eff_bs):
        perm = torch.randperm(middle_pool, generator=g, device=device).cpu()
        first_a = perm[:k]
        # Take the first ``overlap_k`` of perm for the shared core; sample
        # the remaining slots disjointly from the rest of the pool.
        remaining = perm[k:]
        # b reuses the shared prefix then picks unique tail items
        if remaining.numel() >= (k - overlap_k):
            tail_b = remaining[:k - overlap_k]
        else:
            tail_b = remaining[:k - overlap_k]  # may be short on tiny pools — degrades to more overlap
        first_b = torch.cat([first_a[:overlap_k], tail_b])[:k]
        if first_b.numel() < k:
            # Pad with shared items if pool was exhausted (effective full overlap).
            pad = first_a[:k - first_b.numel()]
            first_b = torch.cat([first_b, pad])[:k]

        middle_a_cpu[row] = (first_a + middle_lo).to(torch.int32)
        middle_b_cpu[row] = (first_b + middle_lo).to(torch.int32)
    bt_a[:, bos:bos + k] = middle_a_cpu.to(device)
    bt_b[:, bos:bos + k] = middle_b_cpu.to(device)

    sparse_block_tables_out = torch.zeros(
        (eff_bs, max_blocks_per_seq), dtype=torch.int32, device=device,
    )
    sparse_seqlens_out = torch.zeros((eff_bs,), dtype=torch.int32, device=device)

    last_block_len = block_size  # synthetic dense_seqlens are block-aligned
    return (
        dense_seqlens,
        dense_block_tables,
        bt_a, sl_a,
        bt_b, sl_b,
        sparse_block_tables_out,
        sparse_seqlens_out,
        last_block_len,
    )


# ---------------------------------------------------------------------------
# Kernel call wrapper (shared ABI for the 3 variants)
# ---------------------------------------------------------------------------

def _call(submission, inputs, eff_bs, max_blocks_per_seq, block_size):
    (dense_seqlens, dense_block_tables,
     bt_a, sl_a, bt_b, sl_b,
     sparse_block_tables, sparse_seqlens, _last) = inputs
    submission.topk(
        dense_seqlens,
        sparse_seqlens,
        dense_block_tables,
        bt_a, sl_a,
        bt_b, sl_b,
        sparse_block_tables,
        eff_bs,
        max_blocks_per_seq,
        block_size,
    )


# ---------------------------------------------------------------------------
# Pure-torch reference implementation (ground truth).
# ---------------------------------------------------------------------------

def union_reference_torch(
    *,
    dense_seqlens: torch.Tensor,
    dense_block_tables: torch.Tensor,
    block_tables_0: torch.Tensor,
    seqlens_0: torch.Tensor,
    block_tables_1: torch.Tensor,
    seqlens_1: torch.Tensor,
    max_blocks_per_seq: int,
    block_size: int,
):
    """Reference Union — pure torch / python, slow but obviously correct.

    Mirrors the per-row contract documented on ``output_func.Union``:

      * Dedupe ``block_tables_0[bx, :blocks_0] ∪ block_tables_1[bx, :blocks_1]``
        (excluding ``last_block_id``), append ``last_block_id`` at the tail.
      * ``sparse_seqlens[bx] = u * block_size + last_block_len`` where ``u``
        is the dedup count and ``last_block_len`` is the partial token
        count of the dense path's trailing block.

    Returns the same ``(sparse_block_tables, sparse_seqlens)`` shapes as
    the CUDA kernels.
    """
    eff_bs = int(dense_seqlens.shape[0])
    device = dense_seqlens.device
    out_bt = torch.zeros((eff_bs, max_blocks_per_seq), dtype=torch.int32, device=device)
    out_sl = torch.zeros((eff_bs,), dtype=torch.int32, device=device)

    ds = dense_seqlens.cpu().tolist()
    dt = dense_block_tables.cpu().tolist()
    bt0 = block_tables_0.cpu().tolist()
    sl0 = seqlens_0.cpu().tolist()
    bt1 = block_tables_1.cpu().tolist()
    sl1 = seqlens_1.cpu().tolist()

    bt_out = [[0] * max_blocks_per_seq for _ in range(eff_bs)]
    sl_out = [0] * eff_bs

    for bx in range(eff_bs):
        tokens = ds[bx]
        if tokens <= 0:
            sl_out[bx] = 0
            continue
        dense_block_len = (tokens + block_size - 1) // block_size
        last_mod = tokens % block_size
        last_block_len = block_size if last_mod == 0 else last_mod
        last_block_id = dt[bx][dense_block_len - 1]

        blocks0 = (sl0[bx] + block_size - 1) // block_size
        blocks1 = (sl1[bx] + block_size - 1) // block_size

        seen = []  # preserve first-seen order; semantics don't depend on it
        seen_set = set()
        for src in (bt0[bx][:blocks0], bt1[bx][:blocks1]):
            for v in src:
                if v == last_block_id:
                    continue
                if v in seen_set:
                    continue
                seen_set.add(v)
                seen.append(v)
        for i, v in enumerate(seen):
            bt_out[bx][i] = v
        u = len(seen)
        bt_out[bx][u] = last_block_id
        sl_out[bx] = u * block_size + last_block_len

    out_bt.copy_(torch.tensor(bt_out, dtype=torch.int32, device=device))
    out_sl.copy_(torch.tensor(sl_out, dtype=torch.int32, device=device))
    return out_bt, out_sl


# ---------------------------------------------------------------------------
# Correctness check (set equality + identical seqlens, tail = last block)
# ---------------------------------------------------------------------------

def _decode_rows(sparse_bt: torch.Tensor, sparse_sl: torch.Tensor, block_size: int):
    """Return per-row (set_excluding_tail, tail_block, num_blocks)."""
    eff_bs = sparse_bt.shape[0]
    bt = sparse_bt.cpu().tolist()
    sl = sparse_sl.cpu().tolist()
    out = []
    for row in range(eff_bs):
        tokens = sl[row]
        nblk = (tokens + block_size - 1) // block_size
        ids = bt[row][:nblk]
        if not ids:
            out.append((frozenset(), None, 0))
            continue
        tail = ids[-1]
        out.append((frozenset(ids[:-1]), tail, nblk))
    return out


def _verify(reference, proposal, *, label_ref: str, label_prop: str, eff_bs: int):
    """Assert each row's tail block matches and the leading set matches."""
    for row in range(eff_bs):
        ref_set, ref_tail, ref_nblk = reference[row]
        prop_set, prop_tail, prop_nblk = proposal[row]
        if ref_tail != prop_tail:
            raise RuntimeError(
                f"{label_prop} disagrees with {label_ref} on row {row}: "
                f"tail block {prop_tail} vs {ref_tail}"
            )
        if ref_nblk != prop_nblk:
            raise RuntimeError(
                f"{label_prop} disagrees with {label_ref} on row {row}: "
                f"block count {prop_nblk} vs {ref_nblk}"
            )
        if ref_set != prop_set:
            extra = prop_set - ref_set
            missing = ref_set - prop_set
            raise RuntimeError(
                f"{label_prop} disagrees with {label_ref} on row {row}: "
                f"extra={sorted(extra)[:10]}, missing={sorted(missing)[:10]}"
            )


# ---------------------------------------------------------------------------
# Bench loop
# ---------------------------------------------------------------------------

def _bench(fn, warmup: int = 25, rep: int = 100):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="mean")


def _mean(values):
    return sum(values) / len(values)


def _geomean(values):
    return math.exp(sum(math.log(v) for v in values) / len(values))


def _md_table(headers, rows):
    if not rows:
        return ""
    widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    fmt = lambda cells: "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    return "\n".join([fmt(headers), sep, *(fmt(r) for r in rows)])


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m vortex_torch.kernels.topk.benchmark_union",
        description="Latency benchmark for Union() kernel variants (trtllm).",
    )
    parser.add_argument(
        "--num-samples", type=int, default=NUM_SAMPLES_DEFAULT,
        help=f"samples per (k, batch_size, seq_len, overlap) cell (default: {NUM_SAMPLES_DEFAULT})",
    )
    parser.add_argument(
        "--ks", type=str, default=",".join(str(k) for k in DEFAULT_KS),
        help=f"comma-separated k bucket list (default: {DEFAULT_KS})",
    )
    parser.add_argument(
        "--bos", type=int, default=DEFAULT_BOS,
        help=f"block_reserved_bos (default: {DEFAULT_BOS})",
    )
    parser.add_argument(
        "--eos", type=int, default=DEFAULT_EOS,
        help=f"block_reserved_eos (default: {DEFAULT_EOS})",
    )
    parser.add_argument(
        "--block-size", type=int, default=DEFAULT_BLOCK_SIZE,
        help=f"block_size (default: {DEFAULT_BLOCK_SIZE})",
    )
    parser.add_argument(
        "--overlaps", type=str, default="0.0,0.5,1.0",
        help="comma-separated middle-overlap ratios in [0,1] (default: 0.0,0.5,1.0)",
    )
    parser.add_argument(
        "--report", type=Path, default=_REPORTS_DIR / "union_bench.md",
        help="markdown report path",
    )
    args = parser.parse_args(argv)

    ks = [int(s) for s in args.ks.split(",") if s.strip()]
    overlaps = [float(s) for s in args.overlaps.split(",") if s.strip()]

    print(f"compiling Union variants ...")
    submissions = {
        "baseline": load_submission(BASELINE_CONFIG, cpp_source=_CPP_SOURCE_TRTLLM_UNION),
        "hash":     load_submission(HASH_CONFIG,     cpp_source=_CPP_SOURCE_TRTLLM_UNION),
        "sort":     load_submission(SORT_CONFIG,     cpp_source=_CPP_SOURCE_TRTLLM_UNION),
    }
    labels = list(submissions.keys())
    print(f"compiled: {labels}")

    records = []  # one per (k, bs, sl, overlap, sample)
    for k in ks:
        for bs in BATCH_SIZES:
            for sl_tokens in SEQ_LENS:
                max_blocks_per_seq = sl_tokens // args.block_size
                if max_blocks_per_seq < (args.bos + k + args.eos):
                    # No room for selection — synthesis would degrade.
                    continue
                eff_bs = bs  # treat as eff_bs (= real_bs * num_kv_heads conceptually)
                for overlap in overlaps:
                    cell_log_speedups = {label: [] for label in labels if label != "baseline"}
                    cell_ms = {label: [] for label in labels}
                    for sample_idx in range(args.num_samples):
                        inputs = _build_topk_inputs(
                            eff_bs=eff_bs,
                            max_blocks_per_seq=max_blocks_per_seq,
                            k=k, bos=args.bos, eos=args.eos,
                            block_size=args.block_size,
                            overlap_ratio=overlap,
                            device=DEVICE,
                            seed=sample_idx,
                        )

                        # Per-variant fresh outputs (so each call writes its own).
                        outputs = {}
                        for label, sub in submissions.items():
                            outs = list(inputs)
                            # Reset output tensors for this variant.
                            outs[6] = torch.zeros_like(inputs[6])
                            outs[7] = torch.zeros_like(inputs[7])
                            outputs[label] = tuple(outs)
                            _call(sub, outputs[label], eff_bs, max_blocks_per_seq, args.block_size)
                        torch.cuda.synchronize()

                        # Correctness check (set + tail + count) — every
                        # variant (including the C++ baseline) compared
                        # against a slow pure-torch reference, NOT against
                        # any other kernel. Catches bugs where multiple
                        # kernels agree on the wrong answer.
                        ref_bt, ref_sl = union_reference_torch(
                            dense_seqlens=inputs[0],
                            dense_block_tables=inputs[1],
                            block_tables_0=inputs[2],
                            seqlens_0=inputs[3],
                            block_tables_1=inputs[4],
                            seqlens_1=inputs[5],
                            max_blocks_per_seq=max_blocks_per_seq,
                            block_size=args.block_size,
                        )
                        ref_decoded = _decode_rows(ref_bt, ref_sl, args.block_size)
                        for label in labels:
                            prop_decoded = _decode_rows(outputs[label][6], outputs[label][7], args.block_size)
                            _verify(ref_decoded, prop_decoded,
                                    label_ref="torch_ref", label_prop=label, eff_bs=eff_bs)

                        # Time each variant.
                        for label, sub in submissions.items():
                            outs = outputs[label]
                            fn = lambda sub=sub, outs=outs: _call(sub, outs, eff_bs, max_blocks_per_seq, args.block_size)
                            ms = _bench(fn)
                            cell_ms[label].append(ms)
                        for label in labels:
                            if label == "baseline":
                                continue
                            speedup = cell_ms["baseline"][-1] / cell_ms[label][-1]
                            cell_log_speedups[label].append(math.log(speedup))
                            records.append({
                                "k": k, "batch_size": bs, "seq_len": sl_tokens,
                                "overlap": overlap, "variant": label,
                                "ms_baseline": cell_ms["baseline"][-1],
                                "ms_variant":  cell_ms[label][-1],
                                "speedup": speedup,
                            })

                    ms_strs = ", ".join(
                        f"{label}={_mean(cell_ms[label]) * 1e3:.1f}us"
                        for label in labels
                    )
                    speedup_strs = ", ".join(
                        f"{label}={math.exp(_mean(cell_log_speedups[label])):.2f}x"
                        for label in labels if label != "baseline"
                    )
                    print(
                        f"k={k:>3} bs={bs:>3} sl={sl_tokens:>5} ov={overlap:.2f} | "
                        f"latency: {ms_strs} | vs baseline: {speedup_strs}"
                    )

    # ---- report ----
    args.report.parent.mkdir(parents=True, exist_ok=True)
    by_k = defaultdict(list)
    for rec in records:
        by_k[(rec["k"], rec["variant"])].append(rec)

    rows = []
    for (k, variant), rs in sorted(by_k.items()):
        rows.append([
            str(k),
            variant,
            f"{_geomean([r['speedup'] for r in rs]):.3f}x",
            f"{_mean([r['ms_baseline'] for r in rs]) * 1e3:.2f}us",
            f"{_mean([r['ms_variant']  for r in rs]) * 1e3:.2f}us",
            str(len(rs)),
        ])
    table = _md_table(
        ["k", "variant", "geomean speedup", "baseline avg", "variant avg", "n"],
        rows,
    )

    by_k_bs_sl_ov = defaultdict(dict)
    for rec in records:
        key = (rec["k"], rec["batch_size"], rec["seq_len"], rec["overlap"])
        by_k_bs_sl_ov[key][rec["variant"]] = rec["speedup"]
    cell_rows = []
    for (k, bs, sl_tokens, ov), m in sorted(by_k_bs_sl_ov.items()):
        cell_rows.append([
            str(k), str(bs), str(sl_tokens), f"{ov:.2f}",
            *(f"{m.get(v, float('nan')):.2f}x" for v in [l for l in labels if l != "baseline"]),
        ])
    cell_table = _md_table(
        ["k", "batch_size", "seq_len", "overlap",
         *[v for v in labels if v != "baseline"]],
        cell_rows,
    )

    ts = datetime.datetime.now().isoformat(timespec="seconds")
    body = (
        f"# Union() benchmark — `union_trtllm.cu` variants\n"
        f"\n"
        f"- Generated: {ts}\n"
        f"- Variants: {labels}\n"
        f"- Block size: {args.block_size}, BOS: {args.bos}, EOS: {args.eos}\n"
        f"- Samples per cell: {args.num_samples}\n"
        f"\n"
        f"Speedup = `baseline_ms / variant_ms`. Geomean across all cells.\n"
        f"\n"
        f"## 1. By k bucket\n"
        f"\n"
        f"{table}\n"
        f"\n"
        f"## 2. Per cell (speedup over baseline)\n"
        f"\n"
        f"{cell_table}\n"
    )
    args.report.write_text(body)
    print(f"\nreport written: {args.report}")


if __name__ == "__main__":
    main()
