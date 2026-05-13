"""Latency + recall benchmark for a top-k proposal vs. a baseline.

``baseline`` defaults to the CUB-sort kernel (``sort_topk.cu``);
``proposal`` defaults to the 8-bit radix kernel (``radix_topk.cu``).
Both are loaded from JSON configs via ``dispatcher.load_submission``.
Recall is measured with the baseline's top-K as the reference and the
proposal's top-K as the candidate.

For each ``(batch_size, seq_len)`` cell, ``--num-samples`` score tensors
are drawn from a rotating family of distributions (normal at several
scales, uniform, lognormal, exponential, bimodal, sparse-spike,
clustered-near-threshold) so the recall and latency means reflect
sensitivity to the input distribution rather than a single noise pattern.

Run as a script::

    python -m vortex_torch.kernels.topk.benchmark
    python -m vortex_torch.kernels.topk.benchmark --num-samples 50
"""

from __future__ import annotations

import argparse
import datetime
import math
from collections import defaultdict
from pathlib import Path

import torch
import triton

from .dispatcher import load_submission


SEQ_LENS = [1024, 1536, 2048, 4096]
BATCH_SIZES = [16, 32, 64, 128]

DEFAULT_K = 253

RESERVE_BOS = 0
RESERVE_EOS = 0
DEVICE = "cuda"
NUM_SAMPLES = 100


def eval_ks_for(k: int) -> list[int]:
    """Default recall eval points: ``floor(k*3/4)`` and ``k``.

    Matches the historical 189/253 split (189 = floor(253*3/4)).
    """
    return sorted({max(1, k * 3 // 4), k})

# Rotating family of score distributions. Samples cycle through this list
# (sample_idx % len(DISTRIBUTIONS)) so a fixed budget covers each shape
# evenly; per-sample seeds change the realisation within a shape.
DISTRIBUTIONS: list[tuple[str, dict]] = [
    ("normal",              {"mean": 0.0, "std": 1.0}),
    ("normal",              {"mean": 0.0, "std": 0.1}),
    ("normal",              {"mean": 0.0, "std": 10.0}),
    ("normal",              {"mean": 5.0, "std": 1.0}),
    ("uniform",             {"low": -1.0, "high": 1.0}),
    ("uniform",             {"low": 0.0,  "high": 1.0}),
    ("lognormal",           {"mu": 0.0, "sigma": 1.0}),
    ("exponential",         {"rate": 1.0}),
    ("bimodal",             {"m1": -2.0, "m2": 2.0, "std": 0.5}),
    ("bimodal",             {"m1": 0.0,  "m2": 5.0, "std": 1.0}),
    ("triangular",          {"low": -1.0, "high": 1.0}),
    ("triangular",          {"low": 0.0,  "high": 1.0}),
    ("student_t",           {"df": 5}),
    ("clustered_threshold", {"center": 0.0, "outlier_frac": 0.05, "outlier_offset": 3.0}),
]

_HERE = Path(__file__).resolve().parent
_CONFIGS_DIR = _HERE / "configs"
_REPORTS_DIR = _HERE / "reports"
BASELINE_CONFIG = _CONFIGS_DIR / "sort_default.json"
PROPOSAL_CONFIG = _CONFIGS_DIR / "radix_default.json"


def load_submissions(
    baseline_config: str | Path = BASELINE_CONFIG,
    proposal_config: str | Path = PROPOSAL_CONFIG,
):
    """Load both kernels from JSON configs via the dispatcher.

    Each config names a ``.cu`` source and (optionally) a substitutions
    map plus extra nvcc flags. Returns ``(baseline_module, proposal_module)``;
    each exposes ``topk(...)`` with the shared C signature.
    """
    baseline = load_submission(baseline_config)
    proposal = load_submission(proposal_config)
    return baseline, proposal


def make_static_inputs(batch_size, seq_len, k, device="cuda"):
    """Build the deterministic indptr / indices / output buffers.

    These do not depend on the score distribution, so they're reused
    across all samples in a ``(batch_size, seq_len)`` cell.
    """
    dense_kv_indptr = torch.arange(
        0, batch_size * seq_len + 1, seq_len, dtype=torch.int32, device=device
    )
    dense_kv_indices = torch.arange(
        0, batch_size * seq_len, dtype=torch.int32, device=device
    )
    sparse_kv_indptr = torch.arange(
        0, batch_size * k + 1, k, dtype=torch.int32, device=device
    )
    sparse_kv_indices_baseline = torch.empty(
        batch_size * k, dtype=torch.int32, device=device
    )
    sparse_kv_indices_proposal = torch.empty(
        batch_size * k, dtype=torch.int32, device=device
    )
    return (
        dense_kv_indptr,
        dense_kv_indices,
        sparse_kv_indptr,
        sparse_kv_indices_baseline,
        sparse_kv_indices_proposal,
    )


def sample_scores(
    batch_size: int,
    seq_len: int,
    distribution: str,
    params: dict,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
) -> torch.Tensor:
    """Draw a flat score tensor of shape ``[batch_size * seq_len]``.

    Supported distributions: ``normal``, ``uniform``, ``lognormal``,
    ``exponential``, ``bimodal``, ``triangular``, ``student_t``,
    ``clustered_threshold``. Returned in ``dtype`` (default ``bfloat16``
    to match the kernels).
    """
    n = batch_size * seq_len
    g = torch.Generator(device=device).manual_seed(int(seed))
    fp32 = torch.float32

    if distribution == "normal":
        x = torch.randn(n, generator=g, device=device, dtype=fp32)
        x = x * float(params["std"]) + float(params["mean"])
    elif distribution == "uniform":
        u = torch.rand(n, generator=g, device=device, dtype=fp32)
        lo, hi = float(params["low"]), float(params["high"])
        x = u * (hi - lo) + lo
    elif distribution == "lognormal":
        z = torch.randn(n, generator=g, device=device, dtype=fp32)
        x = torch.exp(z * float(params["sigma"]) + float(params["mu"]))
    elif distribution == "exponential":
        u = torch.rand(n, generator=g, device=device, dtype=fp32).clamp_min_(1e-7)
        x = -torch.log(u) / float(params["rate"])
    elif distribution == "bimodal":
        mask = torch.rand(n, generator=g, device=device, dtype=fp32) < 0.5
        noise = torch.randn(n, generator=g, device=device, dtype=fp32) * float(params["std"])
        x = torch.where(
            mask,
            noise + float(params["m1"]),
            noise + float(params["m2"]),
        )
    elif distribution == "triangular":
        # Sum of two uniforms is triangular on the same interval — symmetric
        # bell shape, low Gini, no degenerate tied values.
        u1 = torch.rand(n, generator=g, device=device, dtype=fp32)
        u2 = torch.rand(n, generator=g, device=device, dtype=fp32)
        lo, hi = float(params["low"]), float(params["high"])
        x = (u1 + u2) * 0.5 * (hi - lo) + lo
    elif distribution == "student_t":
        # z / sqrt(chi2_df / df) — heavier tails than normal for small df,
        # asymptotically normal as df → ∞. df=5 keeps moments finite while
        # giving meaningful tail probability.
        df = int(params["df"])
        z = torch.randn(n, generator=g, device=device, dtype=fp32)
        chi2 = torch.zeros(n, device=device, dtype=fp32)
        for _ in range(df):
            ni = torch.randn(n, generator=g, device=device, dtype=fp32)
            chi2 = chi2 + ni * ni
        x = z / torch.sqrt(chi2 / float(df)).clamp_min_(1e-7)
    elif distribution == "clustered_threshold":
        # Most values clustered near a center with tiny noise; small fraction get a large offset.
        x = torch.randn(n, generator=g, device=device, dtype=fp32) * 0.01 + float(params["center"])
        outlier = torch.rand(n, generator=g, device=device, dtype=fp32) < float(params["outlier_frac"])
        x = torch.where(outlier, x + float(params["outlier_offset"]), x)
    else:
        raise ValueError(f"unknown distribution: {distribution!r}")

    return x.to(dtype)


def distribution_spec(sample_idx: int) -> tuple[str, dict]:
    """Pick a ``(distribution, params)`` pair from the rotating list."""
    return DISTRIBUTIONS[sample_idx % len(DISTRIBUTIONS)]


def distribution_label(dist: str, params: dict) -> str:
    """Stable human-readable identifier for a distribution + parameters."""
    if not params:
        return dist
    body = ",".join(f"{k}={v}" for k, v in sorted(params.items()))
    return f"{dist}({body})"


def _geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(v) for v in values) / len(values))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return ""
    widths = [
        max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)
    ]
    fmt_row = lambda cells: "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    return "\n".join([fmt_row(headers), sep, *(fmt_row(r) for r in rows)])


def report_path_for(proposal_config: Path) -> Path:
    """Compute the report path for a given proposal config.

    Swaps the last ``configs`` segment in the resolved path with
    ``reports``, so both layouts work:

      - ``<root>/configs/<tag>/batch_0_id0.json`` →
        ``<root>/reports/<tag>/batch_0_id0.md``
      - ``<root>/k_256/configs/<tag>/batch_0_id0.json`` →
        ``<root>/k_256/reports/<tag>/batch_0_id0.md``

    Configs without any ``configs`` segment fall back to
    ``reports/<stem>.md`` next to this file.
    """
    p = Path(proposal_config).resolve()
    parts = list(p.parts)
    try:
        idx = len(parts) - 1 - parts[::-1].index("configs")
    except ValueError:
        return _REPORTS_DIR / f"{p.stem}.md"
    parts[idx] = "reports"
    return Path(*parts).with_suffix(".md")


def write_report(
    report_path: Path,
    records: list[dict],
    proposal_config: Path,
    baseline_config: Path,
    num_samples: int,
    k: int,
    eval_ks: list[int],
) -> None:
    """Write a markdown report aggregating ``records`` along two axes.

    Records are flat per-sample dicts with keys: ``batch_size``,
    ``seq_len``, ``distribution``, ``speedup``, and one ``recall_<kk>``
    per ``kk in eval_ks``.
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)
    recall_cols = [f"R@{kk}" for kk in eval_ks]
    recall_keys = [f"recall_{kk}" for kk in eval_ks]

    def _recall_cells(rs):
        return [f"{_mean([r[rk] for r in rs]):.4f}" for rk in recall_keys]

    # Table 1: per (batch_size, seq_len)
    g1: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for rec in records:
        g1[(rec["batch_size"], rec["seq_len"])].append(rec)

    table1_rows = []
    for (bs, sl), rs in sorted(g1.items()):
        table1_rows.append([
            str(bs),
            str(sl),
            f"{_geomean([r['speedup'] for r in rs]):.4f}x",
            *_recall_cells(rs),
            str(len(rs)),
        ])
    table1 = _md_table(
        ["batch_size", "seq_len", "geomean speedup", *recall_cols, "n"],
        table1_rows,
    )

    # Table 2: per (seq_len, distribution)
    g2: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for rec in records:
        g2[(rec["seq_len"], rec["distribution"])].append(rec)

    table2_rows = []
    for (sl, dist), rs in sorted(g2.items()):
        table2_rows.append([
            str(sl),
            dist,
            f"{_geomean([r['speedup'] for r in rs]):.4f}x",
            *_recall_cells(rs),
            str(len(rs)),
        ])
    table2 = _md_table(
        ["seq_len", "distribution", "geomean speedup", *recall_cols, "n"],
        table2_rows,
    )

    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    body = (
        f"# Top-k benchmark report — {report_path.stem}\n"
        f"\n"
        f"- Proposal: `{proposal_config}`\n"
        f"- Baseline: `{baseline_config}`\n"
        f"- K (top-k size): {k}\n"
        f"- Samples per cell: {num_samples}\n"
        f"- Generated: {timestamp}\n"
        f"\n"
        f"Speedup is the geometric mean of per-sample "
        f"`baseline_ms / proposal_ms`. Recalls are arithmetic means.\n"
        f"\n"
        f"## 1. By (batch_size, seq_len)\n"
        f"\n"
        f"{table1}\n"
        f"\n"
        f"## 2. By (seq_len, distribution)\n"
        f"\n"
        f"{table2}\n"
    )
    report_path.write_text(body)


def _call(
    submission,
    scores,
    dense_kv_indptr,
    sparse_kv_indptr,
    dense_kv_indices,
    sparse_kv_indices,
    batch_size,
    reserve_bos,
    reserve_eos,
    seq_len,
):
    submission.topk(
        scores,
        dense_kv_indptr,
        sparse_kv_indptr,
        dense_kv_indices,
        sparse_kv_indices,
        batch_size,
        reserve_bos,
        reserve_eos,
        seq_len,
    )


def compute_recall_at_ks(ref_indices, pred_indices, batch_size, k, eval_ks):
    """``ref`` (baseline) is the ordered ground-truth top-k.

    Recall@r = fraction of baseline top-r recovered anywhere in proposal's K.
    """
    ref = ref_indices.view(batch_size, k).to(torch.int64)
    pred = pred_indices.view(batch_size, k).to(torch.int64)

    recalls = {}
    for r in eval_ks:
        ref_r = ref[:, :r]
        match = (ref_r.unsqueeze(2) == pred.unsqueeze(1))  # [B, r, K]
        hit = match.any(dim=2)                              # [B, r]
        recall = hit.float().sum(dim=1) / r
        recalls[r] = recall.mean().item()
    return recalls


def bench_latency(fn, warmup: int = 25, rep: int = 100):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    return triton.testing.do_bench(
        fn,
        warmup=warmup,
        rep=rep,
        return_mode="mean",
    )


def eval_one_sample(
    baseline_submission,
    proposal_submission,
    static_inputs,
    scores,
    batch_size,
    reserve_bos,
    reserve_eos,
    seq_len,
    k,
    eval_ks,
):
    """Run both kernels on a single ``scores`` tensor; return (ms_b, ms_p, recalls)."""
    (
        dense_kv_indptr,
        dense_kv_indices,
        sparse_kv_indptr,
        sparse_kv_indices_baseline,
        sparse_kv_indices_proposal,
    ) = static_inputs

    def fn_baseline():
        _call(
            baseline_submission,
            scores, dense_kv_indptr, sparse_kv_indptr, dense_kv_indices,
            sparse_kv_indices_baseline,
            batch_size, reserve_bos, reserve_eos, seq_len,
        )

    def fn_proposal():
        _call(
            proposal_submission,
            scores, dense_kv_indptr, sparse_kv_indptr, dense_kv_indices,
            sparse_kv_indices_proposal,
            batch_size, reserve_bos, reserve_eos, seq_len,
        )

    fn_baseline()
    fn_proposal()
    torch.cuda.synchronize()
    recalls = compute_recall_at_ks(
        sparse_kv_indices_baseline,
        sparse_kv_indices_proposal,
        batch_size=batch_size,
        k=k,
        eval_ks=eval_ks,
    )

    ms_baseline = bench_latency(fn_baseline)
    ms_proposal = bench_latency(fn_proposal)
    return ms_baseline, ms_proposal, recalls


def warm_run(baseline_submission, proposal_submission):
    """Touch each compiled kernel once on a tiny batch.

    Compilation already happened in ``load_submissions``; this just primes
    the CUDA context / module-load before the timed sweep starts.
    """
    static = make_static_inputs(batch_size=1, seq_len=128, k=16, device=DEVICE)
    scores = sample_scores(1, 128, "normal", {"mean": 0.0, "std": 1.0}, device=DEVICE)
    (dense_indptr, dense_idx, sparse_indptr,
     out_baseline, out_proposal) = static
    baseline_submission.topk(
        scores, dense_indptr, sparse_indptr, dense_idx, out_baseline,
        1, 0, 0, 128,
    )
    proposal_submission.topk(
        scores, dense_indptr, sparse_indptr, dense_idx, out_proposal,
        1, 0, 0, 128,
    )
    torch.cuda.synchronize()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m vortex_torch.kernels.topk.benchmark",
        description="Latency + recall benchmark for a top-k proposal vs. a baseline.",
    )
    parser.add_argument(
        "--proposal-config",
        type=Path,
        default=PROPOSAL_CONFIG,
        help=f"JSON config for the proposal kernel (default: {PROPOSAL_CONFIG})",
    )
    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=BASELINE_CONFIG,
        help=f"JSON config for the baseline kernel (default: {BASELINE_CONFIG})",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=NUM_SAMPLES,
        help=f"per-cell samples drawn from the distribution rotation (default: {NUM_SAMPLES})",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"top-k size (default: {DEFAULT_K}); recall is reported at floor(k*3/4) and k",
    )
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)

    k = args.k
    eval_ks = eval_ks_for(k)

    torch.cuda.init()
    print(f"baseline config: {args.baseline_config}")
    print(f"proposal config: {args.proposal_config}")
    print(f"K (top-k size): {k}")
    print(f"recall eval points: {eval_ks}")
    print(f"samples per cell: {args.num_samples}")
    print("compiling kernels (one-time JIT) ...")
    baseline_submission, proposal_submission = load_submissions(
        baseline_config=args.baseline_config,
        proposal_config=args.proposal_config,
    )
    warm_run(baseline_submission, proposal_submission)
    print("compile done; starting sweep")

    records: list[dict] = []
    for bs in BATCH_SIZES:
        for seq_len in SEQ_LENS:
            cell_log_speedups: list[float] = []
            cell_recalls: dict[int, list[float]] = {r: [] for r in eval_ks}

            static = make_static_inputs(bs, seq_len, k, device=DEVICE)

            for sample_idx in range(args.num_samples):
                dist, params = distribution_spec(sample_idx)
                scores = sample_scores(
                    bs, seq_len, dist, params,
                    device=DEVICE, dtype=torch.bfloat16, seed=sample_idx,
                )
                ms_baseline, ms_proposal, recalls = eval_one_sample(
                    baseline_submission,
                    proposal_submission,
                    static_inputs=static,
                    scores=scores,
                    batch_size=bs,
                    reserve_bos=RESERVE_BOS,
                    reserve_eos=RESERVE_EOS,
                    seq_len=seq_len,
                    k=k,
                    eval_ks=eval_ks,
                )
                speedup = ms_baseline / ms_proposal
                cell_log_speedups.append(math.log(speedup))
                for kk in eval_ks:
                    cell_recalls[kk].append(recalls[kk])
                records.append({
                    "batch_size": bs,
                    "seq_len": seq_len,
                    "sample_idx": sample_idx,
                    "distribution": distribution_label(dist, params),
                    "speedup": speedup,
                    **{f"recall_{kk}": recalls[kk] for kk in eval_ks},
                })

            geomean_speedup = math.exp(sum(cell_log_speedups) / len(cell_log_speedups))
            recall_str = ", ".join(
                f"R@{kk}={_mean(cell_recalls[kk]):.4f}" for kk in eval_ks
            )
            print(
                f"bs={bs:>4}, seq_len={seq_len:>6} | "
                f"speedup_geomean={geomean_speedup:.4f}x | {recall_str}"
            )

    report_path = report_path_for(Path(args.proposal_config))
    write_report(
        report_path,
        records,
        proposal_config=Path(args.proposal_config),
        baseline_config=Path(args.baseline_config),
        num_samples=args.num_samples,
        k=k,
        eval_ks=eval_ks,
    )
    print(f"\nreport written: {report_path}")


if __name__ == "__main__":
    main()
