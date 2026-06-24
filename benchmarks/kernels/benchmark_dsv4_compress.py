# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DVFS-robust micro-benchmark for the DeepSeek-V4 compressor kernels.

Why this exists
---------------
On a shared box the GPU shader clock idles to ~95 MHz and boosts to ~2.4 GHz (a
~25x swing). For microsecond kernels with gaps between launches, single-shot
device-time numbers swing ~10x run-to-run, so any absolute "HIP = 16us" figure
is meaningless on its own. We do NOT lock clocks (shared environment); instead
we measure defensively:

  * Only the **within-process ratio** HIP/Triton is trusted: both kernels are
    measured back-to-back under the same clock state, so DVFS cancels in the
    ratio. Absolute us are NOT comparable across runs.
  * Each kernel is timed by **HIP-graph replay**: N launches are captured into
    one CUDA graph and the whole replay is timed with a single event pair, so
    the GPU stays busy across launches (no host gaps that let the clock idle).
    This is NOT per-call ``cuda.Event`` timing, which folds clock-ramp into the
    kernel on a shared box.
  * The A/B is repeated for R rounds; the **median** is reported to shrug off
    residual ramp drift (min/max shown for stability).

This times the compressor KERNEL only (the HIP kernel is fused/plan-free, so its
device time already includes on-device boundary derivation).

Usage (run from the repo root so ``tests`` is importable):
    python benchmarks/kernels/benchmark_dsv4_compress.py
    python benchmarks/kernels/benchmark_dsv4_compress.py --shape hca --rounds 9
    python benchmarks/kernels/benchmark_dsv4_compress.py \
        --scenarios decode_boundary prefill_4096
"""

import argparse
import statistics
import sys
from pathlib import Path

import torch

# Reuse the test scaffolding (scenario/state-cache builders + runners). The
# helpers live under tests/; add the repo root to sys.path so this script runs
# standalone regardless of cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tests.kernels.attention.dsv4_compress_utils import (  # noqa: E402
    CSA_MAIN,
    HCA_MAIN,
    INDEXER_FP8,
    INDEXER_MXFP4,
    all_scenarios,
    build_shared_input,
    hip_available,
    run_hip,
    run_triton,
)

torch.set_default_device("cuda")

SHAPES = {
    "csa": CSA_MAIN,
    "hca": HCA_MAIN,
    "indexer_fp8": INDEXER_FP8,
    "indexer_mxfp4": INDEXER_MXFP4,
}

DEFAULT_SCENARIOS = [
    "decode_boundary",  # tiny, latency-bound
    "prefill_256",
    "prefill_1024",
    "prefill_4096",  # mid, HIP starts to win
    "prefill_32768",  # large, throughput-bound
]


def _graph_us(fn, n, reps=5, warm=5):
    """Per-launch device time via HIP-graph replay (us).

    ``n`` kernel launches are captured into one CUDA graph; replaying the graph
    keeps the GPU busy across all ``n`` launches (no host-side gaps that let the
    clock idle between them), and a single event pair times the whole replay.
    This is the DVFS-robust methodology — NOT per-call ``cuda.Event``, which on a
    shared box (95 MHz idle .. 2.4 GHz boost, ~25x) folds clock-ramp into the
    kernel. The median over ``reps`` replays is returned.
    """
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        for _ in range(n):
            fn()
    for _ in range(warm):
        g.replay()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        out.append(e0.elapsed_time(e1) * 1000.0 / n)  # ms/replay -> us/launch
    return statistics.median(out)


def _median_min_max(xs):
    return statistics.median(xs), min(xs), max(xs)


def _format_range(median_min_max: tuple[float, float, float]) -> str:
    median, low, high = median_min_max
    return f"{median:.2f}[{low:.2f}-{high:.2f}]"


def bench_scenario(ctx, rounds, n, have_triton):
    ctx.build()
    build_shared_input(ctx)
    kv, kv_3d = ctx.new_kv_cache()

    triton_fn = (lambda: run_triton(ctx, kv_3d)) if have_triton else None
    hip_fn = lambda: run_hip(ctx, kv_3d)

    # Heavy joint warmup to ramp the clock before any timed round.
    for _ in range(30):
        if triton_fn:
            triton_fn()
        hip_fn()
    torch.cuda.synchronize()

    # Interleave at the ROUND level so both kernels see the same clock
    # trajectory; the hip/triton ratio is computed per round then median-reduced.
    t_r, h_r, ratios = [], [], []
    for _ in range(rounds):
        h = _graph_us(hip_fn, n)
        h_r.append(h)
        if triton_fn:
            t = _graph_us(triton_fn, n)
            t_r.append(t)
            ratios.append(h / t if t > 0 else float("nan"))

    res = {"name": ctx.name, "hip": _median_min_max(h_r)}
    if t_r:
        res["triton"] = _median_min_max(t_r)
        res["ratio"] = _median_min_max(ratios)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", choices=list(SHAPES), default="csa")
    ap.add_argument(
        "--rounds",
        type=int,
        default=7,
        help="A/B rounds; median is reported (default 7)",
    )
    ap.add_argument(
        "--launches",
        type=int,
        default=100,
        help="kernel launches captured per CUDA graph (default 100)",
    )
    ap.add_argument(
        "--scenarios",
        nargs="*",
        default=None,
        help="scenario names (default: representative subset)",
    )
    ap.add_argument(
        "--format",
        choices=["text", "markdown"],
        default="text",
        help="output format (default: text)",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA/HIP device is not available")
    if not hip_available():
        raise SystemExit("dsv4 compressor ops are not registered in _rocm_C")

    shape = SHAPES[args.shape]
    # Triton MXFP4 uses NVIDIA PTX and can't run on AMD -> HIP-only for mxfp4.
    have_triton = shape.quant_format != "indexer_mxfp4"

    all_ctx = {c.name: c for c in all_scenarios(shape)}
    wanted = args.scenarios if args.scenarios else DEFAULT_SCENARIOS
    missing = set(wanted) - set(all_ctx)
    if missing:
        raise SystemExit(
            f"unknown scenarios: {sorted(missing)}; "
            f"available: {sorted(all_ctx)}"
        )

    results = []
    ratios = []
    for name in wanted:
        r = bench_scenario(all_ctx[name], args.rounds, args.launches, have_triton)
        if "ratio" in r:
            ratios.append(r["ratio"][0])
        results.append(r)

    if args.format == "markdown":
        print(f"#### {shape.label}")
        print()
        print(
            "| Scenario | Triton us/launch | HIP us/launch | "
            "HIP/Triton median[min..max] |"
        )
        print("|---|---:|---:|---:|")
        for r in results:
            triton = f"{r['triton'][0]:.1f}" if "triton" in r else "n/a"
            ratio = _format_range(r["ratio"]) if "ratio" in r else "n/a"
            print(f"| {r['name']} | {triton} | {r['hip'][0]:.1f} | {ratio} |")
        if ratios:
            g = statistics.geometric_mean(ratios)
            print()
            print(f"Geomean HIP/Triton: **{g:.3f}**.")
        return

    print(
        f"DVFS-robust compress-kernel benchmark  shape={shape.label}  "
        f"(rounds={args.rounds}, launches/graph={args.launches})"
    )
    print(
        "us/launch = HIP-graph-replay median[min..max]; hip/tri ratio is "
        "same-process (<1.0 = HIP faster). KERNEL only.\n"
    )

    hdr = f"{'scenario':<16}{'triton':>10}{'hip':>10}{'hip/tri':>16}"
    print(hdr)
    print("-" * len(hdr))

    for r in results:
        line = f"{r['name']:<16}"
        line += f"{r['triton'][0]:>10.1f}" if "triton" in r else f"{'n/a':>10}"
        line += f"{r['hip'][0]:>10.1f}"
        line += f"{_format_range(r['ratio']):>16}" if "ratio" in r else f"{'n/a':>16}"
        print(line)

    if ratios:
        g = statistics.geometric_mean(ratios)
        print(
            f"\ngeomean hip/triton (kernel device-time) = {g:.3f}  "
            f"({'HIP faster overall' if g < 1 else 'Triton faster overall'})"
        )


if __name__ == "__main__":
    main()
