#!/usr/bin/env python3
"""
Compare runtime of CPU MTS (main.py), single-GPU H100 (mts_h100_optimized.py),
and multi-GPU H100s (mts_h100s_mult.py) over a range of N.

Usage:
    python compare_runtime.py [--N-min 1] [--N-max 50] [--population_size P] [--generations G] [--output PATH]
    python compare_runtime.py --N-max 50 -o results/runtime_comparison.png
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent

CONFIGS = [
    ("main.py", "CPU (main.py)"),
    ("mts_h100_optimized.py", "Single-GPU H100"),
    ("mts_h100s_mult.py", "Multi-GPU H100s"),
]


def run_and_time(script_name: str, N: int, population_size: int, max_generations: int) -> float:
    """Run a script with given args and return wall-clock time in seconds."""
    script = SCRIPT_DIR / script_name
    cmd = [sys.executable, str(script), str(N), str(population_size), str(max_generations)]
    t0 = time.perf_counter()
    result = subprocess.run(
        cmd,
        cwd=str(SCRIPT_DIR),
        capture_output=True,
        text=True,
        timeout=3600,
    )
    t1 = time.perf_counter()
    if result.returncode != 0:
        raise RuntimeError(
            f"{script_name} failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    return t1 - t0


def main():
    parser = argparse.ArgumentParser(
        description="Compare runtime of CPU, single-GPU, and multi-GPU MTS over a range of N."
    )
    parser.add_argument(
        "--N-min",
        type=int,
        default=1,
        help="Minimum sequence length (default: 1)",
    )
    parser.add_argument(
        "--N-max",
        type=int,
        default=50,
        help="Maximum sequence length (default: 50)",
    )
    parser.add_argument(
        "--population_size",
        type=int,
        default=50,
        help="MTS population size (default: 50)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=100,
        help="Max MTS generations (default: 100)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=SCRIPT_DIR / "runtime_comparison.png",
        help="Output plot path (default: impl-mts/runtime_comparison.png)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Only print timings, do not save plot",
    )
    args = parser.parse_args()

    N_min, N_max = args.N_min, args.N_max
    if N_min > N_max:
        N_min, N_max = N_max, N_min
    N_values = np.arange(N_min, N_max + 1, dtype=int)
    pop = args.population_size
    gen = args.generations

    # [implementation_index][n_index] -> runtime in seconds
    runtimes = [[np.nan] * len(N_values) for _ in CONFIGS]

    print(f"Comparing runtimes for N in [{N_min}, {N_max}], population_size={pop}, generations={gen}")
    print("-" * 60)

    for i_n, N in enumerate(N_values):
        print(f"N={N}", end=" ", flush=True)
        for i_impl, (script, label) in enumerate(CONFIGS):
            try:
                t = run_and_time(script, N, pop, gen)
                runtimes[i_impl][i_n] = t
                print(f"  {label.split()[0]}: {t:.2f}s", end="", flush=True)
            except Exception as e:
                print(f"  {label.split()[0]}: FAIL", end="", flush=True)
                runtimes[i_impl][i_n] = np.nan
        print()

    print("-" * 60)

    if args.no_plot:
        return

    # Line plot: N vs runtime for each implementation
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2e86ab", "#a23b72", "#f18f01"]
    for (script, label), times, color in zip(CONFIGS, runtimes, colors):
        ax.plot(
            N_values,
            times,
            "o-",
            label=label,
            color=color,
            markersize=3,
            linewidth=1.5,
        )
    ax.set_xlabel("Sequence length N")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title(f"MTS runtime vs N (pop={pop}, gen={gen})")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(N_min - 0.5, N_max + 0.5)
    if not all(np.isnan(t) for row in runtimes for t in row):
        valid = [t for row in runtimes for t in row if not np.isnan(t)]
        if valid:
            ax.set_ylim(0, max(valid) * 1.05)
    plt.tight_layout()

    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {out}")


if __name__ == "__main__":
    main()
