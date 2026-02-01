#!/usr/bin/env python3
"""
LABS Benchmark Runner

Runs different methods for values of N specified as input.
Uses eval_util from tutorial_notebook/evals for validation and energy computation.
Methods are stubbed for now - implement real algorithms as they become available.

Usage:
    python run_benchmark.py 3 4 5 10 20
    python run_benchmark.py 3-25
"""

import csv
import random
import sys
import time
from pathlib import Path

# Add tutorial evals to path for eval_util
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
EVALS_DIR = REPO_ROOT / "tutorial_notebook" / "evals"
sys.path.insert(0, str(EVALS_DIR))

from eval_util import (
    compute_energy,
    compute_merit_factor,
    sequence_to_runlength,
    get_expected_optimal_energy,
    normalized_energy_distance,
)

ANSWERS_CSV = EVALS_DIR / "answers.csv"
RESULTS_CSV = SCRIPT_DIR / "results.csv"

# ---------------------------------------------------------------------------
# Stubbed methods (replace with real implementations)
# ---------------------------------------------------------------------------

METHODS = ["mts", "tabu", "random"]


def run_mts(N: int) -> tuple[list[int], float]:
    """Stub: Memetic Tabu Search. Returns (sequence, time_sec)."""
    start = time.perf_counter()
    # Stub: return random sequence
    seq = [random.choice([-1, 1]) for _ in range(N)]
    elapsed = time.perf_counter() - start
    return seq, elapsed


def run_tabu(N: int) -> tuple[list[int], float]:
    """Stub: Tabu Search. Returns (sequence, time_sec)."""
    start = time.perf_counter()
    # Stub: return random sequence
    seq = [random.choice([-1, 1]) for _ in range(N)]
    elapsed = time.perf_counter() - start
    return seq, elapsed


def run_random(N: int) -> tuple[list[int], float]:
    """Stub: Random search baseline. Returns (sequence, time_sec)."""
    start = time.perf_counter()
    seq = [random.choice([-1, 1]) for _ in range(N)]
    elapsed = time.perf_counter() - start
    return seq, elapsed


def run_method(method: str, N: int) -> tuple[list[int], float]:
    """Dispatch to the appropriate method. Returns (sequence, time_sec)."""
    if method == "mts":
        return run_mts(N)
    if method == "tabu":
        return run_tabu(N)
    if method == "random":
        return run_random(N)
    raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# Benchmark orchestration
# ---------------------------------------------------------------------------

def parse_n_values(args: list[str]) -> list[int]:
    """Parse N from args. Supports: 3 4 5 10 or 3-10 (inclusive range)."""
    if not args:
        return list(range(3, 26))  # default 3..25

    values = []
    for a in args:
        if "-" in a and a[0].isdigit():
            parts = a.split("-")
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                lo, hi = int(parts[0]), int(parts[1])
                values.extend(range(lo, hi + 1))
                continue
        try:
            values.append(int(a))
        except ValueError:
            pass
    return sorted(set(values))


def run_benchmark(n_values: list[int], methods: list[str], results_path: Path) -> None:
    """Run all (N, method) combinations and write results to CSV."""
    rows = []

    for N in n_values:
        opt_energy = get_expected_optimal_energy(N, ANSWERS_CSV)

        for method in methods:
            try:
                seq, time_sec = run_method(method, N)
                energy = compute_energy(seq)
                F_N = compute_merit_factor(seq, energy)
                seq_rl = sequence_to_runlength(seq)

                norm_dist = None
                if opt_energy is not None:
                    norm_dist = normalized_energy_distance(energy, opt_energy)

                rows.append({
                    "N": N,
                    "method": method,
                    "energy": energy,
                    "F_N": round(F_N, 4),
                    "optimal_energy": opt_energy if opt_energy is not None else "",
                    "normalized_distance": round(norm_dist, 6) if norm_dist is not None else "",
                    "time_sec": round(time_sec, 6),
                    "sequence": seq_rl,
                })
            except Exception as e:
                rows.append({
                    "N": N,
                    "method": method,
                    "energy": "",
                    "F_N": "",
                    "optimal_energy": opt_energy if opt_energy is not None else "",
                    "normalized_distance": "",
                    "time_sec": "",
                    "sequence": f"ERROR: {e}",
                })

    # Write CSV
    fieldnames = ["N", "method", "energy", "F_N", "optimal_energy", "normalized_distance", "time_sec", "sequence"]
    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {results_path}")


def main():
    n_values = parse_n_values(sys.argv[1:])
    print(f"Benchmarking N = {n_values}")
    print(f"Methods: {METHODS}")
    run_benchmark(n_values, METHODS, RESULTS_CSV)


if __name__ == "__main__":
    main()
