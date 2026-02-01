#!/usr/bin/env python3
"""
LABS Benchmark Runner

Runs different methods for values of N specified as input.
Uses eval_util from tutorial_notebook/evals for validation and energy computation.
Methods are stubbed for now - implement real algorithms as they become available.

Usage:
    python run_benchmark.py 3 4 5 10 20
    python run_benchmark.py 3-25
    python run_benchmark.py --classical-gpu 10 20     # include H100-optimized MTS (classical_gpu)
"""

import argparse
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
# Timing wrapper (add perf to any function run)
# ---------------------------------------------------------------------------

def timed_run(fn, *args, **kwargs) -> tuple:
    """Run fn(*args, **kwargs) and return (result, time_sec). Use to add perf timing to any method."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


# ---------------------------------------------------------------------------
# Stubbed methods (replace with real implementations)
# ---------------------------------------------------------------------------

# Base methods (always run). classical_gpu (H100-optimized MTS) is added when --classical-gpu is set.
METHODS_BASE = ["mts", "random", "trotter", "qmf"]
METHOD_CLASSICAL_GPU = "classical_gpu"


def _run_qmf(N: int) -> list[int]:
    """QAOA+Grover+MTS hybrid from impl-qmf/main.py. Returns sequence only (timing via timed_run)."""
    qmf_path = REPO_ROOT / "impl-qmf" / "main.py"
    if not qmf_path.exists():
        raise FileNotFoundError(f"impl-qmf/main.py not found (required for qmf method)")
    import importlib.util
    spec = importlib.util.spec_from_file_location("qmf_main", qmf_path)
    qmf_module = importlib.util.module_from_spec(spec)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec.loader.exec_module(qmf_module)
    seq, _ = qmf_module.run_hybrid(N, verbose=False)
    return seq


def _run_trotter(N: int) -> list[int]:
    """Trotter/counteradiabatic+MTS from impl-trotter/main.py. Returns sequence only (timing via timed_run)."""
    trotter_path = REPO_ROOT / "impl-trotter" / "main.py"
    if not trotter_path.exists():
        raise FileNotFoundError(f"impl-trotter/main.py not found (required for trotter method)")
    import importlib.util
    spec = importlib.util.spec_from_file_location("trotter_main", trotter_path)
    mod = importlib.util.module_from_spec(spec)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec.loader.exec_module(mod)
    seq, _ = mod.run_hybrid(N, verbose=False)
    return seq


def _run_mts(N: int) -> list[int]:
    """Memetic Tabu Search from impl-mts/main.py. Returns sequence only (list of ±1)."""
    mts_path = REPO_ROOT / "impl-mts" / "main.py"
    if not mts_path.exists():
        raise FileNotFoundError(f"impl-mts/main.py not found (required for mts method)")
    import importlib.util
    spec = importlib.util.spec_from_file_location("mts_module", mts_path)
    mts_module = importlib.util.module_from_spec(spec)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec.loader.exec_module(mts_module)
    random.seed(42)
    if hasattr(mts_module, "np"):
        mts_module.np.random.seed(42)
    best_s, _best_energy, _population = mts_module.memetic_tabu_search(
        N=N,
        population_size=50,
        max_generations=100,
        p_combine=0.9,
        initial_population=None,
        verbose=False,
    )
    return best_s.tolist() if hasattr(best_s, "tolist") else list(best_s)



def _run_random(N: int) -> list[int]:
    """Stub: Random search baseline. Returns sequence only."""
    return [random.choice([-1, 1]) for _ in range(N)]


def _run_classical_gpu(N: int) -> list[int]:
    """H100-optimized MTS from impl-mts/mts_h100_optimized.py. Returns sequence only (timing via timed_run)."""
    h100_path = REPO_ROOT / "impl-mts" / "mts_h100_optimized.py"
    if not h100_path.exists():
        raise FileNotFoundError(f"impl-mts/mts_h100_optimized.py not found (required for classical_gpu method)")
    import importlib.util
    spec = importlib.util.spec_from_file_location("mts_h100", h100_path)
    h100_module = importlib.util.module_from_spec(spec)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec.loader.exec_module(h100_module)
    random.seed(42)
    if hasattr(h100_module, "np"):
        h100_module.np.random.seed(42)
    if hasattr(h100_module, "cp"):
        h100_module.cp.random.seed(42)
    best_s, _best_energy, _population = h100_module.memetic_tabu_search(
        N=N,
        population_size=50,
        max_generations=100,
        p_combine=0.9,
        initial_population=None,
        verbose=False,
    )
    return best_s.tolist() if hasattr(best_s, "tolist") else list(best_s)


def run_method(method: str, N: int) -> tuple[list[int], float]:
    """Dispatch to the appropriate method. Returns (sequence, time_sec). Timing via timed_run."""
    if method == "mts":
        return timed_run(_run_mts, N)
    if method == "random":
        return timed_run(_run_random, N)
    if method == "trotter":
        return timed_run(_run_trotter, N)
    if method == "qmf":
        return timed_run(_run_qmf, N)
    if method == METHOD_CLASSICAL_GPU:
        return timed_run(_run_classical_gpu, N)
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
    parser = argparse.ArgumentParser(
        description="LABS Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: python run_benchmark.py --classical-gpu 10 20",
    )
    parser.add_argument(
        "--classical-gpu",
        action="store_true",
        help="Include H100-optimized MTS (classical_gpu) in the methods to run",
    )
    parser.add_argument("n_values", nargs="*", default=[], help="N values, e.g. 3 4 5 10 or 3-10")
    args = parser.parse_args()

    n_values = parse_n_values(args.n_values)
    methods = list(METHODS_BASE)
    if args.classical_gpu:
        methods.append(METHOD_CLASSICAL_GPU)

    results_path = SCRIPT_DIR / "results_classical_gpu.csv" if args.classical_gpu else RESULTS_CSV
    print(f"Benchmarking N = {n_values}")
    print(f"Methods: {methods}")
    run_benchmark(n_values, methods, results_path)


if __name__ == "__main__":
    main()
