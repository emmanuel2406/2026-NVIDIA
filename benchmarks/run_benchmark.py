#!/usr/bin/env python3
"""
LABS Benchmark Runner

Runs different methods for values of N specified as input.
Uses eval_util from tutorial_notebook/evals for validation and energy computation.
Methods are stubbed for now - implement real algorithms as they become available.

Usage:
    python run_benchmark.py 3 4 5 10 20
    python run_benchmark.py 3-25
    python run_benchmark.py --trials 5 10 20         # run each (N, method) 5 times, report mean stats
    python run_benchmark.py --classical-gpu 10 20     # add classical_gpu and H100 MTS in trotter/qmf
    python run_benchmark.py --quantum-gpu 10 20       # trotter and qmf use H100-optimized kernels (cudaq nvidia + GPU MTS)
"""

import argparse
import csv
import random
import sys
import time
from pathlib import Path

# Add tutorial evals and benchmarks dir to path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
EVALS_DIR = REPO_ROOT / "tutorial_notebook" / "evals"
sys.path.insert(0, str(EVALS_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from eval_util import (
    compute_energy,
    compute_merit_factor,
    sequence_to_runlength,
    get_expected_optimal_energy,
    normalized_energy_distance,
)

from plot_utils import plot_normalized_distance_vs_n, plot_energies_bar

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

# Base methods (always run). With --classical-gpu: add classical_gpu and trotter/qmf use H100 MTS.
# With --quantum-gpu: trotter and qmf use full H100-optimized code kernels (cudaq nvidia + GPU MTS).
METHODS_BASE = ["mts", "random", "trotter", "qmf", "nvidia"]
METHOD_CLASSICAL_GPU = "classical_gpu"
METHOD_TROTTER_H100 = "trotter_h100"  # optional extra method name; --quantum-gpu switches kernel for "trotter"/"qmf"


def _run_qmf(N: int, use_gpu_mts: bool = False) -> list[int]:
    """QAOA+Grover+MTS hybrid from impl-qmf/main.py. Returns sequence only (timing via timed_run).
    When use_gpu_mts=True, uses H100-optimized MTS for the classical refinement step."""
    qmf_path = REPO_ROOT / "impl-qmf" / "main.py"
    if not qmf_path.exists():
        raise FileNotFoundError(f"impl-qmf/main.py not found (required for qmf method)")
    import importlib.util
    spec = importlib.util.spec_from_file_location("qmf_main", qmf_path)
    qmf_module = importlib.util.module_from_spec(spec)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec.loader.exec_module(qmf_module)
    seq, _ = qmf_module.run_hybrid(N, verbose=False, use_gpu_mts=use_gpu_mts)
    return seq


def _run_trotter(N: int, use_gpu_mts: bool = False) -> list[int]:
    """Trotter/counteradiabatic+MTS from impl-trotter/main.py. Returns sequence only (timing via timed_run).
    When use_gpu_mts=True, uses H100-optimized MTS for the classical refinement step."""
    trotter_path = REPO_ROOT / "impl-trotter" / "main.py"
    if not trotter_path.exists():
        raise FileNotFoundError(f"impl-trotter/main.py not found (required for trotter method)")
    import importlib.util
    spec = importlib.util.spec_from_file_location("trotter_main", trotter_path)
    mod = importlib.util.module_from_spec(spec)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec.loader.exec_module(mod)
    seq, _ = mod.run_hybrid(N, verbose=False, use_gpu_mts=use_gpu_mts)
    return seq


def _run_trotter_h100(N: int) -> list[int]:
    """H100-optimized Trotter (cudaq nvidia + GPU MTS) from impl-trotter/main.py. Returns sequence only."""
    trotter_path = REPO_ROOT / "impl-trotter" / "main.py"
    if not trotter_path.exists():
        raise FileNotFoundError(f"impl-trotter/main.py not found (required for trotter_h100 method)")
    import importlib.util
    spec = importlib.util.spec_from_file_location("trotter_main", trotter_path)
    mod = importlib.util.module_from_spec(spec)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec.loader.exec_module(mod)
    seq, _ = mod.run_hybrid_h100_optimized(N, verbose=False)
    return seq


def _run_qmf_h100(N: int) -> list[int]:
    """H100-optimized QMF (cudaq nvidia + GPU MTS) from impl-qmf/main.py. Returns sequence only."""
    qmf_path = REPO_ROOT / "impl-qmf" / "main.py"
    if not qmf_path.exists():
        raise FileNotFoundError(f"impl-qmf/main.py not found (required for qmf H100 method)")
    import importlib.util
    spec = importlib.util.spec_from_file_location("qmf_main", qmf_path)
    qmf_module = importlib.util.module_from_spec(spec)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec.loader.exec_module(qmf_module)
    seq, _ = qmf_module.run_hybrid_h100_optimized(N, verbose=False)
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


def _run_nvidia(N: int, use_gpu_mts: bool = False, use_quantum_gpu: bool = False) -> list[int]:
    """NVIDIA tutorial workflow (Trotter counteradiabatic + MTS) from tutorial_notebook/main.py.
    Returns sequence only (timing via timed_run).
    When use_gpu_mts=True (--classical-gpu), uses H100-optimized MTS for classical refinement.
    When use_quantum_gpu=True (--quantum-gpu), uses full H100 path (cudaq nvidia + GPU MTS)."""
    tutorial_path = REPO_ROOT / "tutorial_notebook" / "main.py"
    if not tutorial_path.exists():
        raise FileNotFoundError(
            f"tutorial_notebook/main.py not found (required for nvidia method)"
        )
    import importlib.util

    spec = importlib.util.spec_from_file_location("tutorial_main", tutorial_path)
    mod = importlib.util.module_from_spec(spec)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec.loader.exec_module(mod)
    seq, _ = mod.run_nvidia(
        N,
        use_gpu_mts=use_gpu_mts,
        use_quantum_gpu=use_quantum_gpu,
        verbose=False,
    )
    return seq


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


def run_method(method: str, N: int, use_gpu_mts: bool = False, use_quantum_gpu: bool = False) -> tuple[list[int], float]:
    """Dispatch to the appropriate method. Returns (sequence, time_sec). Timing via timed_run.
    When use_gpu_mts=True, trotter and qmf use H100-optimized MTS for their classical refinement step.
    When use_quantum_gpu=True, trotter and qmf use their full H100-optimized code kernels (cudaq nvidia + GPU MTS)."""
    if method == "mts":
        return timed_run(_run_mts, N)
    if method == "random":
        return timed_run(_run_random, N)
    if method == "trotter":
        if use_quantum_gpu:
            return timed_run(_run_trotter_h100, N)
        return timed_run(_run_trotter, N, use_gpu_mts)
    if method == "qmf":
        if use_quantum_gpu:
            return timed_run(_run_qmf_h100, N)
        return timed_run(_run_qmf, N, use_gpu_mts)
    if method == "nvidia":
        if use_quantum_gpu:
            return timed_run(_run_nvidia, N, False, True)
        return timed_run(_run_nvidia, N, use_gpu_mts, False)
    if method == METHOD_CLASSICAL_GPU:
        return timed_run(_run_classical_gpu, N)
    if method == METHOD_TROTTER_H100:
        return timed_run(_run_trotter_h100, N)
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


def run_benchmark(n_values: list[int], methods: list[str], results_path: Path, use_gpu_mts: bool = False, use_quantum_gpu: bool = False, trials: int = 1) -> None:
    """Run all (N, method) combinations and write results to CSV.
    When trials > 1, each (N, method) is run trials times and quantitative stats (energy, F_N, normalized_distance, time_sec) are reported as arithmetic mean.
    When use_gpu_mts=True, trotter and qmf use H100-optimized MTS for their classical refinement step.
    When use_quantum_gpu=True, trotter and qmf use their full H100-optimized code kernels."""
    rows = []

    for N in n_values:
        opt_energy = get_expected_optimal_energy(N, ANSWERS_CSV)

        for method in methods:
            energies: list[float] = []
            F_Ns: list[float] = []
            norm_dists: list[float] = []
            time_secs: list[float] = []
            best_seq_rl: str | None = None
            best_energy: float | None = None
            last_error: Exception | None = None

            for _ in range(trials):
                try:
                    seq, time_sec = run_method(method, N, use_gpu_mts=use_gpu_mts, use_quantum_gpu=use_quantum_gpu)
                    energy = compute_energy(seq)
                    F_N = compute_merit_factor(seq, energy)
                    seq_rl = sequence_to_runlength(seq)
                    norm_dist = normalized_energy_distance(energy, opt_energy) if opt_energy is not None else None

                    energies.append(energy)
                    F_Ns.append(F_N)
                    if norm_dist is not None:
                        norm_dists.append(norm_dist)
                    time_secs.append(time_sec)
                    if best_energy is None or energy < best_energy:
                        best_energy = energy
                        best_seq_rl = seq_rl
                except Exception as e:
                    last_error = e

            if not energies:
                rows.append({
                    "N": N,
                    "method": method,
                    "energy": "",
                    "F_N": "",
                    "optimal_energy": opt_energy if opt_energy is not None else "",
                    "normalized_distance": "",
                    "time_sec": "",
                    "sequence": f"ERROR: {last_error}",
                })
            else:
                mean_energy = sum(energies) / len(energies)
                mean_F_N = sum(F_Ns) / len(F_Ns)
                mean_time = sum(time_secs) / len(time_secs)
                mean_norm_dist = (sum(norm_dists) / len(norm_dists)) if norm_dists else None
                rows.append({
                    "N": N,
                    "method": method,
                    "energy": round(mean_energy, 6),
                    "F_N": round(mean_F_N, 4),
                    "optimal_energy": opt_energy if opt_energy is not None else "",
                    "normalized_distance": round(mean_norm_dist, 6) if mean_norm_dist is not None else "",
                    "time_sec": round(mean_time, 6),
                    "sequence": best_seq_rl or "",
                })

    # Write CSV
    fieldnames = ["N", "method", "energy", "F_N", "optimal_energy", "normalized_distance", "time_sec", "sequence"]
    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {results_path}")

    # Plot normalized_distance vs N (one line per method)
    plot_path = results_path.with_suffix(".png")
    plot_normalized_distance_vs_n(results_path, out_path=plot_path)
    # Plot energies per method as bar chart with optimal energy line
    energies_plot_path = results_path.with_stem(results_path.stem + "_energies").with_suffix(".png")
    plot_energies_bar(results_path, out_path=energies_plot_path)


def main():
    parser = argparse.ArgumentParser(
        description="LABS Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: python run_benchmark.py --classical-gpu 10 20",
    )
    parser.add_argument(
        "--classical-gpu",
        action="store_true",
        help="Add classical_gpu method and use H100-optimized MTS for trotter/qmf refinement",
    )
    parser.add_argument(
        "--quantum-gpu",
        action="store_true",
        help="Use H100-optimized code kernels for trotter and qmf (cudaq nvidia + GPU MTS)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        metavar="T",
        help="Run each (N, method) T times and report arithmetic mean of energy, F_N, normalized_distance, time_sec (default: 1)",
    )
    parser.add_argument("n_values", nargs="*", default=[], help="N values, e.g. 3 4 5 10 or 3-10")
    args = parser.parse_args()

    n_values = parse_n_values(args.n_values)
    methods = list(METHODS_BASE)
    if args.classical_gpu:
        methods.append(METHOD_CLASSICAL_GPU)
    # --quantum-gpu does not add extra methods; it switches trotter/qmf to H100-optimized kernels

    if args.classical_gpu and args.quantum_gpu:
        results_path = SCRIPT_DIR / "results_gpu.csv"
    elif args.classical_gpu:
        results_path = SCRIPT_DIR / "results_classical_gpu.csv"
    elif args.quantum_gpu:
        results_path = SCRIPT_DIR / "results_quantum_gpu.csv"
    else:
        results_path = RESULTS_CSV
    use_gpu_mts = args.classical_gpu  # trotter and qmf use H100 MTS when --classical-gpu
    use_quantum_gpu = args.quantum_gpu  # trotter and qmf use full H100 kernels when --quantum-gpu
    print(f"Benchmarking N = {n_values}")
    print(f"Methods: {methods}")
    if use_gpu_mts:
        print("(trotter and qmf will use H100-optimized MTS for classical refinement)")
    if use_quantum_gpu:
        print("(trotter and qmf will use H100-optimized code kernels: cudaq nvidia + GPU MTS)")
    if args.trials < 1:
        parser.error("--trials must be >= 1")
    print(f"Trials per (N, method): {args.trials}")
    run_benchmark(n_values, methods, results_path, use_gpu_mts=use_gpu_mts, use_quantum_gpu=use_quantum_gpu, trials=args.trials)


if __name__ == "__main__":
    main()
