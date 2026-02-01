#!/usr/bin/env python3
"""
Tutorial LABS Interface for Automated Testing

Runs the NVIDIA tutorial workflow (quantum-enhanced Trotter counteradiabatic + MTS)
from 01_quantum_enhanced_optimization_LABS.ipynb. Delegates to impl-trotter for
the actual implementation. Compatible with --classical-gpu and --quantum-gpu flags.

Usage:
    python main.py [N] [--classical-gpu] [--quantum-gpu] [--verbose]
    # Or from benchmarks:
    from main import run_nvidia
    seq, time_sec = run_nvidia(N=20, use_gpu_mts=False, use_quantum_gpu=False)
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
TUTORIAL_DIR = Path(__file__).resolve().parent

# Ensure impl-trotter and tutorial auxiliary_files are findable
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(TUTORIAL_DIR))
sys.path.insert(0, str(TUTORIAL_DIR / "auxiliary_files"))


def run_nvidia(
    N: int,
    use_gpu_mts: bool = False,
    use_quantum_gpu: bool = False,
    verbose: bool = False,
    T: float = 1.0,
    n_steps: int = 1,
    shots: int = 200,
    population_size: int = 50,
    max_generations: int = 30,
    p_combine: float = 0.9,
) -> tuple[list[int], float]:
    """
    Run the NVIDIA tutorial workflow (Trotter counteradiabatic + MTS hybrid).

    Uses the same algorithm as impl-trotter, which ports the tutorial notebook.
    Returns (best_sequence_as_list, time_sec). Sequence is list of ±1 for eval_util.

    Args:
        N: Sequence length.
        use_gpu_mts: When True (--classical-gpu), use H100-optimized MTS for
            the classical refinement step. Quantum sampling remains on CPU.
        use_quantum_gpu: When True (--quantum-gpu), use full H100-optimized path:
            cudaq nvidia target for quantum sampling + GPU MTS.
        verbose: Print progress.
        T, n_steps, shots, population_size, max_generations, p_combine:
            Algorithm parameters (same as impl-trotter).

    Returns:
        (sequence, time_sec) where sequence is list of ±1.
    """
    import importlib.util

    trotter_path = REPO_ROOT / "impl-trotter" / "main.py"
    if not trotter_path.exists():
        raise FileNotFoundError(
            f"impl-trotter/main.py not found (required for nvidia method)"
        )

    spec = importlib.util.spec_from_file_location("impl_trotter_main", trotter_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if use_quantum_gpu:
        seq, elapsed = mod.run_hybrid_h100_optimized(
            N,
            T=T,
            n_steps=n_steps,
            shots=shots,
            population_size=population_size,
            max_generations=max_generations,
            p_combine=p_combine,
            verbose=verbose,
        )
    else:
        seq, elapsed = mod.run_hybrid(
            N,
            T=T,
            n_steps=n_steps,
            shots=shots,
            population_size=population_size,
            max_generations=max_generations,
            p_combine=p_combine,
            verbose=verbose,
            use_gpu_mts=use_gpu_mts,
        )

    return seq, elapsed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run NVIDIA tutorial LABS workflow for automated testing"
    )
    parser.add_argument("N", type=int, nargs="?", default=10, help="Sequence length")
    parser.add_argument(
        "--classical-gpu",
        action="store_true",
        dest="classical_gpu",
        help="Use H100-optimized MTS for classical refinement",
    )
    parser.add_argument(
        "--quantum-gpu",
        action="store_true",
        dest="quantum_gpu",
        help="Use full H100 path (cudaq nvidia + GPU MTS)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    seq, elapsed = run_nvidia(
        N=args.N,
        use_gpu_mts=args.classical_gpu,
        use_quantum_gpu=args.quantum_gpu,
        verbose=args.verbose,
    )
    print(f"N={args.N} best sequence (runlength): {seq}")
    print(f"Time: {elapsed:.4f}s")
