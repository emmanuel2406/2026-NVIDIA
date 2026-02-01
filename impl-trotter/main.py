"""
Trotter/Counteradiabatic + MTS Hybrid for LABS (runnable from benchmarks).

Ports the logic from 01_quantum_enhanced_optimization_LABS-checkpoint(1).ipynb:
1. Trotterized counteradiabatic circuit (Eq. 15) -> quantum samples
2. MTS with that distribution as initial population

Usage:
    from main import run_hybrid
    best_seq, time_sec = run_hybrid(N, verbose=False)
Returns best_seq as list of ±1 for eval_util compatibility.
"""

from pathlib import Path
import sys
import time
import random

try:
    import cudaq
    import numpy as np
    _CUDAQ_AVAILABLE = True
except ImportError:
    _CUDAQ_AVAILABLE = False
    np = None

# Add impl-trotter and auxiliary_files to path for labs_utils
TROTTER_DIR = Path(__file__).resolve().parent
REPO_ROOT = TROTTER_DIR.parent
sys.path.insert(0, str(TROTTER_DIR))
sys.path.insert(0, str(TROTTER_DIR / "auxiliary_files"))

# ---------------------------------------------------------------------------
# Trotter Circuit (from notebook Eq. 15)
# ---------------------------------------------------------------------------


def _get_interactions(N: int):
    """
    Generates G2 and G4 based on loop limits in Eq. 15.
    Returns standard 0-based indices as lists of lists.
    """
    G2 = []
    G4 = []
    for i in range(N - 2):
        for k in range(1, (N - i - 1) // 2 + 1):
            G2.append([i, i + k])
    for i in range(N - 3):
        for t in range(1, (N - i - 2) // 2 + 1):
            for k in range(t + 1, N - i - t):
                G4.append([i, i + t, i + k, i + k + t])
    return G2, G4


def _bitstring_to_sequence(bitstring: str) -> "np.ndarray":
    """'0' -> +1, '1' -> -1."""
    return np.array([1 if b == "0" else -1 for b in bitstring])


def _define_kernels():
    """Define CUDA-Q kernels (must be at module level for cudaq)."""
    if not _CUDAQ_AVAILABLE:
        return None, None, None

    @cudaq.kernel
    def two_qubit_block(q0: cudaq.qubit, q1: cudaq.qubit, theta: float):
        """R_YZ(theta) * R_ZY(theta) from Fig. 3."""
        x.ctrl(q0, q1)
        rz(theta, q1)
        x.ctrl(q0, q1)
        rx(np.pi / 2, q0)
        rx(np.pi / 2, q1)
        x.ctrl(q0, q1)
        rz(theta, q1)
        x.ctrl(q0, q1)
        rx(-np.pi / 2, q0)
        rx(-np.pi / 2, q1)

    @cudaq.kernel
    def four_qubit_block(
        q0: cudaq.qubit, q1: cudaq.qubit, q2: cudaq.qubit, q3: cudaq.qubit, theta: float
    ):
        """R_YZZZ * R_ZYZZ * R_ZZYZ * R_ZZZY from Fig. 4."""
        rx(-np.pi / 2, q0)
        ry(np.pi / 2, q1)
        ry(-np.pi / 2, q2)
        rx(-np.pi / 2, q3)
        x.ctrl(q0, q1)
        rz(-np.pi / 2, q1)
        x.ctrl(q0, q1)
        x.ctrl(q2, q3)
        rz(-np.pi / 2, q3)
        x.ctrl(q2, q3)
        ry(-np.pi / 2, q1)
        rx(-np.pi / 2, q1)
        ry(np.pi / 2, q2)
        rx(-np.pi / 2, q2)
        x.ctrl(q1, q2)
        rz(theta, q2)
        x.ctrl(q1, q2)
        rx(np.pi / 2, q1)
        ry(np.pi / 2, q1)
        rx(np.pi / 2, q2)
        ry(-np.pi / 2, q2)
        x.ctrl(q0, q1)
        rz(np.pi / 2, q1)
        x.ctrl(q0, q1)
        x.ctrl(q2, q3)
        rz(np.pi / 2, q3)
        x.ctrl(q2, q3)
        rx(np.pi / 2, q0)
        ry(-np.pi / 2, q1)
        rx(-np.pi / 2, q1)
        ry(np.pi / 2, q2)
        rx(-np.pi / 2, q2)
        rx(np.pi / 2, q3)
        x.ctrl(q0, q1)
        rz(-np.pi / 2, q1)
        x.ctrl(q0, q1)
        x.ctrl(q1, q2)
        rz(theta, q2)
        x.ctrl(q1, q2)
        x.ctrl(q2, q3)
        rz(-np.pi / 2, q3)
        x.ctrl(q2, q3)
        rx(-np.pi, q0)
        rx(np.pi / 2, q1)
        ry(np.pi / 2, q1)
        rx(np.pi / 2, q2)
        ry(-np.pi / 2, q2)
        rx(-np.pi, q3)
        x.ctrl(q0, q1)
        rz(-np.pi / 2, q1)
        x.ctrl(q0, q1)
        x.ctrl(q1, q2)
        rz(theta, q2)
        x.ctrl(q1, q2)
        x.ctrl(q2, q3)
        rz(-np.pi / 2, q3)
        x.ctrl(q2, q3)
        rx(-np.pi / 2, q0)
        ry(-np.pi / 2, q1)
        ry(np.pi / 2, q2)
        rx(-np.pi / 2, q3)
        x.ctrl(q0, q1)
        rz(np.pi / 2, q1)
        x.ctrl(q0, q1)
        x.ctrl(q2, q3)
        rz(np.pi / 2, q3)
        x.ctrl(q2, q3)
        ry(-np.pi / 2, q1)
        rx(-np.pi / 2, q1)
        ry(np.pi / 2, q2)
        rx(-np.pi / 2, q2)
        x.ctrl(q1, q2)
        rz(theta, q2)
        x.ctrl(q1, q2)
        rx(np.pi / 2, q1)
        ry(np.pi / 2, q1)
        rx(np.pi / 2, q2)
        ry(-np.pi / 2, q2)
        x.ctrl(q0, q1)
        rz(np.pi / 2, q1)
        x.ctrl(q0, q1)
        x.ctrl(q2, q3)
        rz(np.pi / 2, q3)
        x.ctrl(q2, q3)
        rx(np.pi / 2, q0)
        ry(-np.pi / 2, q1)
        ry(np.pi / 2, q2)
        rx(np.pi / 2, q3)

    @cudaq.kernel
    def trotterized_circuit(
        N: int,
        G2: list[list[int]],
        G4: list[list[int]],
        steps: int,
        dt: float,
        T: float,
        thetas: list[float],
    ):
        """Full Trotterized counteradiabatic circuit from Eq. 15."""
        reg = cudaq.qvector(N)
        h(reg)
        for step in range(steps):
            theta = thetas[step]
            for pair in G2:
                i, j = pair[0], pair[1]
                two_qubit_block(reg[i], reg[j], 4.0 * theta)
            for quad in G4:
                i0, i1, i2, i3 = quad[0], quad[1], quad[2], quad[3]
                four_qubit_block(reg[i0], reg[i1], reg[i2], reg[i3], 8.0 * theta)

    return two_qubit_block, four_qubit_block, trotterized_circuit


if _CUDAQ_AVAILABLE:
    _, _, _trotterized_circuit = _define_kernels()
else:
    _trotterized_circuit = None


# ---------------------------------------------------------------------------
# MTS Loader
# ---------------------------------------------------------------------------


def _load_mts(repo_root: Path):
    """Load memetic_tabu_search from impl-mts/main.py."""
    import importlib.util

    mts_path = repo_root / "impl-mts" / "main.py"
    if not mts_path.exists():
        raise FileNotFoundError(f"MTS module not found: {mts_path}")
    spec = importlib.util.spec_from_file_location("mts_module", mts_path)
    mts_module = importlib.util.module_from_spec(spec)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    spec.loader.exec_module(mts_module)
    return mts_module


# ---------------------------------------------------------------------------
# Main Entry: run_hybrid
# ---------------------------------------------------------------------------


def run_hybrid(
    N: int,
    T: float = 1.0,
    n_steps: int = 1,
    shots: int = 200,
    population_size: int = 50,
    max_generations: int = 30,
    p_combine: float = 0.9,
    verbose: bool = False,
) -> tuple:
    """
    Run Trotterized counteradiabatic + MTS hybrid.
    Returns (best_sequence_as_list, time_sec).
    best_sequence is list of ±1 for eval_util compatibility.
    """
    start = time.perf_counter()

    if not _CUDAQ_AVAILABLE:
        raise RuntimeError("cudaq required for trotter method")

    import auxiliary_files.labs_utils as utils

    dt = T / n_steps
    G2, G4 = _get_interactions(N)
    thetas = []
    for step in range(1, n_steps + 1):
        t_val = step * dt
        theta_val = utils.compute_theta(t_val, dt, T, N, G2, G4)
        thetas.append(theta_val)

    quantum_result = cudaq.sample(
        _trotterized_circuit,
        N, G2, G4, n_steps, dt, T, thetas,
        shots_count=shots,
    )

    quantum_population = []
    for bitstring, count in quantum_result.items():
        seq = _bitstring_to_sequence(bitstring)
        for _ in range(count):
            quantum_population.append(seq.copy())

    mts_module = _load_mts(REPO_ROOT)
    random.seed(42)
    np.random.seed(42)
    best_s, best_energy, _ = mts_module.memetic_tabu_search(
        N,
        population_size=population_size,
        max_generations=max_generations,
        p_combine=p_combine,
        initial_population=quantum_population[:population_size] if quantum_population else None,
        verbose=verbose,
    )
    elapsed = time.perf_counter() - start
    seq_list = best_s.tolist() if hasattr(best_s, "tolist") else list(best_s)
    return seq_list, elapsed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int, nargs="?", default=10, help="Sequence length")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()
    seq, t = run_hybrid(args.N, verbose=args.verbose)
    print(f"N={args.N} best sequence (runlength): {seq}")
    print(f"Time: {t:.4f}s")
