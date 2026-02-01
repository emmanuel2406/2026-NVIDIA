"""
Hybrid QAOA+Grover+MTS for LABS (runnable from benchmarks).

Runs the main pipeline from 01_quantum_enhanced_optimization_LABS.ipynb:
1. QAOA+Grover sampling -> quantum seed distribution
2. MTS with that distribution as initial population

Usage:
    from hybrid import run_hybrid
    best_seq, time_sec = run_hybrid(N, verbose=False)
Returns best_seq as list of ±1 for eval_util compatibility.
"""

from pathlib import Path
import time
import random

try:
    import cudaq
    import numpy as np
    from math import floor
    _CUDAQ_AVAILABLE = True
except ImportError:
    _CUDAQ_AVAILABLE = False
    np = None


def get_interactions(N):
    """LABS Hamiltonian (Science Eq. 4): G2 = two-body z_i z_{i+2k}, G4 = four-body."""
    G2 = []
    G4 = []
    for i in range(N - 2):
        for k in range(1, int(floor((N - i) / 2)) + 1):
            j = i + 2 * k
            if j < N:
                G2.append([i, j])
    for i in range(N - 3):
        for t in range(1, int(floor((N - i - 1) / 2)) + 1):
            for k in range(t + 1, N - i - t):
                if i + k + t < N:
                    G4.append([i, i + t, i + k, i + k + t])
    return G2, G4


def bitstring_to_sequence(bitstring):
    """'0' -> +1, '1' -> -1."""
    return np.array([1 if b == "0" else -1 for b in bitstring])


def compute_energy(s):
    """Sidelobe energy E(s) = sum_{k=1}^{N-1} C_k^2."""
    N = len(s)
    return sum(
        (sum(s[i] * s[i + k] for i in range(N - k))) ** 2 for k in range(1, N)
    )


def get_fixed_parameters(p, N):
    """Fixed schedule: beta linear ramp, gamma rescaled by 1/N."""
    betas = [0.5 * np.pi * (l + 1) / (p + 1) for l in range(p)]
    gammas = [0.25 * np.pi * (l + 1) / ((p + 1) * N) for l in range(p)]
    return betas, gammas


def _define_kernels():
    """Define CUDA-Q kernels (must be at module level for cudaq)."""
    if not _CUDAQ_AVAILABLE:
        return None, None, None, None, None, None

    @cudaq.kernel
    def rzz_gate(q0: cudaq.qubit, q1: cudaq.qubit, theta: float):
        x.ctrl(q0, q1)
        rz(theta, q1)
        x.ctrl(q0, q1)

    @cudaq.kernel
    def apply_z(q: cudaq.qubit):
        z(q)

    @cudaq.kernel
    def grover_diffusion(qubits: cudaq.qview):
        N = qubits.size()
        h(qubits)
        for i in range(N):
            x(qubits[i])
        ctrl = qubits.front(N - 1)
        cudaq.control(apply_z, ctrl, qubits[N - 1])
        for i in range(N):
            x(qubits[i])
        h(qubits)

    @cudaq.kernel
    def grover_oracle(qubits: cudaq.qview, target_bits: list):
        N = qubits.size()
        for i in range(N):
            if target_bits[i] == 0:
                x(qubits[i])
        ctrl = qubits.front(N - 1)
        cudaq.control(apply_z, ctrl, qubits[N - 1])
        for i in range(N):
            if target_bits[i] == 0:
                x(qubits[i])

    @cudaq.kernel
    def qaoa_kernel(N: int, G2: list, G4: list, num_layers: int, betas: list, gammas: list):
        qubits = cudaq.qvector(N)
        h(qubits)
        for layer in range(num_layers):
            gamma = gammas[layer]
            beta = betas[layer]
            for term in G2:
                rzz_gate(qubits[term[0]], qubits[term[1]], 2.0 * gamma)
            for term in G4:
                i0, i1, i2, i3 = term[0], term[1], term[2], term[3]
                x.ctrl(qubits[i0], qubits[i1])
                x.ctrl(qubits[i1], qubits[i2])
                x.ctrl(qubits[i2], qubits[i3])
                rz(4.0 * gamma, qubits[i3])
                x.ctrl(qubits[i2], qubits[i3])
                x.ctrl(qubits[i1], qubits[i2])
                x.ctrl(qubits[i0], qubits[i1])
            for j in range(N):
                rx(2.0 * beta, qubits[j])

    @cudaq.kernel
    def qaoa_plus_grover_kernel(
        N: int,
        G2: list,
        G4: list,
        num_layers: int,
        betas: list,
        gammas: list,
        target_bits: list,
        num_grover_rounds: int,
    ):
        qubits = cudaq.qvector(N)
        h(qubits)
        for layer in range(num_layers):
            gamma = gammas[layer]
            beta = betas[layer]
            for term in G2:
                rzz_gate(qubits[term[0]], qubits[term[1]], 2.0 * gamma)
            for term in G4:
                i0, i1, i2, i3 = term[0], term[1], term[2], term[3]
                x.ctrl(qubits[i0], qubits[i1])
                x.ctrl(qubits[i1], qubits[i2])
                x.ctrl(qubits[i2], qubits[i3])
                rz(4.0 * gamma, qubits[i3])
                x.ctrl(qubits[i2], qubits[i3])
                x.ctrl(qubits[i1], qubits[i2])
                x.ctrl(qubits[i0], qubits[i1])
            for j in range(N):
                rx(2.0 * beta, qubits[j])
        for _ in range(num_grover_rounds):
            grover_oracle(qubits, target_bits)
            grover_diffusion(qubits)

    return rzz_gate, apply_z, grover_diffusion, grover_oracle, qaoa_kernel, qaoa_plus_grover_kernel


if _CUDAQ_AVAILABLE:
    _rzz_gate, _apply_z, _grover_diffusion, _grover_oracle, _qaoa_kernel, _qaoa_plus_grover_kernel = _define_kernels()
else:
    _qaoa_kernel = _qaoa_plus_grover_kernel = None


def run_qaoa(N, p=2, shots=500):
    """Run QAOA only; returns (samples, best_bs, best_e, best_m, expected_merit)."""
    if not _CUDAQ_AVAILABLE:
        raise RuntimeError("cudaq not available")
    G2, G4 = get_interactions(N)
    betas, gammas = get_fixed_parameters(p, N)
    result = cudaq.sample(_qaoa_kernel, N, G2, G4, p, betas, gammas, shots_count=shots)
    samples = dict(result.items())
    best_bs, best_e, best_m = None, float("inf"), 0.0
    total = sum(samples.values())
    expected_merit = 0.0
    for bs, count in samples.items():
        s = bitstring_to_sequence(bs)
        e = compute_energy(s)
        m = (N * N) / (2 * e) if e > 0 else float("inf")
        expected_merit += (count / total) * m
        if e < best_e:
            best_e, best_bs, best_m = e, bs, m
    return samples, best_bs, best_e, best_m, expected_merit


def run_qaoa_plus_grover(N, p=2, num_grover_rounds=2, target_bitstring=None, shots=500):
    """Run QAOA+Grover; returns (samples, best_bs, best_e, best_m, target_bitstring)."""
    if not _CUDAQ_AVAILABLE:
        raise RuntimeError("cudaq not available")
    G2, G4 = get_interactions(N)
    betas, gammas = get_fixed_parameters(p, N)
    if target_bitstring is None:
        _, best_bs, _, _, _ = run_qaoa(N, p, shots=min(100, shots))
        target_bitstring = best_bs
    target_bits = [1 if b == "1" else 0 for b in target_bitstring]
    result = cudaq.sample(
        _qaoa_plus_grover_kernel,
        N,
        G2,
        G4,
        p,
        betas,
        gammas,
        target_bits,
        num_grover_rounds,
        shots_count=shots,
    )
    samples = dict(result.items())
    best_bs, best_e, best_m = None, float("inf"), 0.0
    for bs, count in samples.items():
        s = bitstring_to_sequence(bs)
        e = compute_energy(s)
        m = (N * N) / (2 * e) if e > 0 else float("inf")
        if e < best_e:
            best_e, best_bs, best_m = e, bs, m
    return samples, best_bs, best_e, best_m, target_bitstring


def _load_mts(repo_root: Path):
    """Load memetic_tabu_search from impl-mts/main.py."""
    import importlib.util
    mts_path = repo_root / "impl-mts" / "main.py"
    if not mts_path.exists():
        raise FileNotFoundError(f"MTS module not found: {mts_path}")
    spec = importlib.util.spec_from_file_location("mts_module", mts_path)
    mts_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mts_module)
    return mts_module


def run_hybrid(
    N: int,
    p: int = 2,
    num_grover_rounds: int = 2,
    shots: int = 500,
    population_size: int = 50,
    max_generations: int = 30,
    p_combine: float = 0.9,
    verbose: bool = False,
) -> tuple:
    """
    Run QAOA+Grover+MTS hybrid. Returns (best_sequence_as_list, time_sec).
    best_sequence is list of ±1 for eval_util compatibility.
    """
    repo_root = Path(__file__).resolve().parent.parent
    start = time.perf_counter()

    if not _CUDAQ_AVAILABLE:
        raise RuntimeError("cudaq required for trotter (QAOA+Grover+MTS) method")

    # Quantum: QAOA+Grover
    samples, best_bs, best_e_q, best_m_q, target = run_qaoa_plus_grover(
        N, p, num_grover_rounds, target_bitstring=None, shots=shots
    )
    quantum_population = []
    for bs, count in samples.items():
        seq = bitstring_to_sequence(bs)
        for _ in range(count):
            quantum_population.append(seq.copy())

    # Classical: MTS
    mts_module = _load_mts(repo_root)
    memetic_tabu_search = mts_module.memetic_tabu_search
    random.seed(42)
    np.random.seed(42)
    best_s, best_energy, _ = memetic_tabu_search(
        N,
        population_size=population_size,
        max_generations=max_generations,
        p_combine=p_combine,
        initial_population=quantum_population[:population_size] if quantum_population else None,
        verbose=verbose,
    )
    elapsed = time.perf_counter() - start
    # Return as list of int for eval_util
    seq_list = best_s.tolist() if hasattr(best_s, "tolist") else list(best_s)
    return seq_list, elapsed
