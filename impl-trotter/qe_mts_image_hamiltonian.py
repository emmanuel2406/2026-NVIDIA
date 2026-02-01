"""
Quantum-Enhanced Memetic Tabu Search for LABS using Image Hamiltonian

This implementation combines:
1. Image Hamiltonian (H_f) from main.py - supports 1/2/3/4-body terms
2. dcqo_flexible_circuit_v2 for counteradiabatic quantum optimization
3. H100-optimized GPU MTS from impl-mts/mts_h100_optimized.py

Provides full comparison: Random baseline + Classical MTS + QE-MTS with detailed
timing, throughput metrics, and visualization.

Usage:
    python qe_mts_image_hamiltonian.py [N] [population_size] [max_generations] [shots] [trotter_steps]
    python qe_mts_image_hamiltonian.py 25 100 1000 500 10

Based on: "Scaling advantage with quantum-enhanced memetic tabu search for LABS"
https://arxiv.org/abs/2511.04553
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import cudaq
import matplotlib.pyplot as plt

# Add impl-trotter and impl-mts to path for imports (main adds auxiliary_files when loaded)
# impl-trotter must come before impl-mts so "from main import" resolves to impl-trotter/main.py
# (which has dcqo_flexible_circuit_v2, get_image_hamiltonian, etc.), not impl-mts/main.py
_impl_trotter_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_impl_trotter_dir.parent / "impl-mts"))
sys.path.insert(0, str(_impl_trotter_dir))

# Import H100-optimized MTS
try:
    from mts_h100_optimized import (
        memetic_tabu_search as mts_gpu,
        random_search,
        compute_energy,
        compute_merit_factor,
        compute_Ck,
        bitstring_to_sequence,
        sequence_to_bitstring,
        get_config,
    )
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("[WARNING] GPU MTS not available, using CPU fallback")

# Import quantum circuit and Hamiltonians from main (formerly generate_trotterization)
# First two qubits fixed as |1⟩|1⟩ (N even); skew-symmetry reduction (N odd).
from main import (
    dcqo_flexible_circuit_v2,
    get_image_hamiltonian,
    get_labs_hamiltonian,
    reduce_hamiltonian_fix_first_two,
    prepend_fixed_prefix_to_counts,
    get_image_hamiltonian_skew_reduced,
    expand_skew_symmetric_counts,
    FIXED_FIRST_TWO_PREFIX,
    r_z,
    r_zz,
    r_zzz,
    r_zzzz,
    r_yz,
    r_yzzz,
)

# Import labs_utils for theta computation
import auxiliary_files.labs_utils as labs_utils


# ============================================================================
# Configuration and Timing
# ============================================================================

@dataclass
class BenchmarkResult:
    """Stores benchmark results for a single method"""
    method_name: str
    best_energy: int
    best_merit: float
    best_sequence: str
    total_time_s: float
    samples_or_generations: int
    throughput: float  # samples/s or generations/s
    population_mean_energy: float
    population_min_energy: int
    population_max_energy: int
    initial_population_energies: List[int] = None  # For distribution plots
    final_population_energies: List[int] = None    # For distribution plots
    additional_metrics: Dict[str, Any] = None


@dataclass
class RunConfig:
    """Configuration for a benchmark run"""
    N: int
    population_size: int
    max_generations: int
    n_shots: int
    trotter_steps: int
    total_time: float
    p_combine: float
    seed: int
    output_dir: Path


# ============================================================================
# CPU Fallback MTS (when GPU not available)
# ============================================================================

def compute_all_Ck_cpu(s: np.ndarray) -> np.ndarray:
    """Compute all C_k for k = 1..N-1 (CPU version)"""
    N = len(s)
    Ck_values = np.zeros(N, dtype=np.int64)
    for k in range(1, N):
        Ck_values[k] = np.sum(s[:N-k] * s[k:])
    return Ck_values


def compute_delta_energy_cpu(s: np.ndarray, Ck_values: np.ndarray, flip_idx: int) -> int:
    """Change in energy if we flip bit at flip_idx (CPU version)"""
    N = len(s)
    delta = 0
    for k in range(1, N):
        old_Ck = Ck_values[k]
        delta_Ck = 0
        if flip_idx + k < N:
            delta_Ck += -2 * s[flip_idx] * s[flip_idx + k]
        if flip_idx - k >= 0:
            delta_Ck += -2 * s[flip_idx - k] * s[flip_idx]
        new_Ck = old_Ck + delta_Ck
        delta += new_Ck * new_Ck - old_Ck * old_Ck
    return delta


def update_Ck_after_flip_cpu(s: np.ndarray, Ck_values: np.ndarray, flip_idx: int):
    """Update C_k in place after flipping bit at flip_idx (CPU version)"""
    N = len(s)
    for k in range(1, N):
        if flip_idx + k < N:
            Ck_values[k] += 2 * s[flip_idx] * s[flip_idx + k]
        if flip_idx - k >= 0:
            Ck_values[k] += 2 * s[flip_idx - k] * s[flip_idx]


def tabu_search_cpu(s: np.ndarray, max_iter: int = None,
                    min_tabu_factor: float = 0.1,
                    max_tabu_factor: float = 0.12,
                    fixed_indices: List[int] = None) -> Tuple[np.ndarray, int]:
    """CPU tabu search (fallback when GPU not available). fixed_indices: indices that must not be flipped."""
    import random
    N = len(s)
    s = s.copy()
    fixed_set = set(fixed_indices) if fixed_indices else set()
    movable = [i for i in range(N) if i not in fixed_set]

    if max_iter is None:
        max_iter = random.randint(N // 2, 3 * N // 2)

    min_tabu = max(1, int(min_tabu_factor * max_iter))
    max_tabu = max(min_tabu + 1, int(max_tabu_factor * max_iter))

    tabu_list = np.zeros(N, dtype=np.int64)
    Ck_values = compute_all_Ck_cpu(s)
    current_energy = int(np.sum(Ck_values[1:]**2))

    best_s = s.copy()
    best_energy = current_energy

    for t in range(1, max_iter + 1):
        best_move = None
        best_move_energy = float('inf')

        for i in movable:
            delta = compute_delta_energy_cpu(s, Ck_values, i)
            new_energy = current_energy + delta

            is_tabu = tabu_list[i] >= t
            aspiration = new_energy < best_energy

            if (not is_tabu or aspiration) and new_energy < best_move_energy:
                best_move = i
                best_move_energy = new_energy

        if best_move is None and movable:
            best_move = random.choice(movable)
            delta = compute_delta_energy_cpu(s, Ck_values, best_move)
            best_move_energy = current_energy + delta
        elif best_move is None:
            break

        s[best_move] *= -1
        update_Ck_after_flip_cpu(s, Ck_values, best_move)
        current_energy = best_move_energy

        tenure = random.randint(min_tabu, max_tabu)
        tabu_list[best_move] = t + tenure

        if current_energy < best_energy:
            best_energy = current_energy
            best_s = s.copy()

    return best_s, best_energy


def memetic_tabu_search_cpu(N: int, population_size: int = 100,
                            max_generations: int = 1000,
                            p_combine: float = 0.9,
                            initial_population: List[np.ndarray] = None,
                            verbose: bool = True,
                            fixed_indices: List[int] = None,
                            fixed_values: np.ndarray = None) -> Tuple[np.ndarray, int, List[np.ndarray]]:
    """CPU memetic tabu search (fallback). fixed_indices/fixed_values: positions that must not be changed."""
    import random

    if verbose:
        print(f"[MTS-CPU] N={N}, pop={population_size}, gens={max_generations}")
    if fixed_indices is not None:
        if fixed_values is None or len(fixed_values) != len(fixed_indices):
            raise ValueError("fixed_values must be provided and match length of fixed_indices")
        fixed_values = np.asarray(fixed_values, dtype=np.int32)

    def _apply_fixed(s: np.ndarray) -> None:
        if fixed_indices is not None:
            for j, i in enumerate(fixed_indices):
                s[i] = fixed_values[j]

    if initial_population is not None:
        population = [seq.copy() for seq in initial_population[:population_size]]
        while len(population) < population_size:
            new_s = np.random.choice([-1, 1], size=N)
            _apply_fixed(new_s)
            population.append(new_s)
    else:
        population = []
        for _ in range(population_size):
            s = np.random.choice([-1, 1], size=N)
            _apply_fixed(s)
            population.append(s)

    energies = [compute_energy(s) for s in population]
    best_idx = np.argmin(energies)
    best_s = population[best_idx].copy()
    best_energy = energies[best_idx]

    start_time = time.time()
    fixed_set = set(fixed_indices) if fixed_indices else set()

    for gen in range(max_generations):
        if random.random() < p_combine:
            idx1, idx2 = random.sample(range(population_size), 2)
            k = random.randint(1, N - 1)
            child = np.concatenate([population[idx1][:k], population[idx2][k:]])
            if fixed_indices is not None:
                for j, i in enumerate(fixed_indices):
                    child[i] = population[idx1][i]
        else:
            idx = random.randint(0, population_size - 1)
            child = population[idx].copy()

        # Mutate (skip fixed indices)
        for i in range(N):
            if i not in fixed_set and random.random() < 1.0 / N:
                child[i] *= -1

        improved_child, child_energy = tabu_search_cpu(child, fixed_indices=fixed_indices)
        _apply_fixed(improved_child)

        if child_energy < best_energy:
            best_energy = child_energy
            best_s = improved_child.copy()
            if verbose:
                print(f"[MTS-CPU] Gen {gen}: NEW BEST E={best_energy}")

        replace_idx = random.randint(0, population_size - 1)
        population[replace_idx] = improved_child
        energies[replace_idx] = child_energy

        if verbose and gen % 50 == 0:
            elapsed = time.time() - start_time
            print(f"[MTS-CPU] Gen {gen}/{max_generations}: best_E={best_energy}, elapsed={elapsed:.1f}s")

    return best_s, best_energy, population


# ============================================================================
# Unified MTS Interface
# ============================================================================

def run_mts(N: int, population_size: int, max_generations: int,
            p_combine: float = 0.9, initial_population: List[np.ndarray] = None,
            verbose: bool = True,
            fixed_indices: List[int] = None,
            fixed_values: np.ndarray = None) -> Tuple[np.ndarray, int, List[np.ndarray]]:
    """Run MTS using GPU if available, otherwise CPU.
    fixed_indices/fixed_values: when set (e.g. [0,1] and [-1,-1] for truncated Hamiltonian),
    MTS will never change those positions."""
    if GPU_AVAILABLE:
        return mts_gpu(N, population_size, max_generations, p_combine,
                       initial_population, verbose=verbose,
                       fixed_indices=fixed_indices, fixed_values=fixed_values)
    else:
        return memetic_tabu_search_cpu(N, population_size, max_generations,
                                       p_combine, initial_population, verbose,
                                       fixed_indices=fixed_indices, fixed_values=fixed_values)


def run_mts_single(
    N: int,
    population_size: int = 50,
    max_generations: int = 100,
    p_combine: float = 0.9,
    seed: int = 42,
    verbose: bool = False,
) -> Tuple[List[int], float]:
    """
    Run MTS once (GPU if available, else CPU) and return (sequence_as_list, time_sec).
    For use by benchmarks/run_benchmark.py and eval_util. Sequence is list of ±1.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    if GPU_AVAILABLE:
        try:
            import cupy as cp
            cp.random.seed(seed)
        except Exception:
            pass
    start = time.perf_counter()
    best_s, best_energy, _ = run_mts(
        N=N,
        population_size=population_size,
        max_generations=max_generations,
        p_combine=p_combine,
        initial_population=None,
        verbose=verbose,
    )
    elapsed = time.perf_counter() - start
    seq_list = best_s.tolist() if hasattr(best_s, "tolist") else list(best_s)
    return seq_list, elapsed


def run_trotter_qe_single(
    N: int,
    n_shots: int = 500,
    trotter_steps: int = 10,
    total_time: float = 2.0,
    population_size: int = 50,
    max_generations: int = 100,
    p_combine: float = 0.9,
    seed: int = 42,
    verbose: bool = False,
) -> Tuple[List[int], float]:
    """
    Run QE-MTS (Image Hamiltonian): sample_quantum_population + MTS with fixed [0,1].
    Returns (best_sequence_as_list, total_time_sec). For use by benchmarks/run_benchmark.py.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    start = time.perf_counter()
    quantum_population, _ = sample_quantum_population(
        N=N,
        n_shots=n_shots,
        trotter_steps=trotter_steps,
        total_time=total_time,
        verbose=verbose,
    )
    initial_pop = quantum_population[:population_size]
    best_s, best_energy, _ = run_mts(
        N=N,
        population_size=population_size,
        max_generations=max_generations,
        p_combine=p_combine,
        initial_population=initial_pop,
        verbose=verbose,
        fixed_indices=[0, 1],
        fixed_values=np.array([-1, -1], dtype=np.int32),
    )
    elapsed = time.perf_counter() - start
    seq_list = best_s.tolist() if hasattr(best_s, "tolist") else list(best_s)
    return seq_list, elapsed


# ============================================================================
# Quantum Circuit Sampling with Image Hamiltonian
# ============================================================================

def get_interactions_from_image_hamiltonian(N: int) -> Tuple[List, List]:
    """
    Extract G2 and G4 interaction indices from the image Hamiltonian.
    Used for computing theta values via labs_utils.compute_theta.
    """
    t1, t2, t3, t4 = get_image_hamiltonian(N)

    # G2: 2-body interactions (extract indices only)
    G2 = [[int(term[0]), int(term[1])] for term in t2]

    # G4: 4-body interactions (extract indices only)
    G4 = [[int(term[0]), int(term[1]), int(term[2]), int(term[3])] for term in t4]

    return G2, G4


def sample_quantum_population(N: int, n_shots: int, trotter_steps: int,
                              total_time: float = 2.0,
                              verbose: bool = True) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Sample a population from the quantum circuit using the image Hamiltonian.

    Args:
        N: Sequence length
        n_shots: Number of quantum samples
        trotter_steps: Number of Trotter steps
        total_time: Total annealing time T
        verbose: Print progress

    Returns:
        Tuple of (population list, quantum metrics dict)
    """
    start_time = time.time()

    if N < 3:
        raise ValueError("sample_quantum_population requires N >= 3")

    use_skew = (N % 2 == 1)
    if use_skew:
        if verbose:
            print(f"[QUANTUM] Preparing circuit: N={N}, shots={n_shots}, steps={trotter_steps}, T={total_time} (skew-symmetry, {(N+1)//2} qubits)")
        t1r, t2r, t3r, t4r, num_qubits_circuit = get_image_hamiltonian_skew_reduced(N)
    else:
        if verbose:
            print(f"[QUANTUM] Preparing circuit: N={N}, shots={n_shots}, steps={trotter_steps}, T={total_time} (first two bits fixed as 11, N-2 qubits)")
        t1, t2, t3, t4 = get_image_hamiltonian(N)
        t1r, t2r, t3r, t4r = reduce_hamiltonian_fix_first_two(t1, t2, t3, t4)
        num_qubits_circuit = N - 2

    if verbose:
        print(f"[QUANTUM] Hamiltonian terms (reduced): {len(t1r)} (1-body), {len(t2r)} (2-body), "
              f"{len(t3r)} (3-body), {len(t4r)} (4-body)")

    t1_flat = [list(map(float, t)) for t in t1r]
    t2_flat = [list(map(float, t)) for t in t2r]
    t3_flat = [list(map(float, t)) for t in t3r]
    t4_flat = [list(map(float, t)) for t in t4r]

    dt = total_time / trotter_steps
    t_points = np.linspace(0, total_time, trotter_steps)
    lambda_sched = np.sin((np.pi / 2) * (t_points / total_time))**2
    lambda_dot_sched = (np.pi / total_time) * np.sin(np.pi * t_points / total_time) / 2.0

    if verbose:
        print(cudaq.draw(dcqo_flexible_circuit_v2, num_qubits_circuit, trotter_steps,
            t1_flat, t2_flat, t3_flat, t4_flat,
            lambda_sched.tolist(), lambda_dot_sched.tolist(), dt))

    circuit_start = time.time()

    result = cudaq.sample(
        dcqo_flexible_circuit_v2,
        num_qubits_circuit, trotter_steps,
        t1_flat, t2_flat, t3_flat, t4_flat,
        lambda_sched.tolist(), lambda_dot_sched.tolist(), dt,
        shots_count=n_shots
    )

    circuit_time = time.time() - circuit_start

    # Full bitstrings: skew expansion (N odd) or prepend "11" (N even)
    population = []
    energies = []
    if use_skew:
        full_counts = expand_skew_symmetric_counts(result, N)
        for full_bitstring, count in full_counts.items():
            seq = bitstring_to_sequence(full_bitstring)
            energy = compute_energy(seq)
            for _ in range(count):
                population.append(seq.copy())
                energies.append(energy)
        most_probable_full = max(full_counts.keys(), key=lambda k: full_counts[k]) if full_counts else ""
    else:
        for bitstring, count in result.items():
            full_bitstring = FIXED_FIRST_TWO_PREFIX + bitstring
            seq = bitstring_to_sequence(full_bitstring)
            energy = compute_energy(seq)
            for _ in range(count):
                population.append(seq.copy())
                energies.append(energy)
        most_probable_full = FIXED_FIRST_TWO_PREFIX + result.most_probable()

    total_time_elapsed = time.time() - start_time

    metrics = {
        "n_unique_bitstrings": len(result),
        "most_probable": most_probable_full,
        "circuit_time_s": circuit_time,
        "total_time_s": total_time_elapsed,
        "mean_energy": float(np.mean(energies)),
        "min_energy": int(np.min(energies)),
        "max_energy": int(np.max(energies)),
        "std_energy": float(np.std(energies)),
        "trotter_steps": trotter_steps,
        "annealing_time": total_time,
    }

    if verbose:
        print(f"[QUANTUM] Sampled {len(population)} sequences ({len(result)} unique)")
        print(f"[QUANTUM] Circuit execution: {circuit_time:.3f}s")
        print(f"[QUANTUM] Energy stats: mean={metrics['mean_energy']:.1f}, "
              f"min={metrics['min_energy']}, max={metrics['max_energy']}")

    return population, metrics

def sample_quantum_population_opt_labs(N: int, n_shots: int, trotter_steps: int,
                              total_time: float = 2.0,
                              verbose: bool = True) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Sample a population from the quantum circuit using the optimized labs Hamiltonian.

    Args:
        N: Sequence length
        n_shots: Number of quantum samples
        trotter_steps: Number of Trotter steps
        total_time: Total annealing time T
        verbose: Print progress

    Returns:
        Tuple of (population list, quantum metrics dict)
    """
    start_time = time.time()

    if N < 3:
        raise ValueError("sample_quantum_population requires N >= 3")

    use_skew = (N % 2 == 1)
    if use_skew:
        if verbose:
            print(f"[QUANTUM] Preparing LABS SKEW circuit: N={N}, shots={n_shots}, steps={trotter_steps}, T={total_time} (skew-symmetry, {(N+1)//2} qubits)")
        t1r, t2r, t3r, t4r, num_qubits_circuit = get_image_hamiltonian_skew_reduced(N)
    else:
        if verbose:
            print(f"[QUANTUM] Preparing LABS FIXED circuit: N={N}, shots={n_shots}, steps={trotter_steps}, T={total_time} (first two bits fixed as 11, N-2 qubits)")
        t1, t2, t3, t4 = get_labs_hamiltonian(N)
        t1r, t2r, t3r, t4r = reduce_hamiltonian_fix_first_two(t1, t2, t3, t4)
        num_qubits_circuit = N - 2

    if verbose:
        print(f"[QUANTUM] Hamiltonian terms (reduced): {len(t1r)} (1-body), {len(t2r)} (2-body), "
              f"{len(t3r)} (3-body), {len(t4r)} (4-body)")

    t1_flat = [list(map(float, t)) for t in t1r]
    t2_flat = [list(map(float, t)) for t in t2r]
    t3_flat = [list(map(float, t)) for t in t3r]
    t4_flat = [list(map(float, t)) for t in t4r]

    dt = total_time / trotter_steps
    t_points = np.linspace(0, total_time, trotter_steps)
    lambda_sched = np.sin((np.pi / 2) * (t_points / total_time))**2
    lambda_dot_sched = (np.pi / total_time) * np.sin(np.pi * t_points / total_time) / 2.0

    if verbose:
        print(cudaq.draw(dcqo_flexible_circuit_v2, num_qubits_circuit, trotter_steps,
            t1_flat, t2_flat, t3_flat, t4_flat,
            lambda_sched.tolist(), lambda_dot_sched.tolist(), dt))

    circuit_start = time.time()

    result = cudaq.sample(
        dcqo_flexible_circuit_v2,
        num_qubits_circuit, trotter_steps,
        t1_flat, t2_flat, t3_flat, t4_flat,
        lambda_sched.tolist(), lambda_dot_sched.tolist(), dt,
        shots_count=n_shots
    )

    circuit_time = time.time() - circuit_start

    # Full bitstrings: skew expansion (N odd) or prepend "11" (N even)
    population = []
    energies = []
    if use_skew:
        full_counts = expand_skew_symmetric_counts(result, N)
        for full_bitstring, count in full_counts.items():
            seq = bitstring_to_sequence(full_bitstring)
            energy = compute_energy(seq)
            for _ in range(count):
                population.append(seq.copy())
                energies.append(energy)
        most_probable_full = max(full_counts.keys(), key=lambda k: full_counts[k]) if full_counts else ""
    else:
        for bitstring, count in result.items():
            full_bitstring = FIXED_FIRST_TWO_PREFIX + bitstring
            seq = bitstring_to_sequence(full_bitstring)
            energy = compute_energy(seq)
            for _ in range(count):
                population.append(seq.copy())
                energies.append(energy)
        most_probable_full = FIXED_FIRST_TWO_PREFIX + result.most_probable()

    total_time_elapsed = time.time() - start_time

    metrics = {
        "n_unique_bitstrings": len(result),
        "most_probable": most_probable_full,
        "circuit_time_s": circuit_time,
        "total_time_s": total_time_elapsed,
        "mean_energy": float(np.mean(energies)),
        "min_energy": int(np.min(energies)),
        "max_energy": int(np.max(energies)),
        "std_energy": float(np.std(energies)),
        "trotter_steps": trotter_steps,
        "annealing_time": total_time,
    }

    if verbose:
        print(f"[QUANTUM] Sampled {len(population)} sequences ({len(result)} unique)")
        print(f"[QUANTUM] Circuit execution: {circuit_time:.3f}s")
        print(f"[QUANTUM] Energy stats: mean={metrics['mean_energy']:.1f}, "
              f"min={metrics['min_energy']}, max={metrics['max_energy']}")

    return population, metrics


def sample_quantum_population_labs(N: int, n_shots: int, trotter_steps: int,
                                   total_time: float = 2.0,
                                   verbose: bool = True) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Sample a population from the quantum circuit using the LABS Hamiltonian.

    Args:
        N: Sequence length
        n_shots: Number of quantum samples
        trotter_steps: Number of Trotter steps
        total_time: Total annealing time T
        verbose: Print progress

    Returns:
        Tuple of (population list, quantum metrics dict)
    """
    start_time = time.time()

    if verbose:
        print(f"[QUANTUM-LABS] Preparing circuit: N={N}, shots={n_shots}, steps={trotter_steps}, T={total_time}")

    # LABS Hamiltonian: full N qubits, no reduction
    t1, t2, t3, t4 = get_labs_hamiltonian(N)

    if verbose:
        print(f"[QUANTUM-LABS] Hamiltonian terms: {len(t1)} (1-body), {len(t2)} (2-body), "
              f"{len(t3)} (3-body), {len(t4)} (4-body)")

    t1_flat = [list(map(float, t)) for t in t1]
    t2_flat = [list(map(float, t)) for t in t2]
    t3_flat = [list(map(float, t)) for t in t3]
    t4_flat = [list(map(float, t)) for t in t4]

    dt = total_time / trotter_steps
    t_points = np.linspace(0, total_time, trotter_steps)
    lambda_sched = np.sin((np.pi / 2) * (t_points / total_time))**2
    lambda_dot_sched = (np.pi / total_time) * np.sin(np.pi * t_points / total_time) / 2.0

    circuit_start = time.time()

    result = cudaq.sample(
        dcqo_flexible_circuit_v2,
        N, trotter_steps,
        t1_flat, t2_flat, t3_flat, t4_flat,
        lambda_sched.tolist(), lambda_dot_sched.tolist(), dt,
        shots_count=n_shots
    )

    circuit_time = time.time() - circuit_start

    population = []
    energies = []
    for bitstring, count in result.items():
        seq = bitstring_to_sequence(bitstring)
        energy = compute_energy(seq)
        for _ in range(count):
            population.append(seq.copy())
            energies.append(energy)

    total_time_elapsed = time.time() - start_time

    metrics = {
        "n_unique_bitstrings": len(result),
        "most_probable": result.most_probable(),
        "circuit_time_s": circuit_time,
        "total_time_s": total_time_elapsed,
        "mean_energy": float(np.mean(energies)),
        "min_energy": int(np.min(energies)),
        "max_energy": int(np.max(energies)),
        "std_energy": float(np.std(energies)),
        "trotter_steps": trotter_steps,
        "annealing_time": total_time,
    }

    if verbose:
        print(f"[QUANTUM-LABS] Sampled {len(population)} sequences ({len(result)} unique)")
        print(f"[QUANTUM-LABS] Circuit execution: {circuit_time:.3f}s")
        print(f"[QUANTUM-LABS] Energy stats: mean={metrics['mean_energy']:.1f}, "
              f"min={metrics['min_energy']}, max={metrics['max_energy']}")

    return population, metrics


# ============================================================================
# Benchmark Methods
# ============================================================================

def benchmark_random_search(config: RunConfig, verbose: bool = True) -> BenchmarkResult:
    """Run and benchmark random search baseline"""
    if verbose:
        print("\n" + "=" * 70)
        print("METHOD 1: Random Search (Baseline)")
        print("=" * 70)

    start_time = time.time()

    population = [np.random.choice([-1, 1], size=config.N) for _ in range(config.n_shots)]
    energies = [compute_energy(s) for s in population]

    total_time = time.time() - start_time

    best_idx = np.argmin(energies)
    best_s = population[best_idx]
    best_energy = energies[best_idx]
    best_merit = compute_merit_factor(best_s, best_energy)

    if verbose:
        print(f"[RANDOM] Generated {config.n_shots} sequences in {total_time:.3f}s")
        print(f"[RANDOM] Best: E={best_energy}, F={best_merit:.4f}")

    return BenchmarkResult(
        method_name="Random Search",
        best_energy=int(best_energy),
        best_merit=float(best_merit),
        best_sequence=sequence_to_bitstring(best_s),
        total_time_s=total_time,
        samples_or_generations=config.n_shots,
        throughput=config.n_shots / total_time,
        population_mean_energy=float(np.mean(energies)),
        population_min_energy=int(np.min(energies)),
        population_max_energy=int(np.max(energies)),
        initial_population_energies=[int(e) for e in energies],  # Same as final for random
        final_population_energies=[int(e) for e in energies],
    )


def benchmark_classical_mts(config: RunConfig, verbose: bool = True) -> BenchmarkResult:
    """Run and benchmark classical MTS (random initialization)"""
    if verbose:
        print("\n" + "=" * 70)
        print("METHOD 2: Classical Memetic Tabu Search")
        print("=" * 70)

    # Generate initial random population to track its energy distribution
    initial_population = [np.random.choice([-1, 1], size=config.N)
                          for _ in range(config.population_size)]
    initial_energies = [compute_energy(s) for s in initial_population]

    if verbose:
        print(f"[MTS] Initial population: mean_E={np.mean(initial_energies):.1f}, "
              f"min_E={np.min(initial_energies)}, max_E={np.max(initial_energies)}")

    start_time = time.time()

    best_s, best_energy, population = run_mts(
        N=config.N,
        population_size=config.population_size,
        max_generations=config.max_generations,
        p_combine=config.p_combine,
        initial_population=initial_population,
        verbose=verbose
    )

    total_time = time.time() - start_time

    best_merit = compute_merit_factor(best_s, best_energy)
    final_energies = [compute_energy(s) for s in population]

    return BenchmarkResult(
        method_name="Classical MTS",
        best_energy=int(best_energy),
        best_merit=float(best_merit),
        best_sequence=sequence_to_bitstring(best_s),
        total_time_s=total_time,
        samples_or_generations=config.max_generations,
        throughput=config.max_generations / total_time,
        population_mean_energy=float(np.mean(final_energies)),
        population_min_energy=int(np.min(final_energies)),
        population_max_energy=int(np.max(final_energies)),
        initial_population_energies=[int(e) for e in initial_energies],
        final_population_energies=[int(e) for e in final_energies],
    )


def benchmark_quantum_enhanced_mts(config: RunConfig, verbose: bool = True) -> BenchmarkResult:
    """Run and benchmark quantum-enhanced MTS with Image Hamiltonian"""
    if verbose:
        print("\n" + "=" * 70)
        print("METHOD 3: Quantum-Enhanced MTS (Image Hamiltonian)")
        print("=" * 70)

    start_time = time.time()

    # Step 1: Sample quantum population
    quantum_population, quantum_metrics = sample_quantum_population(
        N=config.N,
        n_shots=config.n_shots,
        trotter_steps=config.trotter_steps,
        total_time=config.total_time,
        verbose=verbose
    )

    # Track initial (quantum) population energies
    initial_pop = quantum_population[:config.population_size]
    initial_energies = [compute_energy(s) for s in initial_pop]

    quantum_time = time.time() - start_time

    # Step 2: Run MTS with quantum-seeded population (first two bits fixed as 11 for truncated Hamiltonian)
    mts_start = time.time()

    best_s, best_energy, population = run_mts(
        N=config.N,
        population_size=config.population_size,
        max_generations=config.max_generations,
        p_combine=config.p_combine,
        initial_population=initial_pop,
        verbose=verbose,
        fixed_indices=[0, 1],
        fixed_values=np.array([-1, -1], dtype=np.int32),
    )

    mts_time = time.time() - mts_start
    total_time = time.time() - start_time

    best_merit = compute_merit_factor(best_s, best_energy)
    final_energies = [compute_energy(s) for s in population]

    return BenchmarkResult(
        method_name="QE-MTS (Image H)",
        best_energy=int(best_energy),
        best_merit=float(best_merit),
        best_sequence=sequence_to_bitstring(best_s),
        total_time_s=total_time,
        samples_or_generations=config.max_generations,
        throughput=config.max_generations / total_time,
        population_mean_energy=float(np.mean(final_energies)),
        population_min_energy=int(np.min(final_energies)),
        population_max_energy=int(np.max(final_energies)),
        initial_population_energies=[int(e) for e in initial_energies],
        final_population_energies=[int(e) for e in final_energies],
        additional_metrics={
            "quantum_sampling_time_s": quantum_time,
            "mts_time_s": mts_time,
            "quantum_initial_mean_energy": quantum_metrics["mean_energy"],
            "quantum_initial_min_energy": quantum_metrics["min_energy"],
            "n_unique_quantum_states": quantum_metrics["n_unique_bitstrings"],
            "circuit_execution_time_s": quantum_metrics["circuit_time_s"],
        }
    )


def benchmark_quantum_enhanced_mts_labs(config: RunConfig, verbose: bool = True) -> BenchmarkResult:
    """Run and benchmark quantum-enhanced MTS with LABS Hamiltonian"""
    if verbose:
        print("\n" + "=" * 70)
        print("METHOD 4: Quantum-Enhanced MTS (LABS Hamiltonian)")
        print("=" * 70)

    start_time = time.time()

    # Step 1: Sample quantum population using LABS Hamiltonian
    quantum_population, quantum_metrics = sample_quantum_population_labs(
        N=config.N,
        n_shots=config.n_shots,
        trotter_steps=config.trotter_steps,
        total_time=config.total_time,
        verbose=verbose
    )

    # Track initial (quantum) population energies
    initial_pop = quantum_population[:config.population_size]
    initial_energies = [compute_energy(s) for s in initial_pop]

    quantum_time = time.time() - start_time

    # Step 2: Run MTS with quantum-seeded population
    mts_start = time.time()

    best_s, best_energy, population = run_mts(
        N=config.N,
        population_size=config.population_size,
        max_generations=config.max_generations,
        p_combine=config.p_combine,
        initial_population=initial_pop,
        verbose=verbose
    )

    mts_time = time.time() - mts_start
    total_time = time.time() - start_time

    best_merit = compute_merit_factor(best_s, best_energy)
    final_energies = [compute_energy(s) for s in population]

    return BenchmarkResult(
        method_name="QE-MTS (LABS H)",
        best_energy=int(best_energy),
        best_merit=float(best_merit),
        best_sequence=sequence_to_bitstring(best_s),
        total_time_s=total_time,
        samples_or_generations=config.max_generations,
        throughput=config.max_generations / total_time,
        population_mean_energy=float(np.mean(final_energies)),
        population_min_energy=int(np.min(final_energies)),
        population_max_energy=int(np.max(final_energies)),
        initial_population_energies=[int(e) for e in initial_energies],
        final_population_energies=[int(e) for e in final_energies],
        additional_metrics={
            "quantum_sampling_time_s": quantum_time,
            "mts_time_s": mts_time,
            "quantum_initial_mean_energy": quantum_metrics["mean_energy"],
            "quantum_initial_min_energy": quantum_metrics["min_energy"],
            "n_unique_quantum_states": quantum_metrics["n_unique_bitstrings"],
            "circuit_execution_time_s": quantum_metrics["circuit_time_s"],
        }
    )

    

def benchmark_quantum_enhanced_mts_opt_labs(config: RunConfig, verbose: bool = True) -> BenchmarkResult:
    """Run and benchmark quantum-enhanced MTS with LABS Hamiltonian"""
    if verbose:
        print("\n" + "=" * 70)
        print("METHOD 4: Quantum-Enhanced MTS (LABS Hamiltonian)")
        print("=" * 70)

    start_time = time.time()

    # Step 1: Sample quantum population using LABS Hamiltonian
    quantum_population, quantum_metrics = sample_quantum_population_opt_labs(
        N=config.N,
        n_shots=config.n_shots,
        trotter_steps=config.trotter_steps,
        total_time=config.total_time,
        verbose=verbose
    )

    # Track initial (quantum) population energies
    initial_pop = quantum_population[:config.population_size]
    initial_energies = [compute_energy(s) for s in initial_pop]

    quantum_time = time.time() - start_time

    # Step 2: Run MTS with quantum-seeded population
    mts_start = time.time()

    best_s, best_energy, population = run_mts(
        N=config.N,
        population_size=config.population_size,
        max_generations=config.max_generations,
        p_combine=config.p_combine,
        initial_population=initial_pop,
        verbose=verbose
    )

    mts_time = time.time() - mts_start
    total_time = time.time() - start_time

    best_merit = compute_merit_factor(best_s, best_energy)
    final_energies = [compute_energy(s) for s in population]

    return BenchmarkResult(
        method_name="QE-MTS (LABS OPT H)",
        best_energy=int(best_energy),
        best_merit=float(best_merit),
        best_sequence=sequence_to_bitstring(best_s),
        total_time_s=total_time,
        samples_or_generations=config.max_generations,
        throughput=config.max_generations / total_time,
        population_mean_energy=float(np.mean(final_energies)),
        population_min_energy=int(np.min(final_energies)),
        population_max_energy=int(np.max(final_energies)),
        initial_population_energies=[int(e) for e in initial_energies],
        final_population_energies=[int(e) for e in final_energies],
        additional_metrics={
            "quantum_sampling_time_s": quantum_time,
            "mts_time_s": mts_time,
            "quantum_initial_mean_energy": quantum_metrics["mean_energy"],
            "quantum_initial_min_energy": quantum_metrics["min_energy"],
            "n_unique_quantum_states": quantum_metrics["n_unique_bitstrings"],
            "circuit_execution_time_s": quantum_metrics["circuit_time_s"],
        }
    )

# ============================================================================
# Visualization
# ============================================================================

def create_comparison_plot(results: List[BenchmarkResult], config: RunConfig,
                           output_path: Path):
    """Create comprehensive comparison visualization with initial population distributions"""
    n_methods = len(results)
    colors = ['gray', 'orange', 'dodgerblue', 'green', 'red'][:n_methods]

    # Create figure with 4 rows:
    # Row 1: Summary bar charts (Energy, Merit, Time)
    # Row 2: Initial population distributions
    # Row 3: Final population distributions
    # Row 4: Autocorrelations of best sequences
    fig = plt.figure(figsize=(4 * n_methods, 16))

    methods = [r.method_name for r in results]
    energies = [r.best_energy for r in results]
    merits = [r.best_merit for r in results]
    times = [r.total_time_s for r in results]

    # --- Row 1: Summary Bar Charts ---
    # Plot 1: Best Energy Comparison
    ax1 = fig.add_subplot(4, 3, 1)
    bars1 = ax1.bar(range(n_methods), energies, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Best Energy (lower is better)')
    ax1.set_title(f'Best Energy (N={config.N})')
    ax1.set_xticks(range(n_methods))
    ax1.set_xticklabels(methods, rotation=20, ha='right', fontsize=8)
    for bar, e in zip(bars1, energies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{e}', ha='center', va='bottom', fontsize=9)

    # Plot 2: Merit Factor Comparison
    ax2 = fig.add_subplot(4, 3, 2)
    bars2 = ax2.bar(range(n_methods), merits, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Merit Factor (higher is better)')
    ax2.set_title('Merit Factor')
    ax2.set_xticks(range(n_methods))
    ax2.set_xticklabels(methods, rotation=20, ha='right', fontsize=8)
    for bar, m in zip(bars2, merits):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{m:.3f}', ha='center', va='bottom', fontsize=9)

    # Plot 3: Execution Time
    ax3 = fig.add_subplot(4, 3, 3)
    bars3 = ax3.bar(range(n_methods), times, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Time (seconds)')
    ax3.set_title('Execution Time')
    ax3.set_xticks(range(n_methods))
    ax3.set_xticklabels(methods, rotation=20, ha='right', fontsize=8)
    for bar, t in zip(bars3, times):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{t:.1f}s', ha='center', va='bottom', fontsize=9)

    # --- Row 2: Initial Population Energy Distributions ---
    for idx, result in enumerate(results):
        ax = fig.add_subplot(4, n_methods, n_methods + 1 + idx)
        if result.initial_population_energies:
            init_energies = result.initial_population_energies
            ax.hist(init_energies, bins=20, alpha=0.7, color=colors[idx], edgecolor='black')
            ax.axvline(np.min(init_energies), color='red', linestyle='--', linewidth=1.5,
                       label=f'min={np.min(init_energies)}')
            ax.axvline(np.mean(init_energies), color='black', linestyle='-', linewidth=1.5,
                       label=f'mean={np.mean(init_energies):.0f}')
            ax.legend(fontsize=7, loc='upper right')
        ax.set_xlabel('Energy')
        ax.set_ylabel('Count')
        ax.set_title(f'{result.method_name}\nInitial Pop Distribution')

    # --- Row 3: Final Population Energy Distributions ---
    for idx, result in enumerate(results):
        ax = fig.add_subplot(4, n_methods, 2 * n_methods + 1 + idx)
        if result.final_population_energies:
            final_energies = result.final_population_energies
            ax.hist(final_energies, bins=20, alpha=0.7, color=colors[idx], edgecolor='black')
            ax.axvline(result.best_energy, color='red', linestyle='--', linewidth=1.5,
                       label=f'best={result.best_energy}')
            ax.axvline(np.mean(final_energies), color='black', linestyle='-', linewidth=1.5,
                       label=f'mean={np.mean(final_energies):.0f}')
            ax.legend(fontsize=7, loc='upper right')
        ax.set_xlabel('Energy')
        ax.set_ylabel('Count')
        ax.set_title(f'{result.method_name}\nFinal Pop Distribution')

    # --- Row 4: Autocorrelations of Best Sequences ---
    # for idx, result in enumerate(results):
    #     ax = fig.add_subplot(4, n_methods, 3 * n_methods + 1 + idx)
    #     seq = bitstring_to_sequence(result.best_sequence)
    #     N = len(seq)
    #     Ck_values = [compute_Ck(seq, k) for k in range(1, N)]

    #     # Color by magnitude
    #     bar_colors = [colors[idx] if c == 0 else ('green' if abs(c) <= 1 else 'red')
    #                   for c in Ck_values]
    #     ax.bar(range(1, N), Ck_values, color=bar_colors, alpha=0.7)
    #     ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    #     ax.set_xlabel('Lag k')
    #     ax.set_ylabel('C_k')
    #     ax.set_title(f'{result.method_name}\nE={result.best_energy}, F={result.best_merit:.3f}')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OUTPUT] Plot saved to: {output_path}")


# ============================================================================
# Output Generation
# ============================================================================

def save_results(results: List[BenchmarkResult], config: RunConfig):
    """Save all results to JSON, CSV, and PNG"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"qe_mts_N{config.N}_{timestamp}"

    # Ensure output directory exists
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = config.output_dir / f"{base_name}_results.json"
    json_data = {
        "config": {
            "N": config.N,
            "population_size": config.population_size,
            "max_generations": config.max_generations,
            "n_shots": config.n_shots,
            "trotter_steps": config.trotter_steps,
            "total_time": config.total_time,
            "p_combine": config.p_combine,
            "seed": config.seed,
            "timestamp": timestamp,
            "gpu_available": GPU_AVAILABLE,
        },
        "results": [asdict(r) for r in results]
    }

    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"[OUTPUT] JSON saved to: {json_path}")

    # Save CSV with per-method summary
    csv_path = config.output_dir / f"{base_name}_summary.csv"
    with open(csv_path, 'w') as f:
        f.write("method,best_energy,best_merit,time_s,throughput,pop_mean_energy\n")
        for r in results:
            f.write(f"{r.method_name},{r.best_energy},{r.best_merit:.6f},"
                    f"{r.total_time_s:.3f},{r.throughput:.2f},{r.population_mean_energy:.2f}\n")
    print(f"[OUTPUT] CSV saved to: {csv_path}")

    # Save visualization
    png_path = config.output_dir / f"{base_name}_comparison.png"
    create_comparison_plot(results, config, png_path)

    return json_path, csv_path, png_path


def print_final_summary(results: List[BenchmarkResult], config: RunConfig):
    """Print formatted final summary"""
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nConfiguration:")
    print(f"  N = {config.N}")
    print(f"  Population Size = {config.population_size}")
    print(f"  Max Generations = {config.max_generations}")
    print(f"  Quantum Shots = {config.n_shots}")
    print(f"  Trotter Steps = {config.trotter_steps}")
    print(f"  GPU Available = {GPU_AVAILABLE}")

    print("\n" + "-" * 70)
    print("Results by Method:")
    print("-" * 70)

    for r in results:
        print(f"\n{r.method_name}:")
        print(f"  Best Energy:   {r.best_energy}")
        print(f"  Best Merit:    {r.best_merit:.4f}")
        print(f"  Time:          {r.total_time_s:.2f}s")
        print(f"  Throughput:    {r.throughput:.2f} ops/s")
        print(f"  Sequence:      {r.best_sequence}")

        if r.additional_metrics:
            print(f"  Quantum Time:  {r.additional_metrics.get('quantum_sampling_time_s', 0):.2f}s")

    # Ranking
    sorted_results = sorted(results, key=lambda x: x.best_energy)

    print("\n" + "-" * 70)
    print("RANKING (by energy, lower is better):")
    for rank, r in enumerate(sorted_results, 1):
        print(f"  {rank}. {r.method_name}: E={r.best_energy}, F={r.best_merit:.4f}")

    # Improvements
    random_energy = results[0].best_energy

    print("\n" + "-" * 70)
    print("IMPROVEMENTS vs Random Baseline:")
    for r in results[1:]:
        improvement = random_energy - r.best_energy
        pct = 100 * improvement / random_energy if random_energy > 0 else 0
        print(f"  {r.method_name}: {improvement} energy units ({pct:.1f}%)")

    # Compare quantum methods if we have both
    if len(results) >= 4:
        mts_energy = results[1].best_energy
        qe_image_energy = results[2].best_energy
        qe_labs_energy = results[3].best_energy
        qe_labs_opt_energy = results[4].best_energy
        print("\n" + "-" * 70)
        print("QUANTUM vs CLASSICAL MTS:")
        print(f"  QE-MTS (Image) vs Classical MTS: {mts_energy - qe_image_energy} energy units")
        print(f"  QE-MTS (LABS) vs Classical MTS:  {mts_energy - qe_labs_energy} energy units")
        print(f"  QE-MTS (Image) vs QE-MTS (LABS): {qe_labs_energy - qe_image_energy} energy units")
        print(f"  QE-MTS (Image) vs QE-MTS (LABS OPT): {qe_labs_opt_energy - qe_image_energy} energy units")
    print("=" * 70)


# ============================================================================
# Main Entry Point
# ============================================================================

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Quantum-Enhanced MTS for LABS using Image Hamiltonian",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("N", type=int, nargs="?", default=25,
                        help="Sequence length")
    parser.add_argument("population_size", type=int, nargs="?", default=100,
                        help="MTS population size")
    parser.add_argument("max_generations", type=int, nargs="?", default=1000,
                        help="Maximum MTS generations")
    parser.add_argument("n_shots", type=int, nargs="?", default=500,
                        help="Number of quantum samples")
    parser.add_argument("trotter_steps", type=int, nargs="?", default=10,
                        help="Number of Trotter steps")

    parser.add_argument("--total-time", "-T", type=float, default=2.0,
                        help="Total annealing time")
    parser.add_argument("--p-combine", type=float, default=0.9,
                        help="Crossover probability")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output-dir", "-o", type=str, default="./results",
                        help="Output directory for results")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Reduce output verbosity")

    return parser.parse_args()


def main():
    """Main execution"""
    args = parse_args()
    verbose = not args.quiet

    # Set random seeds
    np.random.seed(args.seed)
    import random
    random.seed(args.seed)

    # Create configuration
    config = RunConfig(
        N=args.N,
        population_size=args.population_size,
        max_generations=args.max_generations,
        n_shots=args.n_shots,
        trotter_steps=args.trotter_steps,
        total_time=args.total_time,
        p_combine=args.p_combine,
        seed=args.seed,
        output_dir=Path(args.output_dir),
    )

    print("=" * 70)
    print("QUANTUM-ENHANCED MTS: IMAGE vs LABS HAMILTONIAN COMPARISON")
    print("=" * 70)
    print(f"Configuration: N={config.N}, pop={config.population_size}, "
          f"gens={config.max_generations}, shots={config.n_shots}, steps={config.trotter_steps}")

    if GPU_AVAILABLE:
        gpu_config = get_config()
        print(f"GPU: {gpu_config.device_name}")
    else:
        print("GPU: Not available (using CPU)")

    print("=" * 70)

    results = []

    # Reset seed before each method for fair comparison
    np.random.seed(config.seed)
    random.seed(config.seed)

    # Method 1: Random Search
    result_random = benchmark_random_search(config, verbose=verbose)
    results.append(result_random)

    # Method 2: Classical MTS
    np.random.seed(config.seed)
    random.seed(config.seed)
    result_mts = benchmark_classical_mts(config, verbose=verbose)
    results.append(result_mts)

    # Method 3: Quantum-Enhanced MTS (Image Hamiltonian)
    np.random.seed(config.seed)
    random.seed(config.seed)
    result_qe_image = benchmark_quantum_enhanced_mts(config, verbose=verbose)
    results.append(result_qe_image)

    # Method 4: Quantum-Enhanced MTS (LABS Hamiltonian)
    np.random.seed(config.seed)
    random.seed(config.seed)
    result_qe_labs = benchmark_quantum_enhanced_mts_labs(config, verbose=verbose)
    results.append(result_qe_labs)
    
    # Method 5: Quantum-Enhanced OPT MTS (LABS Hamiltonian)
    np.random.seed(config.seed)
    random.seed(config.seed)
    result_qe_opt_labs = benchmark_quantum_enhanced_mts_opt_labs(config, verbose=verbose)
    results.append(result_qe_opt_labs)

    # Save outputs
    save_results(results, config)

    # Print summary
    print_final_summary(results, config)

    return results


if __name__ == "__main__":
    main()
