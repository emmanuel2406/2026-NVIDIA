"""
LABS Optimization Comparison: Random Search vs MTS vs Quantum-Enhanced MTS

Compares three methods for solving the Low Autocorrelation Binary Sequences (LABS) problem:
1. Random Search (baseline)
2. Memetic Tabu Search (MTS)
3. Quantum-Enhanced MTS using counteradiabatic optimization

Based on: "Scaling advantage with quantum-enhanced memetic tabu search for LABS"
https://arxiv.org/abs/2511.04553
"""

import cudaq
import numpy as np
from math import floor
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import random
import time
import auxiliary_files.labs_utils as utils


# ============================================================================
# LABS Energy and Merit Factor Computation
# ============================================================================

def compute_Ck(s: np.ndarray, k: int) -> int:
    """
    Compute C_k for a binary sequence.
    """
    N = len(s)
    return np.sum(s[:N-k] * s[k:])

def compute_energy(s: np.ndarray) -> int:
    N = len(s)
    energy = 0
    for k in range(1, N):
        Ck = compute_Ck(s, k)
        energy += Ck * Ck
    return energy

def compute_merit_factor(s: np.ndarray, energy: int = None) -> float:
    N = len(s)
    if energy is None:
        energy = compute_energy(s)

    if energy == 0:
        return float('inf')  # Perfect sequence (only possible for very small N)

    return (N * N) / (2.0 * energy)

def energy_and_merit(s: np.ndarray) -> Tuple[int, float]:
    energy = compute_energy(s)
    merit = compute_merit_factor(s, energy)
    return energy, merit

def bitstring_to_sequence(bitstring: str) -> np.ndarray:
    """Convert a '0'/'1' bitstring to a +1/-1 sequence."""
    return np.array([1 if b == '0' else -1 for b in bitstring])

def sequence_to_bitstring(s: np.ndarray) -> str:
    """Convert a +1/-1 sequence to a '0'/'1' bitstring."""
    return ''.join(['0' if x == 1 else '1' for x in s])

# O(N) Optimization with computing all C_k
def compute_all_Ck(s: np.ndarray) -> np.ndarray:
    """
    Compute all C_k values for k = 1 to N-1.
    """
    N = len(s)
    Ck_values = np.zeros(N, dtype=np.int64)
    for k in range(1, N):
        Ck_values[k] = compute_Ck(s, k)
    return Ck_values

def compute_delta_energy(s: np.ndarray, Ck_values: np.ndarray, flip_idx: int) -> int:
    """
    Compute the change in energy if we flip bit at flip_idx.
    """
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

def update_Ck_after_flip(s: np.ndarray, Ck_values: np.ndarray, flip_idx: int):
    """
    Update C_k values after flipping bit at flip_idx (in-place).
    Note: s should already have been flipped.

    Args:
        s: Binary sequence (already flipped at flip_idx)
        Ck_values: C_k values to update in place
        flip_idx: Index that was flipped
    """
    N = len(s)
    for k in range(1, N):
        if flip_idx + k < N:
            Ck_values[k] += 2 * s[flip_idx] * s[flip_idx + k]
        if flip_idx - k >= 0:
            Ck_values[k] += 2 * s[flip_idx - k] * s[flip_idx]

# ============================================================================
# Tabu Search Implementation
# ============================================================================

def tabu_search(s: np.ndarray, max_iter: int = None,
                min_tabu_factor: float = 0.1,
                max_tabu_factor: float = 0.12,
                tabu_id: int = None) -> Tuple[np.ndarray, int]:
    """
    Perform tabu search starting from sequence s.

    Args:
        s: Initial binary sequence
        max_iter: Maximum iterations (default: random in [N/2, 3N/2])
        min_tabu_factor: Minimum tabu tenure as fraction of max_iter
        max_tabu_factor: Maximum tabu tenure as fraction of max_iter
        tabu_id: Optional ID for logging purposes

    Returns:
        Tuple of (best_sequence, best_energy)
    """
    N = len(s)
    s = s.copy()
    prefix = f"[TABU-{tabu_id}]" if tabu_id is not None else "[TABU]"

    # Set max iterations if not specified
    if max_iter is None:
        max_iter = random.randint(N // 2, 3 * N // 2)

    # Compute tabu tenure bounds
    min_tabu = max(1, int(min_tabu_factor * max_iter))
    max_tabu = max(min_tabu + 1, int(max_tabu_factor * max_iter))

    print(f"{prefix} Starting tabu search: N={N}, max_iter={max_iter}, "
          f"tenure_range=[{min_tabu}, {max_tabu}]")

    # Initialize tabu list
    tabu_list = np.zeros(N, dtype=np.int64)

    # Initialize C_k values for incremental updates
    Ck_values = compute_all_Ck(s)
    current_energy = np.sum(Ck_values[1:]**2)

    # Track best solution found
    best_s = s.copy()
    best_energy = current_energy

    init_merit = compute_merit_factor(s, current_energy)
    print(f"{prefix} Initial: energy={current_energy}, merit={init_merit:.4f}")

    improvements = 0
    aspiration_used = 0

    for t in range(1, max_iter + 1):
        best_move = None
        best_move_energy = float('inf')
        best_move_delta = float('inf')
        used_aspiration = False

        # Evaluate all possible single-bit flips
        for i in range(N):
            delta = compute_delta_energy(s, Ck_values, i)
            new_energy = current_energy + delta

            is_tabu = (tabu_list[i] >= t)
            aspiration = (new_energy < best_energy)

            if (not is_tabu or aspiration) and new_energy < best_move_energy:
                best_move = i
                best_move_energy = new_energy
                best_move_delta = delta
                if is_tabu and aspiration:
                    used_aspiration = True

        # If no valid move found, pick any move
        if best_move is None:
            best_move = random.randint(0, N - 1)
            best_move_delta = compute_delta_energy(s, Ck_values, best_move)
            best_move_energy = current_energy + best_move_delta

        if used_aspiration:
            aspiration_used += 1

        # Execute the best move
        s[best_move] *= -1
        update_Ck_after_flip(s, Ck_values, best_move)
        current_energy = best_move_energy

        # Update tabu list with random tenure
        tenure = random.randint(min_tabu, max_tabu)
        tabu_list[best_move] = t + tenure

        # Update best solution if improved
        if current_energy < best_energy:
            best_energy = current_energy
            best_s = s.copy()
            improvements += 1
            merit = compute_merit_factor(s, current_energy)
            print(f"{prefix} Iter {t}/{max_iter}: New best energy={best_energy}, "
                  f"merit={merit:.4f} (flipped bit {best_move})")

    final_merit = compute_merit_factor(best_s, best_energy)
    print(f"{prefix} Completed: energy={best_energy}, merit={final_merit:.4f}, "
          f"improvements={improvements}, aspiration_moves={aspiration_used}")

    return best_s, best_energy

# ============================================================================
# Combine (Crossover) and Mutate Functions
# ============================================================================

def combine(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    """
    Single-point crossover: combine two parent sequences.

    Args:
        parent1: First parent sequence
        parent2: Second parent sequence

    Returns:
        Child sequence
    """
    N = len(parent1)
    k = random.randint(1, N - 1)
    child = np.concatenate([parent1[:k], parent2[k:]])
    print(f"[COMBINE] Cut point k={k}, child inherits bits [0:{k}) from P1, [{k}:{N}) from P2")

    return child

def mutate(s: np.ndarray, p_mut: float = None) -> np.ndarray:
    """
    Mutate a sequence by flipping each bit independently with probability p_mut.

    Args:
        s: Binary sequence
        p_mut: Mutation probability per bit (default: 1/N)

    Returns:
        Mutated sequence (copy)
    """
    N = len(s)
    if p_mut is None:
        p_mut = 1.0 / N

    child = s.copy()
    flipped_positions = []

    for i in range(N):
        if random.random() < p_mut:
            child[i] *= -1
            flipped_positions.append(i)

    print(f"[MUTATE] p_mut={p_mut:.4f}, flipped {len(flipped_positions)} bits: {flipped_positions}")

    return child

# ============================================================================
# Memetic Tabu Search (MTS) Algorithm
# ============================================================================

def memetic_tabu_search(N: int,
                        population_size: int = 100,
                        max_generations: int = 1000,
                        p_combine: float = 0.9,
                        initial_population: List[np.ndarray] = None,
                        target_energy: int = None) -> Tuple[np.ndarray, int, List[np.ndarray]]:
    """
    Memetic Tabu Search (MTS) for the LABS problem.

    Args:
        N: Sequence length
        population_size: Number of individuals in population (K=100 in paper)
        max_generations: Maximum number of generations
        p_combine: Probability of using crossover (0.9 in paper)
        initial_population: Optional list of initial sequences
        target_energy: Optional target energy to stop early

    Returns:
        Tuple of (best_sequence, best_energy, final_population)
    """
    print("=" * 70)
    print("[MTS] MEMETIC TABU SEARCH - INITIALIZATION")
    print("=" * 70)
    print(f"[MTS] Parameters:")
    print(f"      - Sequence length N: {N}")
    print(f"      - Population size: {population_size}")
    print(f"      - Max generations: {max_generations}")
    print(f"      - Crossover probability (p_combine): {p_combine}")
    print(f"      - Mutation probability (p_mut): {1.0/N:.4f} (= 1/N)")
    print(f"      - Target energy: {target_energy if target_energy else 'None (run until max_generations)'}")

    # Initialize population
    if initial_population is not None:
        print(f"[MTS] Using provided initial population of {len(initial_population)} sequences")
        population = [seq.copy() for seq in initial_population]
        while len(population) < population_size:
            population.append(np.random.choice([-1, 1], size=N))
        population = population[:population_size]
        print(f"[MTS] Population adjusted to size {len(population)}")
    else:
        print(f"[MTS] Generating random initial population of {population_size} sequences")
        population = [np.random.choice([-1, 1], size=N) for _ in range(population_size)]

    # Compute energies and merit factors for initial population
    print(f"[MTS] Computing initial population energies and merit factors...")
    energies = [compute_energy(s) for s in population]
    merits = [compute_merit_factor(s, e) for s, e in zip(population, energies)]

    # Log initial population statistics
    print(f"[MTS] Initial population statistics:")
    print(f"      - Energy:  min={np.min(energies)}, max={np.max(energies)}, "
          f"mean={np.mean(energies):.2f}, std={np.std(energies):.2f}")
    print(f"      - Merit:   min={np.min(merits):.4f}, max={np.max(merits):.4f}, "
          f"mean={np.mean(merits):.4f}, std={np.std(merits):.4f}")

    # Find initial best
    best_idx = np.argmin(energies)
    best_s = population[best_idx].copy()
    best_energy = energies[best_idx]
    best_merit = merits[best_idx]

    print(f"[MTS] Initial best: index={best_idx}, energy={best_energy}, merit={best_merit:.4f}")
    print(f"[MTS] Initial best sequence: {sequence_to_bitstring(best_s)}")
    print("=" * 70)
    print("[MTS] STARTING EVOLUTION")
    print("=" * 70)

    start_time = time.time()
    improvements_count = 0
    crossover_count = 0
    selection_count = 0

    for gen in range(max_generations):
        # Check stopping criterion
        if target_energy is not None and best_energy <= target_energy:
            print(f"[MTS] TARGET REACHED at generation {gen}!")
            print(f"      Energy {best_energy} <= {target_energy}, Merit={best_merit:.4f}")
            break

        # Decide crossover vs selection
        if random.random() < p_combine:
            idx1, idx2 = random.sample(range(population_size), 2)
            child = combine(population[idx1], population[idx2])
            crossover_count += 1
            operation = f"crossover(P[{idx1}], P[{idx2}])"
        else:
            idx = random.randint(0, population_size - 1)
            child = population[idx].copy()
            selection_count += 1
            operation = f"select(P[{idx}])"

        # Mutate
        child = mutate(child)

        # Run tabu search
        improved_child, child_energy = tabu_search(
            child, tabu_id=gen
        )
        child_merit = compute_merit_factor(improved_child, child_energy)

        # Update best if improved
        if child_energy < best_energy:
            old_best = best_energy
            old_merit = best_merit
            best_energy = child_energy
            best_merit = child_merit
            best_s = improved_child.copy()
            improvements_count += 1
            print(f"[MTS] Gen {gen}: NEW BEST! energy: {old_best}->{best_energy}, "
                  f"merit: {old_merit:.4f}->{best_merit:.4f} via {operation}")

        # Replace random population member
        replace_idx = random.randint(0, population_size - 1)
        population[replace_idx] = improved_child
        energies[replace_idx] = child_energy
        merits[replace_idx] = child_merit
        elapsed = time.time() - start_time
        print(f"[MTS] Gen {gen+1}/{max_generations}: best_energy={best_energy}, "
              f"best_merit={best_merit:.4f}, pop_mean_energy={np.mean(energies):.1f}, "
              f"pop_mean_merit={np.mean(merits):.4f}, elapsed={elapsed:.1f}s")

    total_time = time.time() - start_time

    final_merits = [compute_merit_factor(s, e) for s, e in zip(population, energies)]

    print("=" * 70)
    print("[MTS] EVOLUTION COMPLETE")
    print("=" * 70)
    print(f"[MTS] Final Results:")
    print(f"      - Best energy: {best_energy}")
    print(f"      - Best merit factor: {best_merit:.4f}")
    print(f"      - Best sequence: {sequence_to_bitstring(best_s)}")
    print(f"      - Total generations: {gen + 1}")
    print(f"      - Total improvements: {improvements_count}")
    print(f"      - Crossover operations: {crossover_count}")
    print(f"      - Selection operations: {selection_count}")
    print(f"      - Total time: {total_time:.2f} seconds")
    print(f"      - Time per generation: {total_time/(gen+1)*1000:.2f} ms")
    print(f"[MTS] Final population statistics:")
    print(f"      - Energy:  min={np.min(energies)}, max={np.max(energies)}, "
          f"mean={np.mean(energies):.2f}, std={np.std(energies):.2f}")
    print(f"      - Merit:   min={np.min(final_merits):.4f}, max={np.max(final_merits):.4f}, "
          f"mean={np.mean(final_merits):.4f}, std={np.std(final_merits):.4f}")
    print("=" * 70)

    return best_s, best_energy, population

# ============================================================================
# Visualization Functions
# ============================================================================

def visualize_mts_results(best_s: np.ndarray, best_energy: int,
                          population: List[np.ndarray], title_prefix: str = "MTS"):
    """
    Visualize MTS results including:
    - Energy distribution of final population
    - Merit factor distribution of final population
    - Autocorrelation of best sequence
    - Best sequence visualization
    """
    N = len(best_s)
    pop_energies = [compute_energy(s) for s in population]
    pop_merits = [compute_merit_factor(s, e) for s, e in zip(population, pop_energies)]
    best_merit = compute_merit_factor(best_s, best_energy)

    print(f"\n[VIZ] Generating visualization for {title_prefix}...")
    print(f"[VIZ] Sequence length N={N}")
    print(f"[VIZ] Best energy={best_energy}, Best merit={best_merit:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Energy distribution
    ax1 = axes[0, 0]
    ax1.hist(pop_energies, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(best_energy, color='red', linestyle='--', linewidth=2,
                label=f'Best: {best_energy}')
    ax1.set_xlabel('Energy (lower is better)')
    ax1.set_ylabel('Count')
    ax1.set_title(f'{title_prefix}: Population Energy Distribution (N={N})')
    ax1.legend()

    # Plot 2: Merit factor distribution
    ax2 = axes[0, 1]
    ax2.hist(pop_merits, bins=30, edgecolor='black', alpha=0.7, color='forestgreen')
    ax2.axvline(best_merit, color='red', linestyle='--', linewidth=2,
                label=f'Best: {best_merit:.4f}')
    ax2.set_xlabel('Merit Factor (higher is better)')
    ax2.set_ylabel('Count')
    ax2.set_title(f'{title_prefix}: Population Merit Factor Distribution (N={N})')
    ax2.legend()

    # Plot 3: Autocorrelation
    ax3 = axes[1, 0]
    Ck_values = [compute_Ck(best_s, k) for k in range(1, N)]
    colors_ck = ['steelblue' if c == 0 else ('green' if abs(c) <= 1 else 'orange') for c in Ck_values]
    ax3.bar(range(1, N), Ck_values, color=colors_ck, alpha=0.7)
    ax3.set_xlabel('Lag k')
    ax3.set_ylabel('C_k')
    ax3.set_title(f'{title_prefix}: Autocorrelation of Best Sequence (E={best_energy}, F={best_merit:.4f})')
    ax3.axhline(0, color='black', linestyle='-', linewidth=0.5)

    # Plot 4: Best sequence
    ax4 = axes[1, 1]
    colors = ['green' if x == 1 else 'red' for x in best_s]
    ax4.bar(range(N), best_s, color=colors, alpha=0.7)
    ax4.set_xlabel('Position')
    ax4.set_ylabel('Value (+1/-1)')
    ax4.set_title(f'{title_prefix}: Best Sequence')
    ax4.set_ylim(-1.5, 1.5)

    textstr = f'N={N}\nEnergy={best_energy}\nMerit={best_merit:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax4.text(0.02, 0.98, textstr, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.show()

    print(f"\n[VIZ] {title_prefix} Results Summary:")
    print(f"      - Sequence length N: {N}")
    print(f"      - Best energy: {best_energy}")
    print(f"      - Best merit factor: {best_merit:.4f}")
    print(f"      - Population size: {len(population)}")
    print(f"      - Population energy:  mean={np.mean(pop_energies):.2f}, "
          f"min={np.min(pop_energies)}, max={np.max(pop_energies)}, std={np.std(pop_energies):.2f}")
    print(f"      - Population merit:   mean={np.mean(pop_merits):.4f}, "
          f"min={np.min(pop_merits):.4f}, max={np.max(pop_merits):.4f}, std={np.std(pop_merits):.4f}")

# Global configuration
SEED = 42

# ============================================================================
# Quantum Circuit Components (Counteradiabatic Optimization)
# ============================================================================

# Helper kernel for RZZ gate: exp(-i * theta/2 * Z⊗Z)
@cudaq.kernel
def rzz_gate(q0: cudaq.qubit, q1: cudaq.qubit, theta: float):
    """RZZ gate decomposed into CNOT and RZ"""
    x.ctrl(q0, q1)
    rz(theta, q1)
    x.ctrl(q0, q1)

# 2-qubit block implementing R_YZ(theta) * R_ZY(theta)
# From Fig. 3: requires 2 entangling RZZ gates and 4 single-qubit rotations
@cudaq.kernel
def two_qubit_block(q0: cudaq.qubit, q1: cudaq.qubit, theta: float):
    """
    Implements the combined R_YZ(theta) and R_ZY(theta) rotations.
    Circuit from Fig. 3 of the paper.
    """
    # First RZZ(theta)
    x.ctrl(q0, q1)
    rz(theta, q1)
    x.ctrl(q0, q1)

    # Rx(pi/2) on both qubits
    rx(np.pi/2, q0)
    rx(np.pi/2, q1)

    # Second RZZ(theta)
    x.ctrl(q0, q1)
    rz(theta, q1)
    x.ctrl(q0, q1)

    # Rx†(pi/2) = Rx(-pi/2) on both qubits
    rx(-np.pi/2, q0)
    rx(-np.pi/2, q1)

# 4-qubit block implementing R_YZZZ(theta) * R_ZYZZ(theta) * R_ZZYZ(theta) * R_ZZZY(theta)
# From Fig. 4: requires 10 entangling RZZ gates and 28 single-qubit rotations
@cudaq.kernel
def four_qubit_block(q0: cudaq.qubit, q1: cudaq.qubit, q2: cudaq.qubit, q3: cudaq.qubit, theta: float):
    """
    Implements the combined 4-qubit Pauli rotations R_YZZZ, R_ZYZZ, R_ZZYZ, R_ZZZY.
    Circuit from Fig. 4 of the paper.
    Qubits q0, q1, q2, q3 correspond to indices i, i+t, i+k, i+k+t in the formula.
    """
    # Column 1: Initial single-qubit rotations
    rx(-np.pi/2, q0)   # Rx†(π/2)
    ry(np.pi/2, q1)    # Ry(π/2)
    ry(-np.pi/2, q2)   # Ry†(π/2)
    rx(-np.pi/2, q3)   # Rx†(π/2)

    # Column 2: RZZ†(π/2) = RZZ(-π/2) between q0-q1 and q2-q3
    x.ctrl(q0, q1)
    rz(-np.pi/2, q1)
    x.ctrl(q0, q1)

    x.ctrl(q2, q3)
    rz(-np.pi/2, q3)
    x.ctrl(q2, q3)

    # Column 3: Single-qubit gates on q1 and q2
    ry(-np.pi/2, q1)   # Ry†(π/2)
    rx(-np.pi/2, q1)   # Rx†(π/2)
    ry(np.pi/2, q2)    # Ry(π/2)
    rx(-np.pi/2, q2)   # Rx†(π/2)

    # Column 4: RZZ(theta) between q1-q2
    x.ctrl(q1, q2)
    rz(theta, q2)
    x.ctrl(q1, q2)

    # Column 5: Single-qubit gates on q1 and q2
    rx(np.pi/2, q1)    # Rx(π/2)
    ry(np.pi/2, q1)    # Ry(π/2)
    rx(np.pi/2, q2)    # Rx(π/2)
    ry(-np.pi/2, q2)   # Ry†(π/2)

    # Column 6: RZZ(π/2) between q0-q1 and q2-q3
    x.ctrl(q0, q1)
    rz(np.pi/2, q1)
    x.ctrl(q0, q1)

    x.ctrl(q2, q3)
    rz(np.pi/2, q3)
    x.ctrl(q2, q3)

    # Column 7: Single-qubit rotations
    rx(np.pi/2, q0)    # Rx(π/2)
    ry(-np.pi/2, q1)   # Ry†(π/2)
    rx(-np.pi/2, q1)   # Rx†(π/2)
    ry(np.pi/2, q2)    # Ry(π/2)
    rx(-np.pi/2, q2)   # Rx†(π/2)
    rx(np.pi/2, q3)    # Rx(π/2)

    # Column 8: RZZ†(π/2) on q0-q1, RZZ(theta) on q1-q2, RZZ†(π/2) on q2-q3
    x.ctrl(q0, q1)
    rz(-np.pi/2, q1)
    x.ctrl(q0, q1)

    x.ctrl(q1, q2)
    rz(theta, q2)
    x.ctrl(q1, q2)

    x.ctrl(q2, q3)
    rz(-np.pi/2, q3)
    x.ctrl(q2, q3)

    # Column 9: Single-qubit rotations
    rx(-np.pi, q0)     # Rx†(π)
    rx(np.pi/2, q1)    # Rx(π/2)
    ry(np.pi/2, q1)    # Ry(π/2)
    rx(np.pi/2, q2)    # Rx(π/2)
    ry(-np.pi/2, q2)   # Ry†(π/2)
    rx(-np.pi, q3)     # Rx†(π)

    # Column 10: RZZ†(π/2) on q0-q1, RZZ(theta) on q1-q2, RZZ†(π/2) on q2-q3
    x.ctrl(q0, q1)
    rz(-np.pi/2, q1)
    x.ctrl(q0, q1)

    x.ctrl(q1, q2)
    rz(theta, q2)
    x.ctrl(q1, q2)

    x.ctrl(q2, q3)
    rz(-np.pi/2, q3)
    x.ctrl(q2, q3)

    # Column 11: Single-qubit rotations
    rx(-np.pi/2, q0)   # Rx†(π/2)
    ry(-np.pi/2, q1)   # Ry†(π/2)
    ry(np.pi/2, q2)    # Ry(π/2)
    rx(-np.pi/2, q3)   # Rx†(π/2)

    # Column 12: RZZ(π/2) between q0-q1 and q2-q3
    x.ctrl(q0, q1)
    rz(np.pi/2, q1)
    x.ctrl(q0, q1)

    x.ctrl(q2, q3)
    rz(np.pi/2, q3)
    x.ctrl(q2, q3)

    # Column 13: Final single-qubit rotations
    ry(-np.pi/2, q1)   # Ry†(π/2)
    rx(-np.pi/2, q1)   # Rx†(π/2)
    ry(np.pi/2, q2)    # Ry(π/2)
    rx(-np.pi/2, q2)   # Rx†(π/2)

    # Column 14: RZZ(theta) between q1-q2
    x.ctrl(q1, q2)
    rz(theta, q2)
    x.ctrl(q1, q2)

    # Column 15: Final single-qubit rotations
    rx(np.pi/2, q1)    # Rx(π/2)
    ry(np.pi/2, q1)    # Ry(π/2)
    rx(np.pi/2, q2)    # Rx(π/2)
    ry(-np.pi/2, q2)   # Ry†(π/2)

    # Column 16: Final RZZ(π/2) between q0-q1 and q2-q3
    x.ctrl(q0, q1)
    rz(np.pi/2, q1)
    x.ctrl(q0, q1)

    x.ctrl(q2, q3)
    rz(np.pi/2, q3)
    x.ctrl(q2, q3)

    # Column 17: Final single-qubit rotations
    rx(np.pi/2, q0)    # Rx(π/2)
    ry(-np.pi/2, q1)   # Ry†(π/2)
    ry(np.pi/2, q2)    # Ry(π/2)
    rx(np.pi/2, q3)    # Rx(π/2)


def get_interactions(N):
    """
    Generates the interaction sets G2 and G4 based on the loop limits in Eq. 15.
    Returns standard 0-based indices as lists of lists of ints.

    Args:
        N (int): Sequence length.

    Returns:
        G2: List of lists containing two body term indices
        G4: List of lists containing four body term indices
    """
    G2 = []
    G4 = []

    # Two-body terms
    # Formula: i from 0 to N-3, k from 1 to floor((N-i-1)/2)
    # Indices: [i, i+k]
    for i in range(N - 2):
        for k in range(1, (N - i - 1) // 2 + 1):  # k = 1, ..., floor((N-i-1)/2)
            G2.append([i, i + k])

    # Four-body terms
    # Formula (1-based): i from 0 to N-4, t from 1 to floor((N-i-2)/2), k from t+1 to N-i-1-t
    # Indices: [i, i+t, i+k, i+k+t]
    for i in range(N - 3):
        for t in range(1, (N - i - 2) // 2 + 1):
            for k in range(t + 1, N - i - t):
                G4.append([i, i + t, i + k, i + k + t])

    return G2, G4


@cudaq.kernel
def trotterized_circuit(N: int, G2: list[list[int]], G4: list[list[int]],
                        steps: int, dt: float, T: float, thetas: list[float]):
    """
    Implements the full Trotterized counteradiabatic circuit from Eq. 15.

    Args:
        N: Number of qubits (sequence length)
        G2: List of two-body interaction indices [[i, i+k], ...]
        G4: List of four-body interaction indices [[i, i+t, i+k, i+k+t], ...]
        steps: Number of Trotter steps
        dt: Time step size
        T: Total evolution time
        thetas: List of theta values for each Trotter step
    """
    # Initialize qubits in |+⟩ state (ground state of H_i = sum_i h_i^x * sigma_i^x)
    reg = cudaq.qvector(N)
    h(reg)

    # Apply Trotter steps
    for step in range(steps):
        theta = thetas[step]

        # Apply two-body terms: R_YZ(4*theta) * R_ZY(4*theta) for each pair in G2
        # The two_qubit_block implements the combined operation
        for pair in G2:
            i = pair[0]
            j = pair[1]
            two_qubit_block(reg[i], reg[j], 4.0 * theta)

        # Apply four-body terms: R_YZZZ * R_ZYZZ * R_ZZYZ * R_ZZZY (each with 8*theta)
        # The four_qubit_block implements the combined operation
        for quad in G4:
            i0 = quad[0]
            i1 = quad[1]
            i2 = quad[2]
            i3 = quad[3]
            four_qubit_block(reg[i0], reg[i1], reg[i2], reg[i3], 8.0 * theta)


# ============================================================================
# COMPARISON CONFIGURATION
# ============================================================================

N = 25                  # Sequence length
population_size = 50    # Population size for MTS
max_generations = 20    # Reduced for faster execution (increase for better results)
n_shots = 200           # Number of quantum samples to generate

def bitstring_to_runlength(bitstring: str) -> str:
    """
    Convert a bitstring to run-length notation.
    Example: '00110' -> '221'
    """
    if not bitstring:
        return ""

    runs = []
    current_char = bitstring[0]
    count = 1

    for char in bitstring[1:]:
        if char == current_char:
            count += 1
        else:
            runs.append(str(count))
            current_char = char
            count = 1

    runs.append(str(count))
    return ''.join(runs)


# ============================================================================
# Random Search Baseline
# ============================================================================

def random_search(N: int, n_samples: int) -> Tuple[np.ndarray, int, List[np.ndarray]]:
    """
    Pure random search: generate random sequences and return the best one.
    No optimization is performed - this serves as a baseline.

    Args:
        N: Sequence length
        n_samples: Number of random samples to generate

    Returns:
        Tuple of (best_sequence, best_energy, all_samples)
    """
    print("=" * 70)
    print("[RANDOM] PURE RANDOM SEARCH")
    print("=" * 70)
    print(f"[RANDOM] Generating {n_samples} random sequences of length N={N}")

    start_time = time.time()

    # Generate random population
    population = [np.random.choice([-1, 1], size=N) for _ in range(n_samples)]
    energies = [compute_energy(s) for s in population]

    # Find best
    best_idx = np.argmin(energies)
    best_s = population[best_idx].copy()
    best_energy = energies[best_idx]
    best_merit = compute_merit_factor(best_s, best_energy)

    total_time = time.time() - start_time

    print(f"[RANDOM] Population statistics:")
    print(f"      - Energy:  min={np.min(energies)}, max={np.max(energies)}, "
          f"mean={np.mean(energies):.2f}, std={np.std(energies):.2f}")
    merits = [compute_merit_factor(s, e) for s, e in zip(population, energies)]
    print(f"      - Merit:   min={np.min(merits):.4f}, max={np.max(merits):.4f}, "
          f"mean={np.mean(merits):.4f}")
    print("=" * 70)
    print(f"[RANDOM] Results:")
    print(f"      - Best energy: {best_energy}")
    print(f"      - Best merit factor: {best_merit:.4f}")
    print(f"      - Best sequence: {sequence_to_bitstring(best_s)}")
    print(f"      - Total time: {total_time:.2f} seconds")
    print("=" * 70)

    return best_s, best_energy, population


# ============================================================================
# COMPARISON: Memetic Tabu Search vs Random Search vs Quantum-Enhanced MTS
# ============================================================================

print("\n" + "=" * 70)
print("COMPARISON: MTS vs Random Search vs Quantum-Enhanced MTS")
print("=" * 70)
print(f"Configuration: N={N}, population_size={population_size}, "
      f"max_generations={max_generations}, n_shots={n_shots}")
print("=" * 70)

# ============================================================================
# Method 1: Pure Random Search (Baseline)
# ============================================================================

print("\n" + "=" * 70)
print("METHOD 1: Pure Random Search (Baseline)")
print("=" * 70)

random.seed(SEED)
np.random.seed(SEED)

random_best_s, random_best_energy, random_samples = random_search(
    N=N,
    n_samples=n_shots
)

# ============================================================================
# Method 2: Memetic Tabu Search (Random Initialization)
# ============================================================================

print("\n" + "=" * 70)
print("METHOD 2: Memetic Tabu Search (Random Initialization)")
print("=" * 70)

random.seed(SEED)
np.random.seed(SEED)

mts_best_s, mts_best_energy, mts_final_population = memetic_tabu_search(
    N=N,
    population_size=population_size,
    max_generations=max_generations,
    p_combine=0.9,
    initial_population=None  # Random initialization
)

# ============================================================================
# Method 3: Quantum-Enhanced MTS
# ============================================================================

print("\n" + "=" * 70)
print("METHOD 3: Quantum-Enhanced MTS")
print("=" * 70)

# Generate quantum population
T = 1.0
n_steps = 1
dt = T / n_steps
G2, G4 = get_interactions(N)

thetas = []
for step in range(1, n_steps + 1):
    t = step * dt
    theta_val = utils.compute_theta(t, dt, T, N, G2, G4)
    thetas.append(theta_val)

print(f"[QE-MTS] Sampling {n_shots} bitstrings from quantum circuit...")
quantum_result = cudaq.sample(
    trotterized_circuit,
    N, G2, G4, n_steps, dt, T, thetas,
    shots_count=n_shots
)

# Convert quantum samples to initial population
quantum_population = []
quantum_energies = []

for bitstring, count in quantum_result.items():
    seq = bitstring_to_sequence(bitstring)
    energy = compute_energy(seq)
    for _ in range(count):
        quantum_population.append(seq.copy())
        quantum_energies.append(energy)

print(f"[QE-MTS] Quantum population statistics ({len(quantum_population)} samples):")
print(f"      - Mean energy: {np.mean(quantum_energies):.2f}")
print(f"      - Min energy: {np.min(quantum_energies)}")

# Run MTS with quantum-seeded population
random.seed(SEED)
np.random.seed(SEED)

qe_best_s, qe_best_energy, qe_final_population = memetic_tabu_search(
    N=N,
    population_size=population_size,
    max_generations=max_generations,
    p_combine=0.9,
    initial_population=quantum_population[:population_size]
)

# ============================================================================
# Results Comparison
# ============================================================================

print("\n" + "=" * 70)
print("RESULTS COMPARISON")
print("=" * 70)

random_best_merit = compute_merit_factor(random_best_s, random_best_energy)
mts_best_merit = compute_merit_factor(mts_best_s, mts_best_energy)
qe_best_merit = compute_merit_factor(qe_best_s, qe_best_energy)

print("\n1. Random Search (Baseline):")
print(f"   Best energy: {random_best_energy}")
print(f"   Best merit:  {random_best_merit:.4f}")
print(f"   Sequence:    {sequence_to_bitstring(random_best_s)}")

print("\n2. Memetic Tabu Search:")
print(f"   Best energy: {mts_best_energy}")
print(f"   Best merit:  {mts_best_merit:.4f}")
print(f"   Sequence:    {sequence_to_bitstring(mts_best_s)}")

print("\n3. Quantum-Enhanced MTS:")
print(f"   Best energy: {qe_best_energy}")
print(f"   Best merit:  {qe_best_merit:.4f}")
print(f"   Sequence:    {sequence_to_bitstring(qe_best_s)}")

# Determine winner
results = [
    ("Random Search", random_best_energy, random_best_merit),
    ("Memetic Tabu Search", mts_best_energy, mts_best_merit),
    ("Quantum-Enhanced MTS", qe_best_energy, qe_best_merit)
]
results_sorted = sorted(results, key=lambda x: x[1])  # Sort by energy (lower is better)

print("\n" + "-" * 70)
print("RANKING (by energy, lower is better):")
for rank, (name, energy, merit) in enumerate(results_sorted, 1):
    print(f"  {rank}. {name}: E={energy}, F={merit:.4f}")

print("\n" + "-" * 70)
print("IMPROVEMENTS vs Random Search Baseline:")
print(f"  MTS improvement:    {random_best_energy - mts_best_energy} energy units")
print(f"  QE-MTS improvement: {random_best_energy - qe_best_energy} energy units")
print(f"  QE-MTS vs MTS:      {mts_best_energy - qe_best_energy} energy units")

# ============================================================================
# Visualization
# ============================================================================

print("\n" + "=" * 70)
print("VISUALIZATION")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Row 1: Energy distributions
random_energies = [compute_energy(s) for s in random_samples]
mts_pop_energies = [compute_energy(s) for s in mts_final_population]
qe_pop_energies = [compute_energy(s) for s in qe_final_population]

ax1 = axes[0, 0]
ax1.hist(random_energies, bins=20, alpha=0.7, color="gray", edgecolor="black")
ax1.axvline(random_best_energy, color="red", linestyle="--", linewidth=2, label=f"Best: {random_best_energy}")
ax1.set_xlabel("Energy")
ax1.set_ylabel("Count")
ax1.set_title("Random Search: Energy Distribution")
ax1.legend()

ax2 = axes[0, 1]
ax2.hist(mts_pop_energies, bins=20, alpha=0.7, color="orange", edgecolor="black")
ax2.axvline(mts_best_energy, color="red", linestyle="--", linewidth=2, label=f"Best: {mts_best_energy}")
ax2.set_xlabel("Energy")
ax2.set_ylabel("Count")
ax2.set_title("MTS: Final Population Energy")
ax2.legend()

ax3 = axes[0, 2]
ax3.hist(qe_pop_energies, bins=20, alpha=0.7, color="blue", edgecolor="black")
ax3.axvline(qe_best_energy, color="red", linestyle="--", linewidth=2, label=f"Best: {qe_best_energy}")
ax3.set_xlabel("Energy")
ax3.set_ylabel("Count")
ax3.set_title("QE-MTS: Final Population Energy")
ax3.legend()

# Row 2: Autocorrelations of best sequences
ax4 = axes[1, 0]
random_Ck = [compute_Ck(random_best_s, k) for k in range(1, N)]
colors_random = ["gray" if c == 0 else ("green" if abs(c) <= 1 else "red") for c in random_Ck]
ax4.bar(range(1, N), random_Ck, color=colors_random, alpha=0.7)
ax4.axhline(0, color="black", linestyle="-", linewidth=0.5)
ax4.set_xlabel("Lag k")
ax4.set_ylabel("C_k")
ax4.set_title(f"Random: Autocorrelation (E={random_best_energy}, F={random_best_merit:.4f})")

ax5 = axes[1, 1]
mts_Ck = [compute_Ck(mts_best_s, k) for k in range(1, N)]
colors_mts = ["orange" if c == 0 else ("green" if abs(c) <= 1 else "red") for c in mts_Ck]
ax5.bar(range(1, N), mts_Ck, color=colors_mts, alpha=0.7)
ax5.axhline(0, color="black", linestyle="-", linewidth=0.5)
ax5.set_xlabel("Lag k")
ax5.set_ylabel("C_k")
ax5.set_title(f"MTS: Autocorrelation (E={mts_best_energy}, F={mts_best_merit:.4f})")

ax6 = axes[1, 2]
qe_Ck = [compute_Ck(qe_best_s, k) for k in range(1, N)]
colors_qe = ["blue" if c == 0 else ("green" if abs(c) <= 1 else "red") for c in qe_Ck]
ax6.bar(range(1, N), qe_Ck, color=colors_qe, alpha=0.7)
ax6.axhline(0, color="black", linestyle="-", linewidth=0.5)
ax6.set_xlabel("Lag k")
ax6.set_ylabel("C_k")
ax6.set_title(f"QE-MTS: Autocorrelation (E={qe_best_energy}, F={qe_best_merit:.4f})")

plt.tight_layout()
plt.savefig("comparison_results.png", dpi=150)
plt.show()

print("\nFigure saved to comparison_results.png")

