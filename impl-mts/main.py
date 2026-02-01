"""
Memetic Tabu Search (MTS) for the LABS problem.

Ported from impl-trotter: LABS energy utilities, tabu search, combine/mutate,
and memetic_tabu_search. Use as a classical component: e.g. quantum samples
(QAOA+Grover) can be converted to ±1 sequences and passed as initial_population
to memetic_tabu_search() for quantum-seeded MTS.

Based on: "Scaling advantage with quantum-enhanced memetic tabu search for LABS"
https://arxiv.org/abs/2511.04553
"""

import numpy as np
from typing import Tuple, List, Optional
import random
import time


# ============================================================================
# LABS Energy and Merit Factor Computation
# ============================================================================

def compute_Ck(s: np.ndarray, k: int) -> int:
    """Compute C_k = sum_i s_i s_{i+k} for a ±1 sequence."""
    N = len(s)
    return int(np.sum(s[: N - k] * s[k:]))


def compute_energy(s: np.ndarray) -> int:
    """Sidelobe energy E(s) = sum_{k=1}^{N-1} C_k^2."""
    N = len(s)
    energy = 0
    for k in range(1, N):
        Ck = compute_Ck(s, k)
        energy += Ck * Ck
    return energy


def compute_merit_factor(s: np.ndarray, energy: Optional[int] = None) -> float:
    """Merit factor F(s) = N^2 / (2*E(s))."""
    N = len(s)
    if energy is None:
        energy = compute_energy(s)
    if energy == 0:
        return float("inf")
    return (N * N) / (2.0 * energy)


def energy_and_merit(s: np.ndarray) -> Tuple[int, float]:
    energy = compute_energy(s)
    merit = compute_merit_factor(s, energy)
    return energy, merit


def bitstring_to_sequence(bitstring: str) -> np.ndarray:
    """Convert a '0'/'1' bitstring to a +1/-1 sequence (0 -> +1, 1 -> -1)."""
    return np.array([1 if b == "0" else -1 for b in bitstring])


def sequence_to_bitstring(s: np.ndarray) -> str:
    """Convert a +1/-1 sequence to a '0'/'1' bitstring."""
    return "".join(["0" if x == 1 else "1" for x in s])


def compute_all_Ck(s: np.ndarray) -> np.ndarray:
    """Compute all C_k for k = 1..N-1."""
    N = len(s)
    Ck_values = np.zeros(N, dtype=np.int64)
    for k in range(1, N):
        Ck_values[k] = compute_Ck(s, k)
    return Ck_values


def compute_delta_energy(
    s: np.ndarray, Ck_values: np.ndarray, flip_idx: int
) -> int:
    """Change in energy if we flip the bit at flip_idx."""
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


def update_Ck_after_flip(s: np.ndarray, Ck_values: np.ndarray, flip_idx: int) -> None:
    """Update C_k in place after flipping bit at flip_idx (s already flipped)."""
    N = len(s)
    for k in range(1, N):
        if flip_idx + k < N:
            Ck_values[k] += 2 * s[flip_idx] * s[flip_idx + k]
        if flip_idx - k >= 0:
            Ck_values[k] += 2 * s[flip_idx - k] * s[flip_idx]


# ============================================================================
# Tabu Search
# ============================================================================


def tabu_search(
    s: np.ndarray,
    max_iter: Optional[int] = None,
    min_tabu_factor: float = 0.1,
    max_tabu_factor: float = 0.12,
    tabu_id: Optional[int] = None,
    verbose: bool = True,
    fixed_indices: Optional[List[int]] = None,
) -> Tuple[np.ndarray, int]:
    """
    Tabu search starting from sequence s.

    fixed_indices: indices that must not be flipped (e.g. [0, 1] when first two bits are fixed).

    Returns:
        (best_sequence, best_energy)
    """
    N = len(s)
    s = s.copy()
    prefix = f"[TABU-{tabu_id}]" if tabu_id is not None else "[TABU]"
    fixed_set = set(fixed_indices) if fixed_indices else set()
    movable = [i for i in range(N) if i not in fixed_set]

    if max_iter is None:
        max_iter = random.randint(N // 2, 3 * N // 2)

    min_tabu = max(1, int(min_tabu_factor * max_iter))
    max_tabu = max(min_tabu + 1, int(max_tabu_factor * max_iter))

    if verbose:
        print(
            f"{prefix} Starting tabu search: N={N}, max_iter={max_iter}, "
            f"tenure_range=[{min_tabu}, {max_tabu}]"
        )

    tabu_list = np.zeros(N, dtype=np.int64)
    Ck_values = compute_all_Ck(s)
    current_energy = int(np.sum(Ck_values[1:] ** 2))
    best_s = s.copy()
    best_energy = current_energy

    if verbose:
        init_merit = compute_merit_factor(s, current_energy)
        print(f"{prefix} Initial: energy={current_energy}, merit={init_merit:.4f}")

    improvements = 0
    aspiration_used = 0

    for t in range(1, max_iter + 1):
        best_move = None
        best_move_energy = float("inf")
        used_aspiration = False

        for i in movable:
            delta = compute_delta_energy(s, Ck_values, i)
            new_energy = current_energy + delta
            is_tabu = tabu_list[i] >= t
            aspiration = new_energy < best_energy

            if (not is_tabu or aspiration) and new_energy < best_move_energy:
                best_move = i
                best_move_energy = new_energy
                if is_tabu and aspiration:
                    used_aspiration = True

        if best_move is None and movable:
            best_move = random.choice(movable)
            best_move_energy = current_energy + compute_delta_energy(
                s, Ck_values, best_move
            )
        elif best_move is None:
            break

        if used_aspiration:
            aspiration_used += 1

        s[best_move] *= -1
        update_Ck_after_flip(s, Ck_values, best_move)
        current_energy = best_move_energy
        tenure = random.randint(min_tabu, max_tabu)
        tabu_list[best_move] = t + tenure

        if current_energy < best_energy:
            best_energy = current_energy
            best_s = s.copy()
            improvements += 1
            if verbose:
                merit = compute_merit_factor(s, current_energy)
                print(
                    f"{prefix} Iter {t}/{max_iter}: New best energy={best_energy}, "
                    f"merit={merit:.4f} (flipped bit {best_move})"
                )

    if verbose:
        final_merit = compute_merit_factor(best_s, best_energy)
        print(
            f"{prefix} Completed: energy={best_energy}, merit={final_merit:.4f}, "
            f"improvements={improvements}, aspiration_moves={aspiration_used}"
        )

    return best_s, best_energy


# ============================================================================
# Combine (Crossover) and Mutate
# ============================================================================


def combine(
    parent1: np.ndarray,
    parent2: np.ndarray,
    verbose: bool = True,
    fixed_indices: Optional[List[int]] = None,
) -> np.ndarray:
    """Single-point crossover. If fixed_indices is set, child keeps parent1's values at those indices."""
    N = len(parent1)
    k = random.randint(1, N - 1)
    child = np.concatenate([parent1[:k], parent2[k:]])
    if fixed_indices:
        for i in fixed_indices:
            child[i] = parent1[i]
    if verbose:
        print(
            f"[COMBINE] Cut point k={k}, child inherits [0:{k}) from P1, [{k}:{N}) from P2"
        )
    return child


def mutate(
    s: np.ndarray,
    p_mut: Optional[float] = None,
    verbose: bool = True,
    fixed_indices: Optional[List[int]] = None,
) -> np.ndarray:
    """Mutate by flipping each bit independently with probability p_mut (default 1/N). Skips fixed_indices."""
    N = len(s)
    if p_mut is None:
        p_mut = 1.0 / N
    fixed_set = set(fixed_indices) if fixed_indices else set()
    child = s.copy()
    flipped = [i for i in range(N) if i not in fixed_set and random.random() < p_mut]
    for i in flipped:
        child[i] *= -1
    if verbose:
        print(f"[MUTATE] p_mut={p_mut:.4f}, flipped {len(flipped)} bits: {flipped}")
    return child


# ============================================================================
# Memetic Tabu Search (MTS)
# ============================================================================


def memetic_tabu_search(
    N: int,
    population_size: int = 100,
    max_generations: int = 1000,
    p_combine: float = 0.9,
    initial_population: Optional[List[np.ndarray]] = None,
    target_energy: Optional[int] = None,
    verbose: bool = True,
    fixed_indices: Optional[List[int]] = None,
    fixed_values: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int, List[np.ndarray]]:
    """
    Memetic Tabu Search for LABS.

    Args:
        N: Sequence length
        population_size: Population size (K=100 in paper)
        max_generations: Max generations
        p_combine: Crossover probability (0.9 in paper)
        initial_population: Optional list of ±1 sequences (e.g. from quantum samples)
        target_energy: Optional early-stop target
        verbose: Whether to print progress
        fixed_indices: Indices that must not be changed (e.g. [0, 1] for truncated Hamiltonian)
        fixed_values: Values at fixed_indices (length must match fixed_indices; e.g. [-1,-1] for bits "11")

    Returns:
        (best_sequence, best_energy, final_population)
    """
    if verbose:
        print("=" * 70)
        print("[MTS] MEMETIC TABU SEARCH - INITIALIZATION")
        print("=" * 70)
        print(
            f"[MTS] N={N}, population_size={population_size}, "
            f"max_generations={max_generations}, p_combine={p_combine}, p_mut=1/N"
        )
    if fixed_indices is not None:
        if fixed_values is None or len(fixed_values) != len(fixed_indices):
            raise ValueError("fixed_values must be provided and match length of fixed_indices")
        fixed_values = np.asarray(fixed_values, dtype=np.int32)
        if verbose:
            print(f"[MTS] Fixed indices: {fixed_indices} -> values {fixed_values.tolist()}")

    def _apply_fixed(s: np.ndarray) -> None:
        if fixed_indices is not None:
            for j, i in enumerate(fixed_indices):
                s[i] = fixed_values[j]
    else:
        def _apply_fixed(s: np.ndarray) -> None:
            pass

    if initial_population is not None:
        if verbose:
            print(f"[MTS] Using provided initial population of {len(initial_population)} sequences")
        population = [seq.copy() for seq in initial_population]
        while len(population) < population_size:
            new_s = np.random.choice([-1, 1], size=N)
            _apply_fixed(new_s)
            population.append(new_s)
        population = population[:population_size]
    else:
        if verbose:
            print(f"[MTS] Generating random initial population of {population_size} sequences")
        population = []
        for _ in range(population_size):
            s = np.random.choice([-1, 1], size=N)
            _apply_fixed(s)
            population.append(s)

    energies = [compute_energy(s) for s in population]
    merits = [compute_merit_factor(s, e) for s, e in zip(population, energies)]

    best_idx = np.argmin(energies)
    best_s = population[best_idx].copy()
    best_energy = int(energies[best_idx])
    best_merit = merits[best_idx]

    if verbose:
        print(
            f"[MTS] Initial best: energy={best_energy}, merit={best_merit:.4f}, "
            f"sequence={sequence_to_bitstring(best_s)}"
        )
        print("[MTS] STARTING EVOLUTION")
        print("=" * 70)

    start_time = time.time()
    improvements_count = 0

    for gen in range(max_generations):
        if target_energy is not None and best_energy <= target_energy:
            if verbose:
                print(f"[MTS] TARGET REACHED at generation {gen}!")
            break

        if random.random() < p_combine:
            idx1, idx2 = random.sample(range(population_size), 2)
            child = combine(
                population[idx1], population[idx2],
                verbose=verbose, fixed_indices=fixed_indices,
            )
            operation = f"crossover(P[{idx1}], P[{idx2}])"
        else:
            idx = random.randint(0, population_size - 1)
            child = population[idx].copy()
            operation = f"select(P[{idx}])"

        child = mutate(child, verbose=verbose, fixed_indices=fixed_indices)
        improved_child, child_energy = tabu_search(
            child, tabu_id=gen, verbose=verbose, fixed_indices=fixed_indices,
        )
        _apply_fixed(improved_child)
        child_merit = compute_merit_factor(improved_child, child_energy)

        if child_energy < best_energy:
            old_best = best_energy
            old_merit = best_merit
            best_energy = child_energy
            best_merit = child_merit
            best_s = improved_child.copy()
            improvements_count += 1
            if verbose:
                print(
                    f"[MTS] Gen {gen}: NEW BEST! energy: {old_best}->{best_energy}, "
                    f"merit: {old_merit:.4f}->{best_merit:.4f} via {operation}"
                )

        replace_idx = random.randint(0, population_size - 1)
        population[replace_idx] = improved_child
        energies[replace_idx] = child_energy
        merits[replace_idx] = child_merit

        if verbose:
            elapsed = time.time() - start_time
            print(
                f"[MTS] Gen {gen+1}/{max_generations}: best_energy={best_energy}, "
                f"best_merit={best_merit:.4f}, pop_mean_energy={np.mean(energies):.1f}, "
                f"elapsed={elapsed:.1f}s"
            )

    total_time = time.time() - start_time
    if verbose:
        print("=" * 70)
        print("[MTS] EVOLUTION COMPLETE")
        print(
            f"[MTS] Best energy={best_energy}, merit={compute_merit_factor(best_s, best_energy):.4f}, "
            f"improvements={improvements_count}, time={total_time:.2f}s"
        )
        print("=" * 70)

    return best_s, best_energy, population


# ============================================================================
# Random Search (baseline)
# ============================================================================


def random_search(
    N: int, n_samples: int, verbose: bool = True
) -> Tuple[np.ndarray, int, List[np.ndarray]]:
    """Pure random search: n_samples random sequences, return best and all."""
    if verbose:
        print(f"[RANDOM] Generating {n_samples} random sequences of length N={N}")
    population = [np.random.choice([-1, 1], size=N) for _ in range(n_samples)]
    energies = [compute_energy(s) for s in population]
    best_idx = np.argmin(energies)
    best_s = population[best_idx].copy()
    best_energy = energies[best_idx]
    if verbose:
        print(
            f"[RANDOM] Best energy={best_energy}, merit={compute_merit_factor(best_s, best_energy):.4f}"
        )
    return best_s, best_energy, population


# ============================================================================
# Entrypoint (run MTS with random init if executed as script)
# ============================================================================

if __name__ == "__main__":
    import sys

    N = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    pop_size = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    max_gen = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    random.seed(42)
    np.random.seed(42)
    best_s, best_energy, pop = memetic_tabu_search(
        N=N, population_size=pop_size, max_generations=max_gen, p_combine=0.9
    )
    print("Final best:", sequence_to_bitstring(best_s), "E=", best_energy)
