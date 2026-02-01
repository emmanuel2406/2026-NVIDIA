# SPDX-License-Identifier: Apache-2.0 AND CC-BY-NC-4.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""# Quantum Enhanced Optimization for Radar and Communications Applications


The Low Autocorrelation Binary Sequences (LABS) is an important and challenging optimization problem with applications related to radar, telecommunications, and other signal related applications. This CUDA-Q Academic module will focus on a clever quantum-enhanced hybrid method developed in a collaboration between Kipu Quantum, University of the Basque Country EHU, and NVIDIA for solving the LABS problem. (This notebook was jointly developed with input from all collaborators.)

Other CUDA-Q Academic modules like [Divide and Conquer MaxCut QAOA](https://github.com/NVIDIA/cuda-q-academic/tree/main/qaoa-for-max-cut) and [Quantum Finance](https://github.com/NVIDIA/cuda-q-academic/blob/main/quantum-applications-to-finance/03_qchop.ipynb), demonstrate how quantum computing can be used outright to solve optimization problems. This notebook demonstrates a slightly different approach. Rather than considering QPUs as the tool to produce the final answer, it demonstrates how quantum can be used to enhance the effectiveness of leading classical methods.  

The benefits of such an approach were highlighted in [Scaling advantage with quantum-enhanced memetic tabu search for LABS](https://arxiv.org/html/2511.04553v1).  This notebook, co-created with the authors of the paper, will allow you to explore the findings of their research and write your own CUDA-Q code that builds a representative quantum-enhanced workflow for solving the LABS problem. Moreover, it will introduce advancements in counteradiabatic optimization techniques on which reduce the quantum resources required to run on a QPU.

**Prerequisites:** This lab assumes you have a basic knowledge of quantum computing, including operators, gates, etc.  For a refresher on some of these topics, explore the [Quick start to Quantum](https://github.com/NVIDIA/cuda-q-academic/tree/main/quick-start-to-quantum) series.

**In this lab you will:**
* 1. Understand the LABS problem and its relation ot radar and communication applications.
* 2. Solve LABS classically with memetic tabu search and learn about the limitations of such methods.
* 3. Code a couteradiabatic algorithm using CUDA-Q to produce approximate solutions to the LABS problem.
* 4. Use the CUDA-Q results to seed your tabu search and understand the potential benefits of this approach.


**Terminology you will use:**
* Low autocorrelation of binary sequences (LABS)
* counteradiabatic optimization
* memetic-tabu search

**CUDA-Q Syntax you will use:**
* cudaq.sample()
* @cudaq.kernel
* ry(), rx(), rz(), x(), h()
* x.ctrl()

Run the code below to initialize the libraries you will need.
"""

import cudaq
import numpy as np
from math import floor
import auxiliary_files.labs_utils as utils

"""## The LABS problem and applications

The **Low Autocorrelation Binary Sequences (LABS)** problem is fundamental to many applications, but originated with applications to radar.

Consider a radar that monitors airport traffic.  The radar signal sent to detect incoming planes must have as much range as possible to ensure safe approaches are planned well in advance.  The range of a radar signal can be increased by sending a longer pulse.  However, in order to differentiate between multiple objects, pulses need to be short to provide high resolution. So, how do you handle situations where you need both?

One solution is a technique called pulse compression.  The idea is to send a long signal, but vary the phase at regular intervals such that the resolution is increased. Generally, the initial signal will encode a binary sequence of phase shifts, where each interval corresponds to a signal with a 0 or 180 degree phase shift.

The tricky part is selecting an optimal encoding sequence.  When the signal returns, it is fed into a matched filter with the hope that a singular sharp peak will appear, indicating clear detection.  The autocorrelation of the original signal, or how similar the signal is to itself,  determines if a single peak or a messier signal with sidelobes will be detected. A signal should have high autocorrelation when overlayed on top of itself, but low autocorrelation when shifted with a lag.

Consider the image below.  The signal on the left has a crisp single peak while the single on the right produces many undesirable sidelobes which can inhibit clear detection.  

<img src="images/quantum_enhanced_optimization_LABS/radar.png" width="800">


So, how do you select a good signal?   This is where LABS comes in, defining these questions as a binary optimization problem. Given a binary sequence of length $N$, $(s_1 \cdots s_N) \in {\pm 1}^N$, the goal is to minimize the following objective function.

$$ E(s) = \sum_{k=1}^{N-1} C_k^2 $$

Where $C_k$ is defined as.

 $$C_k= \sum_{i=1}^{N-k} s_is_{i+k}$$


So, each $C_k$ computes how similar the original signal is to the shifted one for each offset value $k$.  To explore this more, try the interactive widget linked [here](https://nvidia.github.io/cuda-q-academic/interactive_widgets/labs_visualization.html).  See if you can select a very good and very poor sequence and see the difference for the case of $N=7$.

## Classical Solution of the LABS problem

The LABS problem is tricky to solve for a few reasons. First, the configuration space grows exponentially.  Second, underlying symmetries of the problem result in many degeneracies in the optimization landscape severely inhibiting local search methods.

<div style="background-color: #f9fff0; border-left: 6px solid #76b900; padding: 15px; border-radius: 4px; box-shadow: 0px 2px 4px rgba(0,0,0,0.1);">
    <h3 style="color: #76b900; margin-top: 0; margin-bottom: 10px;">Exercise 1:</h3>
    <p style="font-size: 16px; color: #333;">
Using the widget above, try to find some of the symmetries for the LABS problem. That is, for a fixed bitstring length, can you find patterns to produce the same energy with different pulse patterns.
</div>

The best known performance for a classical optimization technique is Memetic Tabu search (MTS) which exhibits a scaling of $O(1.34^N)$.  The MTS algorithm is depicted below.  It begins with a randomly selected population of bitstrings and finds the best solution from them.  Then, a child is selected by sampling directly from or combining multiple bitstrings from the population.  The child is mutated with probability $p_{mutate}$ and then input to a tabu search, which performs a modified greedy local search starting from the child bitstring.  If the result is better than the best in the population, it is updated as the new leader and randomly replaces a  bitstring in the population.


<img src="images/quantum_enhanced_optimization_LABS/mts_algorithm.png" width="500">

Such an approach is fast, parallelizable, and allows for exploration with improved searching of the solution landscape.  

<div style="background-color: #f9fff0; border-left: 6px solid #76b900; padding: 15px; border-radius: 4px; box-shadow: 0px 2px 4px rgba(0,0,0,0.1);">
    <h3 style="color: #76b900; margin-top: 0; margin-bottom: 10px;">Exercise 2:</h3>
    <p style="font-size: 16px; color: #333;">
Before exploring any quantum approach, get a sense for how MTS works by coding it yourself based generally on the figure above. Algorithms for the combine and mutate steps are provided below as used in the paper. You may need to research more specific details of the process, especially for how tabu search is performed. The MTS procedure should output and optimal bitstring and its energy.  Also, write a function to visualize the results including the energy distribution of the final population.
</div>



<img src="images/quantum_enhanced_optimization_LABS/combine_mutate.png" width="400">

*Exercise 1 Solution*
- Complementary symmetry: by flipping  every bit +1 -> -1 and -1 -> +1 this gives the same energy, which is because the double negation essentially cancels out for every multiplication operation.
- Reversal symmetry: reverse the sequence also gives a symmetry since the indicies being reversed just reads the products backwards instead of forwards in the summation
"""



# Exercise 2: Memetic Tabu Search (MTS) Algorithm for LABS

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import random
import time


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

# ============================================================================
# Main Execution
# ============================================================================

SEED=42
random.seed(SEED)
np.random.seed(SEED)
N = 25
population_size = 50
max_generations = 100

print(f"\n[CONFIG] Problem Configuration:")
print(f"         - N (sequence length): {N}")
print(f"         - Population size: {population_size}")
print(f"         - Max generations: {max_generations}")
print(f"         - Random seed: {SEED}")
print()

# Run MTS
best_s, best_energy, final_population = memetic_tabu_search(
    N=N,
    population_size=population_size,
    max_generations=max_generations,
    p_combine=0.9
)

# Visualize results
visualize_mts_results(best_s, best_energy, final_population, title_prefix="MTS")

# Verify energy computation
verified_energy = compute_energy(best_s)
verified_merit = compute_merit_factor(best_s, verified_energy)
print(f"\n[VERIFY] Energy verification: recomputed = {verified_energy}")
print(f"[VERIFY] Merit verification:  recomputed = {verified_merit:.4f}")
print(f"[VERIFY] Energy match: {verified_energy == best_energy}")

# Exercise 2: Memetic Tabu Search (MTS) Algorithm for LABS
#
# This implementation is based on the algorithm described in:
# "Scaling advantage with quantum-enhanced memetic tabu search for LABS"
# https://arxiv.org/abs/2511.04553
# and "New Improvements in Solving Large LABS Instances Using Massively
# Parallelizable Memetic Tabu Search" https://arxiv.org/abs/2504.00987

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import random
import time


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

# ============================================================================
# Main Execution
# ============================================================================

SEED=42
random.seed(SEED)
np.random.seed(SEED)
N = 50
population_size = 50
max_generations = 100

print(f"\n[CONFIG] Problem Configuration:")
print(f"         - N (sequence length): {N}")
print(f"         - Population size: {population_size}")
print(f"         - Max generations: {max_generations}")
print(f"         - Random seed: {SEED}")
print()

# Run MTS
best_s, best_energy, final_population = memetic_tabu_search(
    N=N,
    population_size=population_size,
    max_generations=max_generations,
    p_combine=0.9
)

# Visualize results
visualize_mts_results(best_s, best_energy, final_population, title_prefix="MTS")

# Verify energy computation
verified_energy = compute_energy(best_s)
verified_merit = compute_merit_factor(best_s, verified_energy)
print(f"\n[VERIFY] Energy verification: recomputed = {verified_energy}")
print(f"[VERIFY] Merit verification:  recomputed = {verified_merit:.4f}")
print(f"[VERIFY] Energy match: {verified_energy == best_energy}")

"""It looks like it works! This merit and energy matches with the optimal found in this paper: https://arxiv.org/pdf/1512.02475 with an energy of 26 and merit of 7.692.

## Building a Quantum Enhanced Workflow

Despite the effectiveness of MTS, it still exhibits exponential scaling  $O(1.34^N)$ behavior and becomes intractable for large $N$.  Quantum computing provides a potential alternative method for solving the LABS problem because the properties of entanglement, interference, and superpositon may allow for a better global search.  Recent demonstrations have even produced evidence that the quantum approximate optimization algorithm (QAOA) can be used to reduce the scaling of the LABS problem to $O(1.21^N)$ for $N$ between 28 and 40 with quantum minimum finding.

However, current quantum hardware limitations restrict solution to problems of greater than about $N=20$, meaning that it will be some time before quantum approaches can outperform the classical state of the art. It should also be noted that standard QAOA can struggle with LABS and require many layers to converge the parameters if other tricks are not employed.

The authors of [Scaling advantage with quantum-enhanced memetic tabu search for LABS](https://arxiv.org/html/2511.04553v1) cleverly explored an alternate path that combines quantum and classical approaches and might be able to provide a more near-term benefit.  Instead of expecting the quantum computer to solve the problem entirely, they asked how a quantum approach might enhance MTS.

The basic idea is that a quantum optimization routine could run first and the resulting state be sampled to produce a better population for MTS. Many such heuristics for defining the initial population are possible, but the rest of this notebook will explore their methodology, help you to build the workflow yourself, and allow you to analyze the benefits of their approach.

The first step of quantum enhanced MTS (QE-MTS) is to prepare a circuit with CUDA-Q that approximates the ground state of the Hamiltonian corresponding to the LABS problem. You could do this with any optimization algorithm such as QAOA or using an adiabatic approach.  (See the [Quantum Portfolio Optimization](https://github.com/NVIDIA/cuda-q-academic/blob/main/quantum-applications-to-finance/03_qchop.ipynb) CUDA-Q Academic lab for a detailed comparison of these two common approaches.)

The authors of this work opted for an adiabatic approach (More on why later). Recall that the goal of an adiabatic optimization is to begin with a Hamiltonian that has an easily prepared ground state ($H_i$). Then, the adiabatic Hamiltonian $H_{ad}$ can be constructed as $H_{ad}(\lambda) = (1-\lambda)H_i +\lambda H_f $, where $\lambda$ is a function of time and $H_f$ is the Hamiltonian representing a qubit encoding of the LABS problem.

$$H_f = 2 \sum_{i=1}^{N-2} \sigma_i^z \sum_{k=1}^{\lfloor \frac{N-i}{2} \rfloor} \sigma_{i+k}^z
+ 4 \sum_{i=1}^{N-3} \sigma_i^z \sum_{t=1}^{\lfloor \frac{N-i-1}{2} \rfloor} \sum_{k=t+1}^{N-i-t} \sigma_{i+t}^z \sigma_{i+k}^z \sigma_{i+k+t}^z$$

The authors also selected $H_i = \sum_i h^x_i \sigma^x_i $ which has an easily prepared ground state of $\ket{+}^{\otimes N}$.

The challenge for implementing the optimization procedure becomes selection of an operator that will quickly and accurately evolve to the ground state of $H_f$.  One approach is to use a so-called auxiliary countradiabatic (CD) term $H_{CD}$, which corrects diabatic transitions that jump out of the ground state during the evolution. The figure below demonstrates the benefit of using a CD correction.


<img src="images/quantum_enhanced_optimization_LABS/counteradiabatic.png" width="900">




An operator called the adiabatic gauge potential $A_{\lambda}$ is the ideal choice for the CD term as it suppresses all possible diabatic transitions, resulting in the following total system to evolve.

$$ H(\lambda) = H_{ad}(\lambda) + \lambda H_{CD} (\lambda) $$

$A(\lambda)$ is derrived from $H_{ad}(\lambda)$  (see paper for details) as it contains information about underlying physics of the problem.

There is a problem though.  The $A(\lambda)$ term cannot be efficiently expressed exactly and needs to be approximated.  It also turns out that in the so called impulse regime, where the adiabatic evolution is very fast, $H_{cd} (\lambda)$ dominates $H_{ad}(\lambda)$, meaning that the final implementation corresponds to the operator $H(\lambda) = H^1_{cd}(\lambda)$ where  $H^1_{cd}(\lambda)$ is a first order approximation of $A(\lambda)$ (see equation 7 in the paper).

A final step is to use Trotterization to define the quantum circuit to apply $e^{-\theta (t) i H_{cd}}$. The details for this derivation are shown in the appendix of the paper. and result from equation B3 is shown below.  

$$
\begin{equation}
\begin{aligned}
U(0, T) = \prod_{n=1}^{n_{\text{trot}}} & \left[ \prod_{i=1}^{N-2} \prod_{k=1}^{\lfloor \frac{N-i}{2} \rfloor} R_{Y_i Z_{i+k}}\big(4\theta(n\Delta t)h_i^x\big) R_{Z_i Y_{i+k}}\big(4\theta(n\Delta t)h_{i+k}^x\big) \right] \\
& \times \prod_{i=1}^{N-3} \prod_{t=1}^{\lfloor \frac{N-i-1}{2} \rfloor} \prod_{k=t+1}^{N-i-t} \bigg( R_{Y_i Z_{i+t} Z_{i+k} Z_{i+k+t}}\big(8\theta(n\Delta t)h_i^x\big) \\
& \quad \times R_{Z_i Y_{i+t} Z_{i+k} Z_{i+k+t}}\big(8\theta(n\Delta t)h_{i+t}^x\big) \\
& \quad \times R_{Z_i Z_{i+t} Y_{i+k} Z_{i+k+t}}\big(8\theta(n\Delta t)h_{i+k}^x\big) \\
& \quad \times R_{Z_i Z_{i+t} Z_{i+k} Y_{i+k+t}}\big(8\theta(n\Delta t)h_{i+k+t}^x\big) \bigg)
\end{aligned}
\end{equation}
$$

It turns out that this implementation is more efficient than QAOA in terms of gate count. The authors calculated that for $N=67$, QAOA would require 1.4 million entangling gates while the CD approach derived here requires only 236 thousand entangling gates.


<div style="background-color: #f9fff0; border-left: 6px solid #76b900; padding: 15px; border-radius: 4px; box-shadow: 0px 2px 4px rgba(0,0,0,0.1);">
    <h3 style="color: #76b900; margin-top: 0; margin-bottom: 10px;">Exercise 3:</h3>
    <p style="font-size: 16px; color: #333;">
At first glance, this equation might looks quite complicated. However, observe the structure and note two "blocks" of terms.  Can you spot them?  

They are 2 qubit terms that look like $R_{YZ}(\theta)$ or $R_{ZY}(\theta)$.

As well as 4 qubit terms that look like $R_{YZZZ}(\theta)$, $R_{ZYZZ}(\theta)$, $R_{ZZYZ}(\theta)$, or $R_{ZZZY}(\theta)$.

Thankfully the authors derive a pair of circuit implementations for the two and four qubit terms respectively, shown in the figures below.

Using CUDA-Q, write a kernel for each which will be used later to construct the full implementation.

* Hint: Remember that the adjoint of a rotation gate is the same as rotating in the opposite direction.

* Hint: You may also want to define a CUDA-Q kernel for an R$_{ZZ}$ gate.

* Hint: Implementing a circuit from a paper is a great place where AI can help accelerate your work.  If you have access to a coding assistant, feel free to use it here.
</div>

<img src="images/quantum_enhanced_optimization_LABS/kernels.png" width="1300">
"""

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

# Test the kernels with a simple circuit
@cudaq.kernel
def test_two_qubit(theta: float):
    q = cudaq.qvector(2)
    h(q[0])
    h(q[1])
    two_qubit_block(q[0], q[1], theta)

@cudaq.kernel
def test_four_qubit(theta: float):
    q = cudaq.qvector(4)
    h(q[0])
    h(q[1])
    h(q[2])
    h(q[3])
    four_qubit_block(q[0], q[1], q[2], q[3], theta)

# Test the kernels
print("Testing 2-qubit block:")
result_2q = cudaq.sample(test_two_qubit, 0.5, shots_count=1000)
print(result_2q)
print(cudaq.draw(test_two_qubit, 0.5))

print("\nTesting 4-qubit block:")
result_4q = cudaq.sample(test_four_qubit, 0.5, shots_count=1000)
print(result_4q)
print(cudaq.draw(test_four_qubit, 0.5))

"""**Checking the drawings manually, it seems we've successfully reproduced the 2 and 4 qubit operators!**

There are a few additional items we need to consider before completing the final implementation of the entire circuit.  One simplification we can make is that for our problem the $h_i^x$ terms are all 1 and any $h_b^x$ terms are 0, and are only there for generalizations of this model.

The remaining challenge is derivation of the angles that are used to apply each of the circuits you defined above. These are obtained from two terms $\lambda(t)$ and $\alpha(t)$.  


The $\lambda(t)$ defines an annealing schedule and is generally a Sin function which slowly "turns on" the problem Hamiltonian.  For computing our angles, we need the derivative of $\lambda(t)$.

The $\alpha$ term is a bit trickier and is the solution to a set of differential equations which minimize the distance between $H^1_{CD}(\lambda)$ and $A(\lambda)$.  The result is

$$\alpha(t) = \frac{-\Gamma_1(t)}{\Gamma_2(t)} $$

Where $\Gamma_1(t)$ and $\Gamma_2(t)$ are defined in equations 16 and 17 of the paper and essentially depend on the structure of the optimization problem.  Curious learners can look at the functions in `labs_utils.py`  to see how these are computed, based on the problem size and specific time step in the Trotter process.


<div style="background-color: #f9fff0; border-left: 6px solid #76b900; padding: 15px; border-radius: 4px; box-shadow: 0px 2px 4px rgba(0,0,0,0.1);">
    <h3 style="color: #76b900; margin-top: 0; margin-bottom: 10px;">Exercise 4:</h3>
    <p style="font-size: 16px; color: #333;">
The `compute_theta` function, called in the following cells, requires all indices of the two and four body terms. These will be used again in our main kernel to apply the respective gates.  Use the products in the formula below to finish the function in the cell below.  Save them as `G2` and `G4` where each is a list of lists of indices defining the two and four term interactions. As you are translating an equation to a set of loops, this is a great opportunity to use an AI coding assistant.

$$
\begin{equation}
\begin{aligned}
U(0, T) = \prod_{n=1}^{n_{\text{trot}}} & \left[ \prod_{i=1}^{N-2} \prod_{k=1}^{\lfloor \frac{N-i}{2} \rfloor} R_{Y_i Z_{i+k}}\big(4\theta(n\Delta t)h_i^x\big) R_{Z_i Y_{i+k}}\big(4\theta(n\Delta t)h_{i+k}^x\big) \right] \\
& \times \prod_{i=1}^{N-3} \prod_{t=1}^{\lfloor \frac{N-i-1}{2} \rfloor} \prod_{k=t+1}^{N-i-t} \bigg( R_{Y_i Z_{i+t} Z_{i+k} Z_{i+k+t}}\big(8\theta(n\Delta t)h_i^x\big) \\
& \quad \times R_{Z_i Y_{i+t} Z_{i+k} Z_{i+k+t}}\big(8\theta(n\Delta t)h_{i+t}^x\big) \\
& \quad \times R_{Z_i Z_{i+t} Y_{i+k} Z_{i+k+t}}\big(8\theta(n\Delta t)h_{i+k}^x\big) \\
& \quad \times R_{Z_i Z_{i+t} Z_{i+k} Y_{i+k+t}}\big(8\theta(n\Delta t)h_{i+k+t}^x\big) \bigg)
\end{aligned}
\end{equation}
$$

</div>
"""

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


# Testing with multiple N values
print("Testing get_interactions with N=5:")
G2_test, G4_test = get_interactions(5)
print(f"G2 (two-body terms): {G2_test}")
print(f"Number of G2 terms: {len(G2_test)}")
print(f"G4 (four-body terms): {G4_test}")
print(f"Number of G4 terms: {len(G4_test)}")

print("\nTesting get_interactions with N=6:")
G2_test6, G4_test6 = get_interactions(6)
print(f"G2 (two-body terms): {G2_test6}")
print(f"Number of G2 terms: {len(G2_test6)}")
print(f"G4 (four-body terms): {G4_test6}")
print(f"Number of G4 terms: {len(G4_test6)}")

print("\nTesting get_interactions with N=20:")
G2_20, G4_20 = get_interactions(20)
print(f"Number of G2 terms: {len(G2_20)}")
print(f"Number of G4 terms: {len(G4_20)}")

"""

<div style="background-color: #f9fff0; border-left: 6px solid #76b900; padding: 15px; border-radius: 4px; box-shadow: 0px 2px 4px rgba(0,0,0,0.1);">
    <h3 style="color: #76b900; margin-top: 0; margin-bottom: 10px;">Exercise 5:</h3>
    <p style="font-size: 16px; color: #333;">
You are now ready to construct the entire circuit and run the counteradiabatic optimization procedure. The final kernel needs to apply Equation 15 for a specified total evolution time $T$ and the `n_steps` number of Trotter steps.  It must also take as input, the indices for the two and four body terms and the thetas to be applied each step, as these cannot be computed within a CUDA-Q kernel.

The helper function `compute_theta` computes the theta parameters for you, using a few additional functions in accordance with the equations defined in the paper.
</div>
"""

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
# Test the trotterized circuit
# ============================================================================

T = 1.0             # total evolution time
n_steps = 1         # number of Trotter steps
dt = T / n_steps
N = 10

# Get interaction indices
G2, G4 = get_interactions(N)

print(f"Circuit parameters:")
print(f"  N (qubits) = {N}")
print(f"  T (total time) = {T}")
print(f"  n_steps (Trotter steps) = {n_steps}")
print(f"  dt = {dt}")
print(f"  Number of G2 (two-body) terms = {len(G2)}")
print(f"  Number of G4 (four-body) terms = {len(G4)}")

# Compute theta values for each Trotter step
thetas = []
for step in range(1, n_steps + 1):
    t = step * dt
    theta_val = utils.compute_theta(t, dt, T, N, G2, G4)
    thetas.append(theta_val)
    print(f"  Step {step}: t = {t:.4f}, theta = {theta_val:.6f}")

# Sample the circuit
print("\nSampling the trotterized circuit...")
result = cudaq.sample(trotterized_circuit, N, G2, G4, n_steps, dt, T, thetas, shots_count=1000)

print(f"\nTop 10 most frequent bitstrings:")
sorted_results = sorted(result.items(), key=lambda x: x[1], reverse=True)
for bitstring, count in sorted_results[:10]:
    # Convert bitstring to sequence and compute energy
    seq = bitstring_to_sequence(bitstring)
    energy = compute_energy(seq)
    merit = compute_merit_factor(seq, energy)
    print(f"  {bitstring}: count={count}, energy={energy}, merit={merit:.4f}")

# Compute energy statistics for the sampled distribution
print("\nEnergy statistics for sampled bitstrings:")
energies = []
for bitstring, count in result.items():
    seq = bitstring_to_sequence(bitstring)
    energy = compute_energy(seq)
    energies.extend([energy] * count)

energies = np.array(energies)
print(f"  Mean energy: {np.mean(energies):.2f}")
print(f"  Std energy: {np.std(energies):.2f}")
print(f"  Min energy: {np.min(energies)}")
print(f"  Max energy: {np.max(energies)}")

"""## Generating Quantum Enhanced Results

Recall that the point of this lab is to demonstrate the potential benefits of running a quantum subroutine as a preprocessing step for classical optimization of a challenging problem like LABS. you now have all of the tools you need to try this for yourself.

<div style="background-color: #f9fff0; border-left: 6px solid #76b900; padding: 15px; border-radius: 4px; box-shadow: 0px 2px 4px rgba(0,0,0,0.1);">
    <h3 style="color: #76b900; margin-top: 0; margin-bottom: 10px;">Exercise 6:</h3>
    <p style="font-size: 16px; color: #333;">
Use your CUDA-Q code to prepare an initial population for your memetic search algorithm and see if you can improve the results relative to a random initial population.  If you are running on a CPU, you will need to run smaller problem instances. The code below sets up the problem

</div>

"""

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
# Step 1: Generate quantum-enhanced initial population
# ============================================================================

print("=" * 70)
print("STEP 1: Generating Quantum-Enhanced Initial Population")
print("=" * 70)

T = 1.0
n_steps = 1
dt = T / n_steps
G2, G4 = get_interactions(N)

thetas = []
for step in range(1, n_steps + 1):
    t = step * dt
    theta_val = utils.compute_theta(t, dt, T, N, G2, G4)
    thetas.append(theta_val)

print(f"Sampling {n_shots} bitstrings from quantum circuit...")
quantum_result = cudaq.sample(
    trotterized_circuit,
    N, G2, G4, n_steps, dt, T, thetas,
    shots_count=n_shots
)

# Convert quantum samples to initial population (as +1/-1 sequences)
quantum_population = []
quantum_energies = []

for bitstring, count in quantum_result.items():
    seq = bitstring_to_sequence(bitstring)
    energy = compute_energy(seq)
    # Add each unique bitstring to population (could add duplicates based on count)
    for _ in range(count):
        quantum_population.append(seq.copy())
        quantum_energies.append(energy)

print(f"\nQuantum population statistics ({len(quantum_population)} samples):")
print(f"  Mean energy: {np.mean(quantum_energies):.2f}")
print(f"  Std energy: {np.std(quantum_energies):.2f}")
print(f"  Min energy: {np.min(quantum_energies)}")
print(f"  Max energy: {np.max(quantum_energies)}")

print("\nTop 5 lowest energy sequences from quantum sampling:")
sorted_indices = np.argsort(quantum_energies)[:5]
for idx in sorted_indices:
    bitstring = sequence_to_bitstring(quantum_population[idx])
    runlength = bitstring_to_runlength(bitstring)
    merit = compute_merit_factor(quantum_population[idx], quantum_energies[idx])
    print(f"  {bitstring} (rl: {runlength}): energy={quantum_energies[idx]}, merit={merit:.4f}")

# ============================================================================
# Step 2: Generate random initial population for comparison
# ============================================================================

print("\n" + "=" * 70)
print("STEP 2: Generating Random Initial Population for Comparison")
print("=" * 70)

random_population = [np.random.choice([-1, 1], size=N) for _ in range(n_shots)]
random_energies = [compute_energy(s) for s in random_population]

print(f"\nRandom population statistics ({len(random_population)} samples):")
print(f"  Mean energy: {np.mean(random_energies):.2f}")
print(f"  Std energy: {np.std(random_energies):.2f}")
print(f"  Min energy: {np.min(random_energies)}")
print(f"  Max energy: {np.max(random_energies)}")

# ============================================================================
# Step 3: Run MTS with quantum-seeded population
# ============================================================================

print("\n" + "=" * 70)
print("STEP 3: Running MTS with Quantum-Enhanced Population")
print("=" * 70)

# Reset seed for fair comparison
random.seed(SEED + 1)
np.random.seed(SEED + 1)

qe_best_s, qe_best_energy, qe_final_population = memetic_tabu_search(
    N=N,
    population_size=population_size,
    max_generations=max_generations,
    p_combine=0.9,
    initial_population=quantum_population[:population_size]  # Use quantum samples
)

# ============================================================================
# Step 4: Run MTS with random population
# ============================================================================

print("\n" + "=" * 70)
print("STEP 4: Running MTS with Random Population")
print("=" * 70)

# Reset seed for fair comparison
random.seed(SEED + 1)
np.random.seed(SEED + 1)

rand_best_s, rand_best_energy, rand_final_population = memetic_tabu_search(
    N=N,
    population_size=population_size,
    max_generations=max_generations,
    p_combine=0.9,
    initial_population=None  # Use random initialization
)

# ============================================================================
# Step 5: Compare Results
# ============================================================================

print("\n" + "=" * 70)
print("STEP 5: Comparison of Results")
print("=" * 70)

qe_best_merit = compute_merit_factor(qe_best_s, qe_best_energy)
rand_best_merit = compute_merit_factor(rand_best_s, rand_best_energy)

print("\nQuantum-Enhanced MTS Results:")
print(f"  Best energy: {qe_best_energy}")
print(f"  Best merit factor: {qe_best_merit:.4f}")
print(f"  Best sequence: {sequence_to_bitstring(qe_best_s)}")
print(f"  Run-length: {bitstring_to_runlength(sequence_to_bitstring(qe_best_s))}")

print("\nRandom-Seeded MTS Results:")
print(f"  Best energy: {rand_best_energy}")
print(f"  Best merit factor: {rand_best_merit:.4f}")
print(f"  Best sequence: {sequence_to_bitstring(rand_best_s)}")
print(f"  Run-length: {bitstring_to_runlength(sequence_to_bitstring(rand_best_s))}")

print("\nImprovement from Quantum Enhancement:")
energy_improvement = rand_best_energy - qe_best_energy
merit_improvement = qe_best_merit - rand_best_merit
print(f"  Energy improvement: {energy_improvement} (lower is better)")
print(f"  Merit improvement: {merit_improvement:.4f} (higher is better)")

if qe_best_energy < rand_best_energy:
    print("\n  *** Quantum-enhanced MTS found a BETTER solution! ***")
elif qe_best_energy == rand_best_energy:
    print("\n  Both methods found solutions with the same energy.")
else:
    print("\n  Random-seeded MTS found a better solution this time.")

# ============================================================================
# Step 6: Visualize Comparison
# ============================================================================

print("\n" + "=" * 70)
print("STEP 6: Visualizing Comparison")
print("=" * 70)

# Compute final population energies
qe_pop_energies = [compute_energy(s) for s in qe_final_population]
rand_pop_energies = [compute_energy(s) for s in rand_final_population]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Initial population energy comparison
ax1 = axes[0, 0]
ax1.hist(quantum_energies, bins=20, alpha=0.7, label="Quantum", color="blue", edgecolor="black")
ax1.hist(random_energies, bins=20, alpha=0.7, label="Random", color="orange", edgecolor="black")
ax1.set_xlabel("Energy")
ax1.set_ylabel("Count")
ax1.set_title("Initial Population Energy Distribution")
ax1.legend()

# Plot 2: Final population energy comparison
ax2 = axes[0, 1]
ax2.hist(qe_pop_energies, bins=20, alpha=0.7, label="QE-MTS", color="blue", edgecolor="black")
ax2.hist(rand_pop_energies, bins=20, alpha=0.7, label="Random-MTS", color="orange", edgecolor="black")
ax2.axvline(qe_best_energy, color="blue", linestyle="--", linewidth=2, label=f"QE Best: {qe_best_energy}")
ax2.axvline(rand_best_energy, color="orange", linestyle="--", linewidth=2, label=f"Rand Best: {rand_best_energy}")
ax2.set_xlabel("Energy")
ax2.set_ylabel("Count")
ax2.set_title("Final Population Energy Distribution")
ax2.legend()

# Plot 3: Autocorrelation of QE-MTS best sequence
ax3 = axes[1, 0]
qe_Ck = [compute_Ck(qe_best_s, k) for k in range(1, N)]
colors_qe = ["blue" if c == 0 else ("green" if abs(c) <= 1 else "red") for c in qe_Ck]
ax3.bar(range(1, N), qe_Ck, color=colors_qe, alpha=0.7)
ax3.axhline(0, color="black", linestyle="-", linewidth=0.5)
ax3.set_xlabel("Lag k")
ax3.set_ylabel("C_k")
ax3.set_title(f"QE-MTS Best: Autocorrelation (E={qe_best_energy}, F={qe_best_merit:.4f})")

# Plot 4: Autocorrelation of Random-MTS best sequence
ax4 = axes[1, 1]
rand_Ck = [compute_Ck(rand_best_s, k) for k in range(1, N)]
colors_rand = ["orange" if c == 0 else ("green" if abs(c) <= 1 else "red") for c in rand_Ck]
ax4.bar(range(1, N), rand_Ck, color=colors_rand, alpha=0.7)
ax4.axhline(0, color="black", linestyle="-", linewidth=0.5)
ax4.set_xlabel("Lag k")
ax4.set_ylabel("C_k")
ax4.set_title(f"Random-MTS Best: Autocorrelation (E={rand_best_energy}, F={rand_best_merit:.4f})")

plt.tight_layout()
plt.show()

"""The results clearly show that a population sampled from CUDA-Q results in an improved distribution and a lower energy final result. This is exactly the goal of quantum enhanced optimization.  To not necessarily solve the problem, but improve the effectiveness of state-of-the-art classical approaches.

A few major caveats need to be mentioned here. First, We are comparing a quantum generated population to a random sample.  It is quite likely that other classical or quantum heuristics could be used to produce an initial population that might even beat the counteradiabatic method you used, so we cannot make any claims that this is the best.

Recall that the point of the counteradiabatic approach derived in the paper is that it is more efficient in terms of two-qubit gates relative to QAOA. The benefits of this regime would only truly come into play in a setting (e.g. larger problem instance) where it is too difficult to produce a good initial population with any know classical heuristic, and the counteradiabatic approach is more efficiently run on a QPU compared to alternatives.

We should also note that we are comparing a single sample of each approach.  Maybe the quantum sample got lucky or the randomly generated population was unlucky and a more rigorous comparison would need to repeat the analysis many times to draw any confidently conclusions.  

The authors of the paper discuss all of these considerations, but propose an analysis that is quite interesting related to the scaling of the technique. Rather than run large simulations ourselves, examine their results below.


<img src="images/quantum_enhanced_optimization_LABS/tabu_search_results.png" width="900">

The authors computed replicate median (median of solving the problem repeated with same setup) time to solutions (excluding time to sample from QPU) for problem sizes $N=27$ to $N=37$. Two interesting conclusions can be drawn from this. First, standard memetic tabu search (MTS) is generally faster than quantum enhanced (QE) MTS.  But there are two promising trends. For larger problems, the QE-MTS experiments occasionally have excellent performance with times to solution much smaller than all of the MTS data points.  These outliers indicate there are certain instances where QE-MTS could provide much faster time-to-solution.

More importantly, if a line of best fit is calculated using the median of each set of medians, the slope of the QE-MTS line is smaller than the MTS!  This seems to indicate that QE solution of this problem scales $O(1.24^N)$ which is better than the best known classical heuristic ($O(1.34^N)$) and the best known quantum approach (QAOA - $O(1.46^N)$).

For problems of size of $N=47$ or greater, the authors anticipate that QE-MTS could be a promising technique and produce good initial populations that are difficult to obtain classically.

The study reinforces the potential of hybrid workflows enhanced by quantum data such that a classical routine is still the primary solver, but quantum computers make it much more effective.  Future work can explore improvements to both the quantum and classical sides, such as including GPU accelerated memetic search on the classical side.

## Self-validation for Phase 1

We verify LABS solutions using unit tests and the ground-truth `evals/answers.csv`. The evaluation utility (`evals/eval_util.py`) parses the CSV (multiple optimal solutions per N), converts run-length notation to ±1 sequences, and validates energy E and merit factor F_N. See the code cell above for the validation report.
"""

# =============================================================================
# Self-validation: Verify LABS energy E, merit factor F_N, and sequences
# =============================================================================
# Uses evals/eval_util.py and evals/answers.csv to validate that:
# 1. Run-length notation correctly decodes to ±1 sequences
# 2. Energy E = sum over k of C_k^2 is correct
# 3. Merit factor F_N = N²/(2E) is correct
# 4. Multiple optimal solutions per N are all valid

import sys
from pathlib import Path

# Locate evals dir (notebook runs from tutorial_notebook/)
evals_dir = Path("evals")
if not evals_dir.exists():
    evals_dir = Path.cwd() / "evals"
if not evals_dir.exists():
    evals_dir = Path.cwd() / "tutorial_notebook" / "evals"
sys.path.insert(0, str(evals_dir.parent))

from evals.eval_util import (
    runlength_to_sequence,
    compute_energy,
    compute_merit_factor,
    parse_answers_csv,
    run_full_validation,
    validate_solution,
)

csv_path = evals_dir / "answers.csv"
print("=" * 60)
print("LABS Self-validation Report")
print("=" * 60)
print(f"Answers CSV: {csv_path}")
print()

# Run full validation against answers.csv
report = run_full_validation(csv_path)

# Summary
total_rows = sum(info['total'] for info in report.values())
total_passed = sum(info['passed'] for info in report.values())
all_valid = all(info['all_valid'] for info in report.values())

print("Summary:")
print(f"  Total solutions validated: {total_rows}")
print(f"  Passed: {total_passed}")
print(f"  Status: {'✓ ALL VALID' if all_valid else '✗ SOME FAILED'}")
print()

# Spot-check specific N values (as unit tests)
test_Ns = [3, 6, 20]
print("Spot-check for specific N (unit test style):")
for N in test_Ns:
    rows = [r for r in parse_answers_csv(csv_path) if r['N'] == N]
    for r in rows[:1]:  # First solution per N
        seq = runlength_to_sequence(r['sequence'])
        E = compute_energy(seq)
        F = compute_merit_factor(seq, E)
        ok = E == r['E'] and abs(F - r['F_N']) < 0.01
        print(f"  N={N:2d} seq={r['sequence']:12s} -> E={E} (expected {r['E']}), F_N={F:.3f} (expected {r['F_N']}) {'✓' if ok else '✗'}")

print()
print("Validation by N (multiple optimal solutions per N):")
for n in sorted(report.keys())[:15]:  # First 15 N values
    info = report[n]
    status = "✓" if info['all_valid'] else "✗"
    print(f"  N={n:2d}: {info['passed']}/{info['total']} solutions valid {status}")
if len(report) > 15:
    print(f"  ... (N=16 to N={max(report.keys())})")
print()
print("Conclusion: The eval_util validates optimal E, F_N, and run-length sequences")
print("against answers.csv. Run `python evals/eval_util.py` or `pytest evals/` for full tests.")

