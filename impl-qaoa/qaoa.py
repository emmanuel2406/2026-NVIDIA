"""
Fixed-Parameter Digitized Counterdiabatic QAOA (DC-QAOA) for the LABS Problem

This implementation is based on:
1. "Evidence of scaling advantage for the quantum approximate optimization algorithm
   on a classically intractable problem" - Shaydulin et al., Sci. Adv. 10, eadm6761 (2024)
2. "Digitized-counterdiabatic quantum approximate optimization algorithm" - Chandarana et al.
3. NVIDIA CUDA-Q DC-QAOA tutorial

Key approach:
- Fixed-parameter QAOA (no variational optimization loop)
- Counterdiabatic Y-rotations added after each layer to suppress diabatic transitions
- Designed to integrate with Quantum Minimum Finding (QMF) for improved scaling
"""

import cudaq
import numpy as np
from math import floor
from typing import List, Tuple, Dict


# =============================================================================
# LABS Energy and Merit Factor Computation
# =============================================================================

def compute_Ck(s: np.ndarray, k: int) -> int:
    """Compute autocorrelation C_k = sum_{i=1}^{N-k} s_i * s_{i+k}"""
    N = len(s)
    return sum(s[i] * s[i + k] for i in range(N - k))


def compute_energy(s: np.ndarray) -> int:
    """Compute sidelobe energy E(s) = sum_{k=1}^{N-1} C_k^2"""
    N = len(s)
    return sum(compute_Ck(s, k) ** 2 for k in range(1, N))


def compute_merit_factor(s: np.ndarray, energy: int = None) -> float:
    """Compute merit factor F(s) = N^2 / (2 * E(s))"""
    N = len(s)
    if energy is None:
        energy = compute_energy(s)
    if energy == 0:
        return float('inf')
    return (N * N) / (2 * energy)


def bitstring_to_sequence(bitstring: str) -> np.ndarray:
    """Convert bitstring '010...' to ±1 sequence."""
    return np.array([1 if b == '0' else -1 for b in bitstring])


def sequence_to_bitstring(s: np.ndarray) -> str:
    """Convert ±1 sequence to bitstring '010...'."""
    return ''.join('0' if x == 1 else '1' for x in s)


# =============================================================================
# LABS Hamiltonian Interaction Terms (Eq. 4 from Shaydulin et al.)
# =============================================================================

def get_interactions(N: int) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Generate two-body (G2) and four-body (G4) interaction indices for LABS Hamiltonian.

    H_C = sum_{i,j in G2} z_i*z_j + 2*sum_{i,j,k,l in G4} z_i*z_j*z_k*z_l
    """
    G2 = []  # Two-body: z_i * z_{i+2k}
    G4 = []  # Four-body: z_i * z_{i+t} * z_{i+k} * z_{i+k+t}

    # Two-body terms
    for i in range(N - 2):
        for k in range(1, int(floor((N - i) / 2)) + 1):
            j = i + 2 * k
            if j < N:
                G2.append([i, j])

    # Four-body terms
    for i in range(N - 3):
        for t in range(1, int(floor((N - i - 1) / 2)) + 1):
            for k in range(t + 1, N - i - t):
                if i + k + t < N:
                    G4.append([i, i + t, i + k, i + k + t])

    return G2, G4


# =============================================================================
# Fixed Parameter Schedules (from Shaydulin et al.)
# =============================================================================

def get_fixed_parameters(p: int, N: int) -> Tuple[List[float], List[float], List[float]]:
    """
    Get fixed QAOA parameters for p layers.

    Based on the parameter transfer approach from Shaydulin et al.:
    - Parameters optimized on small instances are averaged and rescaled
    - gamma is rescaled by 1/N for transferability
    - alpha (counterdiabatic) parameters added for DC-QAOA

    Returns:
        betas: Mixing angles (X rotations)
        gammas: Phase angles (problem Hamiltonian)
        alphas: Counterdiabatic angles (Y rotations)
    """
    # Fixed parameters inspired by literature (averaged over small N)
    # These would ideally come from optimization on N=24-31 as in the paper

    # Linear ramp schedule for beta (mixing)
    betas = [0.5 * np.pi * (l + 1) / (p + 1) for l in range(p)]

    # Rescaled gamma schedule (phase) - scaled by 1/N for transferability
    gammas = [0.25 * np.pi * (l + 1) / ((p + 1) * N) for l in range(p)]

    # Counterdiabatic alpha schedule (Y rotations)
    # Small values that help suppress diabatic transitions
    alphas = [0.1 * np.pi * (l + 1) / (p + 1) for l in range(p)]

    return betas, gammas, alphas


# =============================================================================
# CUDA-Q Quantum Kernels
# =============================================================================

@cudaq.kernel
def rzz_gate(q0: cudaq.qubit, q1: cudaq.qubit, theta: float):
    """RZZ gate: exp(-i * theta/2 * Z⊗Z)"""
    x.ctrl(q0, q1)
    rz(theta, q1)
    x.ctrl(q0, q1)


@cudaq.kernel
def qaoa_kernel(N: int, G2: list[list[int]], G4: list[list[int]],
                num_layers: int, betas: list[float], gammas: list[float]):
    """
    Standard QAOA circuit for LABS.

    Structure per layer:
        1. Phase operator: exp(-i * gamma * H_C)
        2. Mixing operator (X): exp(-i * beta * sum_j X_j)
    """
    qubits = cudaq.qvector(N)

    # Initialize in uniform superposition |+>^N
    h(qubits)

    for layer in range(num_layers):
        gamma = gammas[layer]
        beta = betas[layer]

        # === Phase operator: exp(-i * gamma * H_C) ===

        # Two-body terms: z_i * z_j
        for term in G2:
            i = term[0]
            j = term[1]
            rzz_gate(qubits[i], qubits[j], 2.0 * gamma)

        # Four-body terms: z_i * z_j * z_k * z_l (with coefficient 2)
        for term in G4:
            i0 = term[0]
            i1 = term[1]
            i2 = term[2]
            i3 = term[3]
            # Decompose exp(-i * 2*gamma * Z_i*Z_j*Z_k*Z_l)
            x.ctrl(qubits[i0], qubits[i1])
            x.ctrl(qubits[i1], qubits[i2])
            x.ctrl(qubits[i2], qubits[i3])
            rz(4.0 * gamma, qubits[i3])
            x.ctrl(qubits[i2], qubits[i3])
            x.ctrl(qubits[i1], qubits[i2])
            x.ctrl(qubits[i0], qubits[i1])

        # === Mixing operator: exp(-i * beta * sum_j X_j) ===
        for j in range(N):
            rx(2.0 * beta, qubits[j])


@cudaq.kernel
def dc_qaoa_kernel(N: int, G2: list[list[int]], G4: list[list[int]],
                   num_layers: int, betas: list[float], gammas: list[float],
                   alphas: list[float]):
    """
    Digitized Counterdiabatic QAOA (DC-QAOA) circuit for LABS.

    Structure per layer:
        1. Phase operator: exp(-i * gamma * H_C)
        2. Mixing operator (X): exp(-i * beta * sum_j X_j)
        3. Counterdiabatic (Y): exp(-i * alpha * sum_j Y_j)  <- suppresses diabatic transitions
    """
    qubits = cudaq.qvector(N)

    # Initialize in uniform superposition |+>^N
    h(qubits)

    for layer in range(num_layers):
        gamma = gammas[layer]
        beta = betas[layer]
        alpha = alphas[layer]

        # === Phase operator: exp(-i * gamma * H_C) ===

        # Two-body terms: z_i * z_j
        for term in G2:
            i = term[0]
            j = term[1]
            rzz_gate(qubits[i], qubits[j], 2.0 * gamma)

        # Four-body terms: z_i * z_j * z_k * z_l (with coefficient 2)
        for term in G4:
            i0 = term[0]
            i1 = term[1]
            i2 = term[2]
            i3 = term[3]
            # Decompose exp(-i * 2*gamma * Z_i*Z_j*Z_k*Z_l)
            x.ctrl(qubits[i0], qubits[i1])
            x.ctrl(qubits[i1], qubits[i2])
            x.ctrl(qubits[i2], qubits[i3])
            rz(4.0 * gamma, qubits[i3])
            x.ctrl(qubits[i2], qubits[i3])
            x.ctrl(qubits[i1], qubits[i2])
            x.ctrl(qubits[i0], qubits[i1])

        # === Mixing operator: exp(-i * beta * sum_j X_j) ===
        for j in range(N):
            rx(2.0 * beta, qubits[j])

        # === Counterdiabatic term: exp(-i * alpha * sum_j Y_j) ===
        # This is the key DC-QAOA addition that suppresses diabatic transitions
        for j in range(N):
            ry(2.0 * alpha, qubits[j])


# =============================================================================
# Sampling and Evaluation
# =============================================================================

def sample_qaoa(N: int, num_layers: int = 3, shots: int = 1000,
                betas: List[float] = None, gammas: List[float] = None,
                use_counterdiabatic: bool = False,
                alphas: List[float] = None) -> Dict[str, int]:
    """
    Sample from QAOA or DC-QAOA circuit.

    Args:
        N: Sequence length
        num_layers: Number of QAOA layers (p)
        shots: Number of measurement shots
        betas, gammas: Optional custom parameters (uses fixed if None)
        use_counterdiabatic: If True, use DC-QAOA with Y rotations
        alphas: Counterdiabatic angles (only used if use_counterdiabatic=True)

    Returns:
        Dictionary of bitstring -> count
    """
    G2, G4 = get_interactions(N)

    # Use fixed parameters if not provided
    if betas is None or gammas is None:
        betas, gammas, alphas = get_fixed_parameters(num_layers, N)
    elif alphas is None and use_counterdiabatic:
        _, _, alphas = get_fixed_parameters(num_layers, N)

    if use_counterdiabatic:
        result = cudaq.sample(dc_qaoa_kernel, N, G2, G4, num_layers,
                              betas, gammas, alphas, shots_count=shots)
    else:
        result = cudaq.sample(qaoa_kernel, N, G2, G4, num_layers,
                              betas, gammas, shots_count=shots)

    return dict(result.items())


def evaluate_samples(samples: Dict[str, int]) -> Tuple[str, int, float]:
    """Find the best solution from samples."""
    best_bitstring = None
    best_energy = float('inf')
    best_merit = 0.0

    for bitstring in samples:
        seq = bitstring_to_sequence(bitstring)
        energy = compute_energy(seq)

        if energy < best_energy:
            best_energy = energy
            best_bitstring = bitstring
            best_merit = compute_merit_factor(seq, energy)

    return best_bitstring, best_energy, best_merit


def compute_expected_merit(samples: Dict[str, int]) -> float:
    """Compute expected merit factor from samples."""
    total_shots = sum(samples.values())
    expected_merit = 0.0

    for bitstring, count in samples.items():
        seq = bitstring_to_sequence(bitstring)
        energy = compute_energy(seq)
        merit = compute_merit_factor(seq, energy)
        expected_merit += (count / total_shots) * merit

    return expected_merit


def get_population_from_samples(samples: Dict[str, int],
                                 population_size: int = 50) -> List[np.ndarray]:
    """
    Convert quantum samples to a population for seeding MTS.

    Args:
        samples: Dictionary of bitstring -> count
        population_size: Target population size

    Returns:
        List of ±1 sequences
    """
    population = []

    # Sort by count (most frequent first)
    sorted_samples = sorted(samples.items(), key=lambda x: x[1], reverse=True)

    for bitstring, count in sorted_samples:
        seq = bitstring_to_sequence(bitstring)
        # Add sequence (potentially multiple times based on count)
        for _ in range(min(count, population_size - len(population))):
            population.append(seq.copy())
            if len(population) >= population_size:
                break
        if len(population) >= population_size:
            break

    return population


# =============================================================================
# Main Workflow
# =============================================================================

def run_qaoa_labs(N: int, num_layers: int = 3, shots: int = 1000,
                  use_counterdiabatic: bool = True, verbose: bool = True) -> Dict:
    """
    Run fixed-parameter QAOA or DC-QAOA for the LABS problem.

    Args:
        N: Sequence length
        num_layers: Number of QAOA layers (p)
        shots: Number of measurement shots
        use_counterdiabatic: If True, use DC-QAOA; else standard QAOA
        verbose: Print progress information

    Returns:
        Dictionary with results
    """
    mode = "DC-QAOA" if use_counterdiabatic else "QAOA"
    if verbose:
        print(f"Running {mode} for LABS (N={N}, p={num_layers}, shots={shots})")

    G2, G4 = get_interactions(N)
    betas, gammas, alphas = get_fixed_parameters(num_layers, N)

    if verbose:
        print(f"  Two-body terms: {len(G2)}")
        print(f"  Four-body terms: {len(G4)}")
        if use_counterdiabatic:
            print(f"  Parameters (layer 1): beta={betas[0]:.4f}, gamma={gammas[0]:.4f}, alpha={alphas[0]:.4f}")
        else:
            print(f"  Parameters (layer 1): beta={betas[0]:.4f}, gamma={gammas[0]:.4f}")

    # Sample from circuit
    samples = sample_qaoa(N, num_layers, shots, betas, gammas,
                          use_counterdiabatic=use_counterdiabatic, alphas=alphas)

    # Evaluate
    best_bitstring, best_energy, best_merit = evaluate_samples(samples)
    expected_merit = compute_expected_merit(samples)

    if verbose:
        print(f"\nResults:")
        print(f"  Best bitstring: {best_bitstring}")
        print(f"  Best energy: {best_energy}")
        print(f"  Best merit factor: {best_merit:.4f}")
        print(f"  Expected merit: {expected_merit:.4f}")

        print("\nTop 5 sampled solutions:")
        sorted_samples = sorted(samples.items(), key=lambda x: x[1], reverse=True)[:5]
        for bs, count in sorted_samples:
            seq = bitstring_to_sequence(bs)
            e = compute_energy(seq)
            m = compute_merit_factor(seq, e)
            print(f"  {bs}: count={count}, energy={e}, merit={m:.4f}")

    result = {
        'best_bitstring': best_bitstring,
        'best_sequence': bitstring_to_sequence(best_bitstring),
        'best_energy': best_energy,
        'best_merit': best_merit,
        'expected_merit': expected_merit,
        'samples': samples,
        'N': N,
        'num_layers': num_layers,
        'use_counterdiabatic': use_counterdiabatic,
        'parameters': {'betas': betas, 'gammas': gammas}
    }
    if use_counterdiabatic:
        result['parameters']['alphas'] = alphas

    return result


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Fixed-Parameter QAOA for LABS Problem")
    print("=" * 60)

    N = 15

    # Compare standard QAOA vs DC-QAOA
    print("\n--- Standard QAOA ---")
    result_std = run_qaoa_labs(N, num_layers=3, shots=1000, use_counterdiabatic=False)

    print("\n--- DC-QAOA (with counterdiabatic Y rotations) ---")
    result_dc = run_qaoa_labs(N, num_layers=3, shots=1000, use_counterdiabatic=True)

    print("\n" + "=" * 60)
    print("Comparison Summary")
    print("=" * 60)
    print(f"Standard QAOA: energy={result_std['best_energy']}, merit={result_std['best_merit']:.4f}")
    print(f"DC-QAOA:       energy={result_dc['best_energy']}, merit={result_dc['best_merit']:.4f}")
