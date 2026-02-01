# trotter_qaoa_ground_state.py
#
# CUDA-Q script that constructs the problem Hamiltonian H_f for a system
# whose original size is N sites, with sigma^z_1 = sigma^z_2 = 1.
# The resulting Hamiltonian acts non-trivially on N-2 qubits (sites 3..N).
# It then uses a Trotterized QAOA (alternating-operator) ansatz with
# H_d = sum_j X_j as the driver Hamiltonian to approximate the ground state.

import math
import cudaq
from cudaq import spin
from cudaq.algorithms import qaoa


def site_to_qubit(site: int) -> int:
    """Map original lattice site (3..N) to qubit index (0..N-3)."""
    return site - 3  # valid only for site >= 3


def add_term_to_spin_op(H: cudaq.spin_op, coeff: float,
                        sites: list[int]) -> cudaq.spin_op:
    """
    Add coeff * (product over sites of sigma^z_site) to the spin_op H,
    after replacing sigma^z_1 and sigma^z_2 by the scalar 1.

    Any site in {1,2} is dropped from the operator; if all sites are in {1,2}
    the term is a pure constant and we ignore it (it only shifts energies by
    a constant and does not affect the ground state).
    """
    active_sites = [s for s in sites if s >= 3]
    if len(active_sites) == 0:
        # Pure constant term; ignore (global energy shift only).
        return H

    # Build product of Z's on the active qubits
    q0 = site_to_qubit(active_sites[0])
    op = spin.z(q0)
    for s in active_sites[1:]:
        q = site_to_qubit(s)
        op *= spin.z(q)

    H += coeff * op
    return H


def build_problem_hamiltonian(N: int) -> cudaq.spin_op:
    """
    Construct H_f for an original chain of N sites, with sigma^z_1 = sigma^z_2 = 1.
    The non-trivial degrees of freedom live on N-2 qubits (sites 3..N).
    """
    H = cudaq.spin_op()  # empty Hamiltonian

    # First term:
    # 2 * sum_{i=1}^{N-2} sigma_i^z * sum_{k=1}^{floor((N-i)/2)} sigma_{i+k}^z
    for i in range(1, N - 1):  # i = 1..N-2
        max_k = (N - i) // 2
        for k in range(1, max_k + 1):
            sites = [i, i + k]
            H = add_term_to_spin_op(H, 2.0, sites)

    # Second term:
    # 4 * sum_{i=1}^{N-3} sigma_i^z
    #       sum_{t=1}^{floor((N-i-1)/2)}
    #           sum_{k=t+1}^{N-i-t} sigma_{i+t}^z sigma_{i+k}^z sigma_{i+k+t}^z
    for i in range(1, N - 2):  # i = 1..N-3
        max_t = (N - i - 1) // 2
        for t in range(1, max_t + 1):
            for k in range(t + 1, N - i - t + 1):
                # Overall operator: sigma_i^z * sigma_{i+t}^z * sigma_{i+k}^z * sigma_{i+k+t}^z
                sites = [i, i + t, i + k, i + k + t]
                H = add_term_to_spin_op(H, 4.0, sites)

    return H


def build_driver_hamiltonian(num_qubits: int) -> cudaq.spin_op:
    """
    Driver (mixer) Hamiltonian H_d = sum_j X_j on num_qubits qubits.
    """
    H = cudaq.spin_op()
    for q in range(num_qubits):
        H += spin.x(q)
    return H


def main():
    # Choose target backend
    cudaq.set_target("qpp")  # CPU state-vector simulator

    # ----- User parameters -----
    N = 10     # original number of sites (Hamiltonian acts on N-2 qubits)
    p = 3      # number of QAOA / Trotter layers

    # ----- Build Hamiltonians -----
    H_f = build_problem_hamiltonian(N)
    num_qubits = N - 2
    H_d = build_driver_hamiltonian(num_qubits)

    print(f"Original system size N = {N}")
    print(f"Effective number of qubits (N-2) = {num_qubits}")
    print(f"H_f has {len(H_f.get_terms())} Pauli terms.")

    # ----- Set up QAOA (trotterized alternating evolution) -----
    qaoa_solver = qaoa.QAOA(num_layers=p, mixer=H_d)

    # Minimize the expectation value of H_f to approximate ground state
    result = qaoa_solver.minimize(H_f)

    print("\n=== QAOA Result ===")
    print("Estimated ground-state energy:", result.optimal_value)
    print("Optimal parameters (gammas then betas):")
    print(result.optimal_parameters)

    # Optionally sample the optimized circuit to see the most likely bitstring
    print("\nSampling optimized circuit...")
    samples = cudaq.sample(qaoa_solver.optimal_circuit)
    print(samples)


if __name__ == "__main__":
    main()