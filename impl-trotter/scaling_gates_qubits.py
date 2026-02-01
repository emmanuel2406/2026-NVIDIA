#!/usr/bin/env python3
"""
Scaling analysis: number of gates and number of qubits vs N for the Image Hamiltonian
DCQO circuit (dcqo_flexible_circuit_v2) as used in qe_mts_image_hamiltonian.

Uses the same Hamiltonian and reduction as main.py / qe_mts_image_hamiltonian.py:
- N even: first two qubits fixed as |1⟩|1⟩ -> circuit on N-2 qubits (reduce_hamiltonian_fix_first_two).
- N odd: skew-symmetry reduction -> circuit on (N+1)//2 qubits (get_image_hamiltonian_skew_reduced).

Gate counts are derived from dcqo_flexible_circuit_v2 and the kernel definitions in main.py:
- r_z: 1 gate; r_zz: 3; r_zzz: 5; r_zzzz: 7; r_yz: 5; r_yzzz: 9.
- Per 2-body term per step: 2*r_yz + 1*r_zz = 13 gates.
- Per 4-body term per step: 4*r_yzzz + 1*r_zzzz = 43 gates.

CLI: min N, max N, stride. Optionally trotter_steps and output directory.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add impl-trotter so "from main import ..." resolves to impl-trotter/main.py
_impl_trotter_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_impl_trotter_dir))
sys.path.insert(0, str(_impl_trotter_dir / "auxiliary_files"))

from main import (
    get_image_hamiltonian,
    get_image_hamiltonian_skew_reduced,
    reduce_hamiltonian_fix_first_two,
)

# Gate counts per kernel (from main.py kernel definitions)
GATES_R_Z = 1
GATES_R_ZZ = 3
GATES_R_ZZZ = 5
GATES_R_ZZZZ = 7
GATES_R_YZ = 5
GATES_R_YZZZ = 9

# Per-term gate count per Trotter step in dcqo_flexible_circuit_v2
GATES_PER_1BODY_PER_STEP = GATES_R_Z
GATES_PER_2BODY_PER_STEP = 2 * GATES_R_YZ + GATES_R_ZZ
GATES_PER_3BODY_PER_STEP = GATES_R_ZZZ
GATES_PER_4BODY_PER_STEP = 4 * GATES_R_YZZZ + GATES_R_ZZZZ


def num_qubits_circuit(N: int) -> int:
    """
    Deterministic: number of qubits in the circuit for sequence length N.
    - N even: N - 2 (first two fixed as |1⟩|1⟩).
    - N odd: (N + 1) // 2 (skew-symmetry reduced).
    """
    if N < 3:
        raise ValueError("N must be >= 3")
    if N % 2 == 0:
        return N - 2
    return (N + 1) // 2


def count_reduced_terms(N: int) -> tuple[int, int, int, int]:
    """
    Deterministic: (n1, n2, n3, n4) for the reduced Hamiltonian used in the circuit.
    Uses the same reduction as qe_mts_image_hamiltonian (fix first two for N even,
    skew-reduced for N odd).
    """
    if N < 3:
        raise ValueError("N must be >= 3")
    if N % 2 == 1:
        t1r, t2r, t3r, t4r, _ = get_image_hamiltonian_skew_reduced(N)
    else:
        t1, t2, t3, t4 = get_image_hamiltonian(N)
        t1r, t2r, t3r, t4r = reduce_hamiltonian_fix_first_two(t1, t2, t3, t4)
    return len(t1r), len(t2r), len(t3r), len(t4r)


def total_gates(N: int, trotter_steps: int = 1) -> int:
    """
    Deterministic: total gate count for the DCQO circuit (dcqo_flexible_circuit_v2)
    for sequence length N and given trotter_steps.
    Includes: initial H on all qubits, per-step evolution (1/2/3/4-body + rx on all),
    and final mz on all qubits.
    """
    nq = num_qubits_circuit(N)
    n1, n2, n3, n4 = count_reduced_terms(N)
    gates_per_step = (
        n1 * GATES_PER_1BODY_PER_STEP
        + n2 * GATES_PER_2BODY_PER_STEP
        + n3 * GATES_PER_3BODY_PER_STEP
        + n4 * GATES_PER_4BODY_PER_STEP
        + nq  # rx(2*(1-lam)*dt) on each qubit at end of step
    )
    initial_h = nq
    final_mz = nq
    return initial_h + trotter_steps * gates_per_step + final_mz


def run_scaling(min_n: int, max_n: int, stride: int, trotter_steps: int, output_dir: Path):
    """Compute gates and qubits for N in [min_n, max_n] with given stride; plot and optionally save data."""
    n_values = []
    for n in range(min_n, max_n + 1, stride):
        if n < 3:
            continue
        n_values.append(n)
    if not n_values:
        print("No N values in range (need N >= 3).")
        return

    qubits = [num_qubits_circuit(n) for n in n_values]
    gates = [total_gates(n, trotter_steps) for n in n_values]
    term_counts = [count_reduced_terms(n) for n in n_values]
    n1_list = [t[0] for t in term_counts]
    n2_list = [t[1] for t in term_counts]
    n3_list = [t[2] for t in term_counts]
    n4_list = [t[3] for t in term_counts]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Plot 1: Number of gates vs N ---
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(n_values, gates, "o-", color="C0", linewidth=2, markersize=6)
    ax1.set_xlabel("N (sequence length)")
    ax1.set_ylabel("Total gate count")
    ax1.set_title(f"DCQO circuit gate count vs N (trotter_steps={trotter_steps})")
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(n_values)
    plt.tight_layout()
    gates_plot_path = output_dir / "scaling_gates_vs_N.png"
    plt.savefig(gates_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {gates_plot_path}")

    # --- Plot 2: Number of qubits vs N ---
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(n_values, qubits, "s-", color="C1", linewidth=2, markersize=6)
    ax2.set_xlabel("N (sequence length)")
    ax2.set_ylabel("Number of qubits")
    ax2.set_title("Circuit qubit count vs N (Image Hamiltonian, reduced)")
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(n_values)
    plt.tight_layout()
    qubits_plot_path = output_dir / "scaling_qubits_vs_N.png"
    plt.savefig(qubits_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {qubits_plot_path}")

    # --- Optional: combined figure (gates + qubits) ---
    fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    ax3a.plot(n_values, gates, "o-", color="C0", linewidth=2, markersize=6)
    ax3a.set_ylabel("Total gate count")
    ax3a.set_title(f"Gate count vs N (trotter_steps={trotter_steps})")
    ax3a.grid(True, alpha=0.3)
    ax3a.set_xticks(n_values)
    ax3b.plot(n_values, qubits, "s-", color="C1", linewidth=2, markersize=6)
    ax3b.set_xlabel("N (sequence length)")
    ax3b.set_ylabel("Number of qubits")
    ax3b.set_title("Qubit count vs N")
    ax3b.grid(True, alpha=0.3)
    ax3b.set_xticks(n_values)
    plt.tight_layout()
    combined_path = output_dir / "scaling_gates_and_qubits_vs_N.png"
    plt.savefig(combined_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {combined_path}")

    # --- Print table and deterministic function summary ---
    print("\n--- Scaling table ---")
    print(f"{'N':>4} {'qubits':>7} {'gates':>10}  n1   n2   n3   n4")
    print("-" * 50)
    for n, q, g, (n1, n2, n3, n4) in zip(n_values, qubits, gates, term_counts):
        print(f"{n:>4} {q:>7} {g:>10}  {n1:>3} {n2:>3} {n3:>3} {n4:>3}")

    print("\n--- Deterministic functions (use in code) ---")
    print("  num_qubits_circuit(N)     -> qubits (N even: N-2, N odd: (N+1)//2)")
    print("  count_reduced_terms(N)    -> (n1, n2, n3, n4)")
    print("  total_gates(N, trotter_steps=1) -> total gate count")
    print("  Defined in this script; gate counts match dcqo_flexible_circuit_v2 in main.py.")

    # Save CSV
    csv_path = output_dir / "scaling_gates_qubits.csv"
    with open(csv_path, "w") as f:
        f.write("N,qubits,gates,n1,n2,n3,n4\n")
        for n, q, g, (n1, n2, n3, n4) in zip(n_values, qubits, gates, term_counts):
            f.write(f"{n},{q},{g},{n1},{n2},{n3},{n4}\n")
    print(f"\nSaved: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Scaling: gates and qubits vs N for Image Hamiltonian DCQO circuit.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--min-n", type=int, default=4, help="Minimum N (sequence length)")
    parser.add_argument("--max-n", type=int, default=24, help="Maximum N (inclusive)")
    parser.add_argument("--stride", type=int, default=2, help="Stride for N values")
    parser.add_argument(
        "--trotter-steps",
        type=int,
        default=1,
        help="Trotter steps used for gate count",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots and CSV (default: impl-trotter/results)",
    )
    args = parser.parse_args()

    if args.min_n > args.max_n:
        print("Error: --min-n must be <= --max-n")
        sys.exit(1)
    if args.stride < 1:
        print("Error: --stride must be >= 1")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else _impl_trotter_dir / "results"
    run_scaling(
        min_n=args.min_n,
        max_n=args.max_n,
        stride=args.stride,
        trotter_steps=args.trotter_steps,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
