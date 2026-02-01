"""
Trotter/Counteradiabatic + MTS Hybrid for LABS (runnable from benchmarks).

Ports the logic from 01_quantum_enhanced_optimization_LABS-checkpoint(1).ipynb:
1. Trotterized counteradiabatic circuit (Eq. 15) -> quantum samples
2. MTS with that distribution as initial population

Usage:
    from main import run_hybrid, run_hybrid_h100_optimized, h100_optimized
    best_seq, time_sec = run_hybrid(N, verbose=False)
    best_seq, time_sec = run_hybrid_h100_optimized(N, verbose=False)  # GPU path
Returns best_seq as list of ±1 for eval_util compatibility.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time
import random
import math

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


# Public alias for generate_trotterization API / qe_mts_image_hamiltonian
get_interactions = _get_interactions

# ---------------------------------------------------------------------------
# Image Hamiltonian & DCQO helpers (from generate_trotterization)
# ---------------------------------------------------------------------------

FIXED_FIRST_TWO_PREFIX = "11"
"""When using reduced circuit, prepend this to sampled bitstrings to get full N-bit string."""


def reduce_hamiltonian_fix_first_two(terms_1, terms_2, terms_3, terms_4):
    """
    Reduce the full N-qubit Hamiltonian to N-2 qubits by fixing qubits 0 and 1 to |1⟩
    (bit 1 → σ^z eigenvalue -1). Returns (t1_red, t2_red, t3_red, t4_red) with indices
    remapped so original index i (≥2) becomes i-2. Terms involving only qubits 0,1 become
    a constant (dropped; they do not affect which state minimizes energy).
    """
    z_fixed = -1  # eigenvalue of σ^z for |1⟩
    t1_red = {}
    t2_red = {}
    t3_red = {}
    t4_red = {}

    def to_new(i):
        return None if i in (0, 1) else i - 2

    for term in terms_1:
        i, w = int(term[0]), term[1]
        if i in (0, 1):
            continue  # constant z_fixed * w, dropped
        t1_red[to_new(i)] = t1_red.get(to_new(i), 0.0) + w

    for term in terms_2:
        i, j, w = int(term[0]), int(term[1]), term[2]
        ni, nj = to_new(i), to_new(j)
        if ni is None and nj is None:
            continue
        if ni is None and nj is not None:
            t1_red[nj] = t1_red.get(nj, 0.0) + (z_fixed * w)
        elif ni is not None and nj is None:
            t1_red[ni] = t1_red.get(ni, 0.0) + (z_fixed * w)
        else:
            key = (min(ni, nj), max(ni, nj))
            t2_red[key] = t2_red.get(key, 0.0) + w

    for term in terms_3:
        i, j, k = int(term[0]), int(term[1]), int(term[2])
        w = term[3]
        ni, nj, nk = to_new(i), to_new(j), to_new(k)
        fixed_count = (1 if ni is None else 0) + (1 if nj is None else 0) + (1 if nk is None else 0)
        coeff = (z_fixed ** fixed_count) * w
        idxs = [ni, nj, nk]
        active = [x for x in idxs if x is not None]
        if len(active) == 3:
            key = tuple(sorted(active))
            t3_red[key] = t3_red.get(key, 0.0) + coeff
        elif len(active) == 2:
            key = (min(active), max(active))
            t2_red[key] = t2_red.get(key, 0.0) + coeff
        elif len(active) == 1:
            t1_red[active[0]] = t1_red.get(active[0], 0.0) + coeff
        # else constant, drop

    for term in terms_4:
        qa, qb, qc, qd = int(term[0]), int(term[1]), int(term[2]), int(term[3])
        w = term[4]
        na, nb, nc, nd = to_new(qa), to_new(qb), to_new(qc), to_new(qd)
        fixed_count = sum(1 for x in (na, nb, nc, nd) if x is None)
        coeff = (z_fixed ** fixed_count) * w
        active = [x for x in (na, nb, nc, nd) if x is not None]
        if len(active) == 4:
            key = tuple(sorted(active))
            t4_red[key] = t4_red.get(key, 0.0) + coeff
        elif len(active) == 3:
            key = tuple(sorted(active))
            t3_red[key] = t3_red.get(key, 0.0) + coeff
        elif len(active) == 2:
            key = (min(active), max(active))
            t2_red[key] = t2_red.get(key, 0.0) + coeff
        elif len(active) == 1:
            t1_red[active[0]] = t1_red.get(active[0], 0.0) + coeff
        # else constant, drop

    t1_list = [[i, float(w)] for i, w in t1_red.items()]
    t2_list = [[i, j, float(w)] for (i, j), w in t2_red.items()]
    t3_list = [[i, j, k, float(w)] for (i, j, k), w in t3_red.items()]
    t4_list = [[a, b, c, d, float(w)] for (a, b, c, d), w in t4_red.items()]
    return t1_list, t2_list, t3_list, t4_list


def prepend_fixed_prefix_to_counts(sample_result, prefix: str = FIXED_FIRST_TWO_PREFIX):
    """
    Return a counts dict with keys prefix + bitstring for each bitstring in sample_result.
    Use this so that calculate_energy receives full N-bit strings when the circuit
    was run with N-2 qubits and fixed first two bits.
    """
    return {prefix + bs: count for bs, count in sample_result.items()}


def reduce_hamiltonian_skew_symmetry(terms_1, terms_2, terms_3, terms_4, N: int):
    """
    Reduce the N-qubit Hamiltonian to qubits 0..a by substituting Z_i (i > a)
    with (-1)^(i-a) Z_{2a-i}, where a = (N-1)/2 (0-based). Requires N odd.
    Returns (t1_red, t2_red, t3_red, t4_red) with indices only in [0, a].
    Terms that map to duplicate qubit indices are coalesced (Z_i^2 = I, so
    duplicate pairs drop or reduce order) so the circuit never sees repeated
    qubits in a single gate, avoiding cusvsim gate-grouping errors.
    """
    if N % 2 == 0:
        raise ValueError("Skew-symmetry reduction requires odd N")
    a = (N - 1) // 2
    t1_red = {}
    t2_red = {}
    t3_red = {}
    t4_red = {}

    def map_idx(i):
        i = int(i)
        return i if i <= a else 2 * a - i

    def phase_exponent(indices):
        return sum((idx - a) for idx in indices if idx > a)

    def add_1(i, c):
        t1_red[i] = t1_red.get(i, 0.0) + c

    def add_2(i, j, c):
        if i == j:
            return  # Z_i Z_i = I
        i, j = min(i, j), max(i, j)
        key = (i, j)
        t2_red[key] = t2_red.get(key, 0.0) + c

    def add_3(i, j, k, c):
        si, sj, sk = sorted([i, j, k])
        if si == sj == sk:
            add_1(si, c)  # Z^3 = Z
        elif si == sj:
            add_1(sk, c)  # Z_si^2 Z_sk = Z_sk
        elif sj == sk:
            add_1(si, c)  # Z_si Z_sj^2 = Z_si
        else:
            key = (si, sj, sk)
            t3_red[key] = t3_red.get(key, 0.0) + c

    def add_4(na, nb, nc, nd, c):
        s = sorted([na, nb, nc, nd])
        a4, b4, c4, d4 = s[0], s[1], s[2], s[3]
        if a4 == b4 == c4 == d4:
            return  # Z^4 = I
        if a4 == b4 == c4:
            add_2(a4, d4, c)  # Z_a^3 Z_d = Z_a Z_d
        elif b4 == c4 == d4:
            add_2(a4, b4, c)  # Z_a Z_b^3 = Z_a Z_b
        elif a4 == b4 and c4 == d4:
            return  # Z_a^2 Z_c^2 = I
        elif a4 == b4:
            add_2(c4, d4, c)  # Z_a^2 Z_c Z_d = Z_c Z_d
        elif c4 == d4:
            add_2(a4, b4, c)  # Z_a Z_b Z_c^2 = Z_a Z_b
        elif b4 == c4:
            add_2(a4, d4, c)  # Z_a Z_b^2 Z_d = Z_a Z_d
        else:
            key = (a4, b4, c4, d4)
            t4_red[key] = t4_red.get(key, 0.0) + c

    for term in terms_1:
        i, w = int(term[0]), term[1]
        exp = phase_exponent([i])
        coeff = ((-1) ** exp) * w
        i_new = map_idx(i)
        add_1(i_new, coeff)

    for term in terms_2:
        i, j, w = int(term[0]), int(term[1]), term[2]
        exp = phase_exponent([i, j])
        coeff = ((-1) ** exp) * w
        i_new, j_new = map_idx(i), map_idx(j)
        add_2(i_new, j_new, coeff)

    for term in terms_3:
        i, j, k = int(term[0]), int(term[1]), int(term[2])
        w = term[3]
        exp = phase_exponent([i, j, k])
        coeff = ((-1) ** exp) * w
        i_new, j_new, k_new = map_idx(i), map_idx(j), map_idx(k)
        add_3(i_new, j_new, k_new, coeff)

    for term in terms_4:
        qa, qb, qc, qd = int(term[0]), int(term[1]), int(term[2]), int(term[3])
        w = term[4]
        exp = phase_exponent([qa, qb, qc, qd])
        coeff = ((-1) ** exp) * w
        na, nb, nc, nd = map_idx(qa), map_idx(qb), map_idx(qc), map_idx(qd)
        add_4(na, nb, nc, nd, coeff)

    t1_list = [[i, float(w)] for i, w in t1_red.items()]
    t2_list = [[i, j, float(w)] for (i, j), w in t2_red.items()]
    t3_list = [[i, j, k, float(w)] for (i, j, k), w in t3_red.items()]
    t4_list = [[a, b, c, d, float(w)] for (a, b, c, d), w in t4_red.items()]
    return t1_list, t2_list, t3_list, t4_list


def expand_skew_symmetric_bitstring(bitstring: str, N: int) -> str:
    """
    Expand a reduced bitstring (length (N+1)//2, indices 0..a) to full N-bit string
    using skew-symmetry s_i = (-1)^(i-a) s_{2a-i}: for i > a, bit[i] = bit[2a-i] XOR ((i-a) % 2).
    """
    if N % 2 == 0:
        raise ValueError("Skew-symmetry expansion requires odd N")
    a = (N - 1) // 2
    n_reduced = a + 1
    if len(bitstring) != n_reduced:
        raise ValueError(f"Expected bitstring length {n_reduced} for N={N}, got {len(bitstring)}")
    out = list(bitstring)
    for i in range(a + 1, N):
        mirror = 2 * a - i
        flip = (i - a) % 2
        out.append(str(int(bitstring[mirror]) ^ flip))
    return "".join(out)


def expand_skew_symmetric_counts(sample_result, N: int) -> dict:
    """
    Return a counts dict with full N-bit string keys by expanding each reduced
    bitstring via expand_skew_symmetric_bitstring. Use after sampling the
    skew-reduced circuit so downstream calculate_energy and MTS see full-length bitstrings.
    """
    if N % 2 == 0:
        raise ValueError("Skew-symmetry expansion requires odd N")
    result = {}
    for bs, count in sample_result.items():
        full = expand_skew_symmetric_bitstring(bs, N)
        result[full] = result.get(full, 0) + count
    return result


def get_labs_hamiltonian(N):
    """
    Generates 2-body and 4-body terms for standard LABS (Energy = sum_k (sum_i s_i s_{i+k})^2).
    Returns 4 lists to be compatible with dcqo_flexible_circuit_v2.
    """
    terms_2 = {}
    terms_4 = {}
    for k in range(1, N):
        valid = range(N - k)
        for i in valid:
            for j in valid:
                if i == j:
                    continue
                indices = sorted([i, i + k, j, j + k])
                counts = {x: indices.count(x) for x in set(indices)}
                unique = sorted([x for x, c in counts.items() if c % 2 == 1])
                if len(unique) == 2:
                    terms_2[tuple(unique)] = terms_2.get(tuple(unique), 0.0) + 1.0
                elif len(unique) == 4:
                    terms_4[tuple(unique)] = terms_4.get(tuple(unique), 0.0) + 1.0
    t2_list = [[*k, w] for k, w in terms_2.items()]
    t4_list = [[*k, w] for k, w in terms_4.items()]
    return [], t2_list, [], t4_list


def calculate_energy(counts, t1=None, t2=None, t3=None, t4=None):
    """
    Compute average and min LABS energy over bitstrings in counts.
    counts: dict-like (bitstring -> count) or cudaq SampleResult.
    t1..t4 are unused (kept for API compatibility).
    """
    min_energy = float("inf")
    avg_energy = 0.0
    total_shots = sum(counts.values())
    for bitstring, count in counts.items():
        spins = [1 if b == "0" else -1 for b in bitstring]
        N = len(spins)
        E_sample = 0.0
        for k in range(1, N):
            Ck = 0
            for i in range(N - k):
                Ck += spins[i] * spins[i + k]
            E_sample += Ck ** 2
        prob = count / total_shots
        avg_energy += E_sample * prob
        if E_sample < min_energy:
            min_energy = E_sample
    return avg_energy, min_energy


def get_image_hamiltonian(N):
    """
    Generates 1, 2, 3, and 4-body terms based on the H_f equation in the image.
    Returns a tuple of lists: (terms_1, terms_2, terms_3, terms_4)
    Each term includes indices and the weight (coefficient).
    """
    terms_1 = []
    terms_2 = []
    terms_3 = []
    terms_4 = []
    upper_1 = math.floor((N - 1) / 2)
    for k in range(2, upper_1 + 1):
        idx = (1 + k) - 1
        terms_1.append([idx, 2.0])
    upper_2 = math.floor((N - 2) / 2)
    for k in range(1, upper_2 + 1):
        idx = (2 + k) - 1
        terms_1.append([idx, 2.0])
    for i in range(3, N - 2 + 1):
        upper_k = math.floor((N - i) / 2)
        for k in range(1, upper_k + 1):
            idx1 = i - 1
            idx2 = (i + k) - 1
            terms_2.append([idx1, idx2, 2.0])
    for k in range(2, N - 2 + 1):
        idx1 = (1 + k) - 1
        idx2 = (1 + k + 1) - 1
        terms_2.append([idx1, idx2, 4.0])
    upper_t = math.floor((N - 2) / 2)
    for t in range(2, upper_t + 1):
        for k in range(t + 1, (N - 1 - t) + 1):
            idx1 = (1 + t) - 1
            idx2 = (1 + k) - 1
            idx3 = (1 + k + t) - 1
            terms_3.append([idx1, idx2, idx3, 4.0])
    upper_t_6 = math.floor((N - 3) / 2)
    for t in range(1, upper_t_6 + 1):
        for k in range(t + 1, (N - 2 - t) + 1):
            idx1 = (2 + t) - 1
            idx2 = (2 + k) - 1
            idx3 = (2 + k + t) - 1
            terms_3.append([idx1, idx2, idx3, 4.0])
    for i in range(3, N - 3 + 1):
        upper_t_7 = math.floor((N - i - 1) / 2)
        for t in range(1, upper_t_7 + 1):
            upper_k_7 = N - i - t
            for k in range(t + 1, upper_k_7 + 1):
                idx1 = i - 1
                idx2 = (i + t) - 1
                idx3 = (i + k) - 1
                idx4 = (i + k + t) - 1
                terms_4.append([idx1, idx2, idx3, idx4, 4.0])
    return terms_1, terms_2, terms_3, terms_4


def get_image_hamiltonian_skew_reduced(N: int):
    """
    Image Hamiltonian reduced by skew-symmetry (N must be odd). Returns
    (t1r, t2r, t3r, t4r, num_qubits) where num_qubits = (N+1)//2. Callers must
    use expand_skew_symmetric_bitstring or expand_skew_symmetric_counts after sampling.
    """
    if N % 2 == 0:
        raise ValueError("Skew-symmetry requires odd N")
    t1, t2, t3, t4 = get_image_hamiltonian(N)
    t1r, t2r, t3r, t4r = reduce_hamiltonian_skew_symmetry(t1, t2, t3, t4, N)
    num_qubits = (N + 1) // 2
    return t1r, t2r, t3r, t4r, num_qubits


# ---------------------------------------------------------------------------
# Trotter Circuit (from notebook Eq. 15)
# ---------------------------------------------------------------------------


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
# DCQO gate kernels and flexible circuits (from generate_trotterization)
# ---------------------------------------------------------------------------

if _CUDAQ_AVAILABLE:

    @cudaq.kernel
    def r_zz(theta: float, q_i: cudaq.qubit, q_j: cudaq.qubit):
        """Problem Term: exp(-i * theta * Z_i Z_j)"""
        x.ctrl(q_i, q_j)
        rz(2.0 * theta, q_j)
        x.ctrl(q_i, q_j)

    @cudaq.kernel
    def r_zzzz(theta: float, q0: cudaq.qubit, q1: cudaq.qubit, q2: cudaq.qubit, q3: cudaq.qubit):
        """Problem Term: exp(-i * theta * Z_i Z_j Z_k Z_l)"""
        x.ctrl(q0, q1)
        x.ctrl(q1, q2)
        x.ctrl(q2, q3)
        rz(2.0 * theta, q3)
        x.ctrl(q2, q3)
        x.ctrl(q1, q2)
        x.ctrl(q0, q1)

    @cudaq.kernel
    def r_yz(theta: float, q_i: cudaq.qubit, q_j: cudaq.qubit):
        """2-body CD Term: exp(-i * theta * Y_i Z_j)."""
        rx(np.pi / 2, q_i)
        x.ctrl(q_i, q_j)
        rz(2.0 * theta, q_j)
        x.ctrl(q_i, q_j)
        rx(-np.pi / 2, q_i)

    @cudaq.kernel
    def r_z(theta: float, q: cudaq.qubit):
        """Problem Term: exp(-i * theta * Z_i)"""
        rz(2.0 * theta, q)

    @cudaq.kernel
    def r_zzz(theta: float, q0: cudaq.qubit, q1: cudaq.qubit, q2: cudaq.qubit):
        """Problem Term: exp(-i * theta * Z_i Z_j Z_k)"""
        x.ctrl(q0, q1)
        x.ctrl(q1, q2)
        rz(2.0 * theta, q2)
        x.ctrl(q1, q2)
        x.ctrl(q0, q1)

    @cudaq.kernel
    def r_yzzz(theta: float, q0: cudaq.qubit, q1: cudaq.qubit, q2: cudaq.qubit, q3: cudaq.qubit):
        """4-body CD Term: exp(-i * theta * Y_a Z_b Z_c Z_d)."""
        rx(np.pi / 2, q0)
        x.ctrl(q0, q1)
        x.ctrl(q1, q2)
        x.ctrl(q2, q3)
        rz(2.0 * theta, q3)
        x.ctrl(q2, q3)
        x.ctrl(q1, q2)
        x.ctrl(q0, q1)
        rx(-np.pi / 2, q0)

    @cudaq.kernel
    def dcqo_flexible_circuit(
        num_qubits: int,
        num_steps: int,
        terms_2body: list[list[float]],
        terms_4body: list[list[float]],
        lambda_sched: list[float],
        lambda_dot_sched: list[float],
        dt: float,
    ):
        """Trotterized circuit for 2/4-body Hamiltonian with CD driving."""
        q = cudaq.qvector(num_qubits)
        h(q)
        for step in range(num_steps):
            lam = lambda_sched[step]
            lam_dot = lambda_dot_sched[step]
            for term in terms_2body:
                i, j = int(term[0]), int(term[1])
                theta_cd = term[2] * lam_dot * dt
                r_yz(theta_cd, q[i], q[j])
                r_yz(theta_cd, q[j], q[i])
            for term in terms_4body:
                qa, qb, qc, qd = int(term[0]), int(term[1]), int(term[2]), int(term[3])
                theta_cd = term[4] * lam_dot * dt
                r_yzzz(theta_cd, q[qa], q[qb], q[qc], q[qd])
                r_yzzz(theta_cd, q[qb], q[qa], q[qc], q[qd])
                r_yzzz(theta_cd, q[qc], q[qa], q[qb], q[qd])
                r_yzzz(theta_cd, q[qd], q[qa], q[qb], q[qc])
            for term in terms_2body:
                i, j = int(term[0]), int(term[1])
                r_zz(term[2] * lam * dt, q[i], q[j])
            for term in terms_4body:
                qa, qb, qc, qd = int(term[0]), int(term[1]), int(term[2]), int(term[3])
                r_zzzz(term[4] * lam * dt, q[qa], q[qb], q[qc], q[qd])
            rx(2.0 * (1.0 - lam) * dt, q)
        mz(q)

    @cudaq.kernel
    def dcqo_flexible_circuit_v2(
        num_qubits: int,
        num_steps: int,
        terms_1body: list[list[float]],
        terms_2body: list[list[float]],
        terms_3body: list[list[float]],
        terms_4body: list[list[float]],
        lambda_sched: list[float],
        lambda_dot_sched: list[float],
        dt: float,
    ):
        """Trotterized circuit for 1/2/3/4-body Hamiltonian with CD driving."""
        q = cudaq.qvector(num_qubits)
        h(q)
        for step in range(num_steps):
            lam = lambda_sched[step]
            lam_dot = lambda_dot_sched[step]
            for term in terms_2body:
                i, j = int(term[0]), int(term[1])
                theta_cd = term[2] * lam_dot * dt
                r_yz(theta_cd, q[i], q[j])
                r_yz(theta_cd, q[j], q[i])
            for term in terms_4body:
                qa, qb, qc, qd = int(term[0]), int(term[1]), int(term[2]), int(term[3])
                theta_cd = term[4] * lam_dot * dt
                r_yzzz(theta_cd, q[qa], q[qb], q[qc], q[qd])
                r_yzzz(theta_cd, q[qb], q[qa], q[qc], q[qd])
                r_yzzz(theta_cd, q[qc], q[qa], q[qb], q[qd])
                r_yzzz(theta_cd, q[qd], q[qa], q[qb], q[qc])
            for term in terms_1body:
                i = int(term[0])
                r_z(term[1] * lam * dt, q[i])
            for term in terms_2body:
                i, j = int(term[0]), int(term[1])
                r_zz(term[2] * lam * dt, q[i], q[j])
            for term in terms_3body:
                i, j, k = int(term[0]), int(term[1]), int(term[2])
                r_zzz(term[3] * lam * dt, q[i], q[j], q[k])
            for term in terms_4body:
                qa, qb, qc, qd = int(term[0]), int(term[1]), int(term[2]), int(term[3])
                r_zzzz(term[4] * lam * dt, q[qa], q[qb], q[qc], q[qd])
            rx(2.0 * (1.0 - lam) * dt, q)
        mz(q)

else:
    r_zz = r_zzzz = r_yz = r_z = r_zzz = r_yzzz = None
    dcqo_flexible_circuit = None
    dcqo_flexible_circuit_v2 = None


# ---------------------------------------------------------------------------
# MTS Loader
# ---------------------------------------------------------------------------


def _load_mts(repo_root: Path, use_gpu: bool = False):
    """Load memetic_tabu_search from impl-mts/main.py or mts_h100_optimized.py when use_gpu=True."""
    import importlib.util

    if use_gpu:
        mts_path = repo_root / "impl-mts" / "mts_h100_optimized.py"
    else:
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
    use_gpu_mts: bool = False,
) -> tuple:
    """
    Run Trotterized counteradiabatic + MTS hybrid.
    Returns (best_sequence_as_list, time_sec).
    best_sequence is list of ±1 for eval_util compatibility.
    When use_gpu_mts=True, uses H100-optimized MTS from impl-mts/mts_h100_optimized.py.
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

    mts_module = _load_mts(REPO_ROOT, use_gpu=use_gpu_mts)
    random.seed(42)
    np.random.seed(42)
    if use_gpu_mts and hasattr(mts_module, "cp"):
        mts_module.cp.random.seed(42)
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


# ---------------------------------------------------------------------------
# H100-Optimized Entry: run_hybrid_h100_optimized
# ---------------------------------------------------------------------------


def run_hybrid_h100_optimized(
    N: int,
    T: float = 1.0,
    n_steps: int = 1,
    shots: int = 200,
    population_size: int = 50,
    max_generations: int = 30,
    p_combine: float = 0.9,
    verbose: bool = False,
    nvidia_option: str = "fp32",
) -> tuple:
    """
    H100-optimized Trotterized counteradiabatic + MTS hybrid.

    Sets cudaq target to 'nvidia' (GPU) before sampling and uses
    H100-optimized MTS from impl-mts/mts_h100_optimized.py.
    Requires an NVIDIA GPU. Optional precision via nvidia_option ('fp32' or 'fp64').
    Advanced tuning: set env vars CUDAQ_FUSION_*, CUDAQ_MAX_GPU_MEMORY_GB before run.

    Returns (best_sequence_as_list, time_sec). best_sequence is list of ±1 for eval_util.
    """
    start = time.perf_counter()

    if not _CUDAQ_AVAILABLE:
        raise RuntimeError("cudaq required for trotter method")

    cudaq.set_target("nvidia", option=nvidia_option)

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
        N,
        G2,
        G4,
        n_steps,
        dt,
        T,
        thetas,
        shots_count=shots,
    )

    quantum_population = []
    for bitstring, count in quantum_result.items():
        seq = _bitstring_to_sequence(bitstring)
        for _ in range(count):
            quantum_population.append(seq.copy())

    # Sync GPU so cudaq work is done before CuPy MTS runs (avoids allocator/heap conflicts)
    try:
        import cupy as _cp
        _cp.cuda.Stream.null.synchronize()
    except Exception:
        pass

    mts_module = _load_mts(REPO_ROOT, use_gpu=True)
    random.seed(42)
    np.random.seed(42)
    if hasattr(mts_module, "cp"):
        mts_module.cp.random.seed(42)
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


# Alias for backward compatibility and convenience.
h100_optimized = run_hybrid_h100_optimized


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int, nargs="?", default=10, help="Sequence length")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--gpu", "--h100", dest="use_h100", action="store_true", help="Use H100-optimized path (cudaq nvidia + GPU MTS)")
    args = parser.parse_args()
    if args.use_h100:
        seq, t = run_hybrid_h100_optimized(args.N, verbose=args.verbose)
    else:
        seq, t = run_hybrid(args.N, verbose=args.verbose)
    print(f"N={args.N} best sequence (±1): {seq}")
    print(f"Time: {t:.4f}s")
