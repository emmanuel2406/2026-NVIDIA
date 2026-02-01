import cudaq
import numpy as np

# ---------------------------------------------------------------------------
# 1. Gate Kernels (Verified against Paper Ansatz)
# ---------------------------------------------------------------------------

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
    """
    2-body CD Term: exp(-i * theta * Y_i Z_j).
    Implementation: Rx(pi/2) on target -> Z_i Z_j -> Rx(-pi/2) on target.
    """
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
    # CNOT ladder to compute parity
    x.ctrl(q0, q1)
    x.ctrl(q1, q2)
    rz(2.0 * theta, q2)
    x.ctrl(q1, q2)
    x.ctrl(q0, q1)

@cudaq.kernel
def r_yzzz(theta: float, q0: cudaq.qubit, q1: cudaq.qubit, q2: cudaq.qubit, q3: cudaq.qubit):
    """
    4-body CD Term: exp(-i * theta * Y_a Z_b Z_c Z_d).
    Implementation: Rx(pi/2) on q0 -> ZZZZ -> Rx(-pi/2) on q0.
    """
    rx(np.pi / 2, q0)
    x.ctrl(q0, q1)
    x.ctrl(q1, q2)
    x.ctrl(q2, q3)
    rz(2.0 * theta, q3)
    x.ctrl(q2, q3)
    x.ctrl(q1, q2)
    x.ctrl(q0, q1)
    rx(-np.pi / 2, q0)

# ---------------------------------------------------------------------------
# 2. Main DCQO Transformer
# ---------------------------------------------------------------------------

@cudaq.kernel
def dcqo_flexible_circuit(
    num_qubits: int, 
    num_steps: int,
    terms_2body: list[list[float]], 
    terms_4body: list[list[float]],
    lambda_sched: list[float],
    lambda_dot_sched: list[float],
    dt: float
):
    """
    Constructs the Trotterized circuit for ANY Hamiltonian defined by terms_2body and terms_4body.
    Automatically adds Counterdiabatic (CD) driving layers.
    """
    q = cudaq.qvector(num_qubits)
    h(q) # Initialize |+>

    for step in range(num_steps):
        lam = lambda_sched[step]
        lam_dot = lambda_dot_sched[step]
        
        # --- Layer 1: CD Driving (proportional to lambda_dot) ---
        # 2-Body CD: Apply YZ + ZY for every pair
        for term in terms_2body:
            i, j = int(term[0]), int(term[1])
            theta_cd = term[2] * lam_dot * dt
            r_yz(theta_cd, q[i], q[j]) # Y_i Z_j
            r_yz(theta_cd, q[j], q[i]) # Z_i Y_j

        # 4-Body CD: Apply YZZZ + ZYZZ + ZZYZ + ZZZY
        for term in terms_4body:
            qa, qb, qc, qd = int(term[0]), int(term[1]), int(term[2]), int(term[3])
            theta_cd = term[4] * lam_dot * dt
            # Cycle the Y operator through all 4 positions
            r_yzzz(theta_cd, q[qa], q[qb], q[qc], q[qd])
            r_yzzz(theta_cd, q[qb], q[qa], q[qc], q[qd])
            r_yzzz(theta_cd, q[qc], q[qa], q[qb], q[qd])
            r_yzzz(theta_cd, q[qd], q[qa], q[qb], q[qc])

        # --- Layer 2: Problem Hamiltonian (proportional to lambda) ---
        for term in terms_2body:
            i, j = int(term[0]), int(term[1])
            r_zz(term[2] * lam * dt, q[i], q[j])

        for term in terms_4body:
            qa, qb, qc, qd = int(term[0]), int(term[1]), int(term[2]), int(term[3])
            r_zzzz(term[4] * lam * dt, q[qa], q[qb], q[qc], q[qd])

        # --- Layer 3: Mixer (proportional to 1-lambda) ---
        rx(2.0 * (1.0 - lam) * dt, q)

    mz(q)

@cudaq.kernel
def dcqo_flexible_circuit_v2(
    num_qubits: int, 
    num_steps: int,
    # Updated arguments to accept all term types
    terms_1body: list[list[float]],
    terms_2body: list[list[float]], 
    terms_3body: list[list[float]],
    terms_4body: list[list[float]],
    lambda_sched: list[float],
    lambda_dot_sched: list[float],
    dt: float
):
    q = cudaq.qvector(num_qubits)
    h(q)

    for step in range(num_steps):
        lam = lambda_sched[step]
        lam_dot = lambda_dot_sched[step]
        
        # --- Layer 1: CD Driving (Existing 2-body/4-body logic) ---
        # (Kept same as your original code for 2/4 body terms)
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

        # --- Layer 2: Problem Hamiltonian (UPDATED) ---
        
        # 1-Body Terms (New)
        for term in terms_1body:
            i = int(term[0])
            r_z(term[1] * lam * dt, q[i])

        # 2-Body Terms
        for term in terms_2body:
            i, j = int(term[0]), int(term[1])
            r_zz(term[2] * lam * dt, q[i], q[j])

        # 3-Body Terms (New)
        for term in terms_3body:
            i, j, k = int(term[0]), int(term[1]), int(term[2])
            r_zzz(term[3] * lam * dt, q[i], q[j], q[k])

        # 4-Body Terms
        for term in terms_4body:
            qa, qb, qc, qd = int(term[0]), int(term[1]), int(term[2]), int(term[3])
            r_zzzz(term[4] * lam * dt, q[qa], q[qb], q[qc], q[qd])

        # --- Layer 3: Mixer ---
        rx(2.0 * (1.0 - lam) * dt, q)

    mz(q)

# ---------------------------------------------------------------------------
# 2b. Fixed first two qubits as |1⟩|1⟩ (bits "11")
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


# ---------------------------------------------------------------------------
# 2b2. Skew-symmetry: reduce to (N+1)/2 qubits when N is odd
# ---------------------------------------------------------------------------

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
# 2c. Interaction sets G2, G4 (paper Eq. 15 / tutorial Exercise 4)
# ---------------------------------------------------------------------------

def get_interactions(N: int):
    """
    Generate two-body (G2) and four-body (G4) interaction index sets for the
    LABS DCQO circuit (paper Eq. 15; tutorial Exercise 4).
    Used with compute_theta(t, dt, total_time, N, G2, G4) from labs_utils
    for the CD angle schedule.
    Returns:
        G2: list of [i, i+k] for i = 0..N-3, k = 1..floor((N-i-1)/2)
        G4: list of [i, i+t, i+k, i+k+t] for the triple loop (i, t, k)
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


# ---------------------------------------------------------------------------
# 3. Helper to Generate LABS Hamiltonian
# ---------------------------------------------------------------------------

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
                if i == j: continue 
                indices = sorted([i, i+k, j, j+k])
                counts = {x: indices.count(x) for x in set(indices)}
                # Keep indices that appear an odd number of times (c % 2 == 1)
                unique = sorted([x for x, c in counts.items() if c % 2 == 1])
                
                if len(unique) == 2:
                    terms_2[tuple(unique)] = terms_2.get(tuple(unique), 0.0) + 1.0
                elif len(unique) == 4:
                    terms_4[tuple(unique)] = terms_4.get(tuple(unique), 0.0) + 1.0

    # Format: [[idx1, idx2, weight], ...]
    t2_list = [[*k, w] for k, w in terms_2.items()]
    t4_list = [[*k, w] for k, w in terms_4.items()]
    
    # Return empty lists for 1-body and 3-body terms
    return [], t2_list, [], t4_list

def calculate_energy(counts, t1=None, t2=None, t3=None, t4=None):
    """
    Compute average and min LABS energy over bitstrings in counts.
    counts: dict-like (bitstring -> count) or cudaq SampleResult.
    t1..t4 are unused (kept for API compatibility).
    """
    min_energy = float('inf')
    avg_energy = 0.0
    total_shots = sum(counts.values())

    for bitstring, count in counts.items():
        # 1. Convert bitstring to spins: '0' -> +1, '1' -> -1
        # Note: Adjust order if your backend is little-endian vs big-endian
        spins = [1 if b == '0' else -1 for b in bitstring]
        N = len(spins)
        
        # 2. Calculate E = sum_{k=1}^{N-1} (C_k)^2
        E_sample = 0.0
        for k in range(1, N):
            Ck = 0
            for i in range(N - k):
                Ck += spins[i] * spins[i+k]
            E_sample += Ck**2
            
        # 3. Accumulate statistics
        prob = count / total_shots
        avg_energy += E_sample * prob
        if E_sample < min_energy:
            min_energy = E_sample
            
    return avg_energy, min_energy

import math

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

    # Term 1: 1-body
    # Sum k=2 to floor((N-1)/2) of Z_{1+k}
    upper_1 = math.floor((N - 1) / 2)
    for k in range(2, upper_1 + 1):
        idx = (1 + k) - 1
        terms_1.append([idx, 2.0])

    # Term 2: 1-body
    # Sum k=1 to floor((N-2)/2) of Z_{2+k}
    upper_2 = math.floor((N - 2) / 2)
    for k in range(1, upper_2 + 1):
        idx = (2 + k) - 1
        terms_1.append([idx, 2.0])

    # Term 3: 2-body
    # Sum i=3 to N-2, Sum k=1 to floor((N-i)/2) of Z_i Z_{i+k}
    for i in range(3, N - 2 + 1):
        upper_k = math.floor((N - i) / 2)
        for k in range(1, upper_k + 1):
            idx1 = i - 1
            idx2 = (i + k) - 1
            terms_2.append([idx1, idx2, 2.0])

    # Term 4: 2-body
    # Sum k=2 to N-2 of Z_{1+k} Z_{1+k+1}
    for k in range(2, N - 2 + 1):
        idx1 = (1 + k) - 1
        idx2 = (1 + k + 1) - 1
        terms_2.append([idx1, idx2, 4.0])

    # Term 5: 3-body
    # Sum t=2 to floor((N-2)/2), Sum k=t+1 to N-1-t
    upper_t = math.floor((N - 2) / 2)
    for t in range(2, upper_t + 1):
        for k in range(t + 1, (N - 1 - t) + 1):
            idx1 = (1 + t) - 1
            idx2 = (1 + k) - 1
            idx3 = (1 + k + t) - 1
            terms_3.append([idx1, idx2, idx3, 4.0])

    # Term 6: 3-body
    # Sum t=1 to floor((N-3)/2), Sum k=t+1 to N-2-t
    upper_t_6 = math.floor((N - 3) / 2)
    for t in range(1, upper_t_6 + 1):
        for k in range(t + 1, (N - 2 - t) + 1):
            idx1 = (2 + t) - 1
            idx2 = (2 + k) - 1
            idx3 = (2 + k + t) - 1
            terms_3.append([idx1, idx2, idx3, 4.0])

    # Term 7: 4-body
    # Sum i=3 to N-3, Sum t=1 to floor((N-i-1)/2), Sum k=t+1 to N-i-t
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
    
# ---------------------------------------------------------------------------
# 4. Example Usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ---------------- Configuration ----------------
    N = 20              # System size (small enough for quick simulation)
    steps = 10         # Trotter steps
    total_time = 2.0   # Annealing time
    shots = 2000       # Sampling shots
    # First two qubits are fixed as |1⟩|1⟩ (bits "11"); circuit uses N-2 qubits only.
    num_qubits_circuit = N - 2

    # Annealing Schedule
    t_points = np.linspace(0, total_time, steps)
    dt = total_time / steps
    lambda_sched = np.sin((np.pi/2) * (t_points / total_time))**2
    lambda_dot_sched = (np.pi/total_time) * np.sin(np.pi * t_points / total_time) / 2.0

    print(f"--- Comparing Hamiltonians for N={N} ---\n")

    # ---------------- 1. Default LABS Hamiltonian (full N qubits, no reduction) ----------------
    print(">> Processing Default LABS Hamiltonian...")
    d_t1, d_t2, d_t3, d_t4 = get_labs_hamiltonian(N)

    print(f"   Terms: {len(d_t1)} (1-body), {len(d_t2)} (2-body), {len(d_t3)} (3-body), {len(d_t4)} (4-body)")

    d_flat = (
        [list(map(float, t)) for t in d_t1],
        [list(map(float, t)) for t in d_t2],
        [list(map(float, t)) for t in d_t3],
        [list(map(float, t)) for t in d_t4]
    )

    res_default = cudaq.sample(
        dcqo_flexible_circuit_v2, N, steps,
        d_flat[0], d_flat[1], d_flat[2], d_flat[3],
        lambda_sched.tolist(), lambda_dot_sched.tolist(), dt,
        shots_count=shots
    )
    avg_energy_default, min_energy_default = calculate_energy(res_default, d_t1, d_t2, d_t3, d_t4)
    print(f"   Result: <E> = {avg_energy_default:.4f}")
    print(f"   Top bitstring: {res_default.most_probable()}\n")

    # ---------------- 2. Image H_f Hamiltonian (first two bits fixed as 11, N-2 qubits) ----------------
    print(">> Processing Image H_f Hamiltonian (first two bits fixed as 11)...")
    i_t1, i_t2, i_t3, i_t4 = get_image_hamiltonian(N)
    i_t1r, i_t2r, i_t3r, i_t4r = reduce_hamiltonian_fix_first_two(i_t1, i_t2, i_t3, i_t4)

    print(f"   Full terms: {len(i_t1)} (1-body), {len(i_t2)} (2-body), {len(i_t3)} (3-body), {len(i_t4)} (4-body)")
    print(f"   Reduced terms: {len(i_t1r)} (1-body), {len(i_t2r)} (2-body), {len(i_t3r)} (3-body), {len(i_t4r)} (4-body)")

    i_flat = (
        [list(map(float, t)) for t in i_t1r],
        [list(map(float, t)) for t in i_t2r],
        [list(map(float, t)) for t in i_t3r],
        [list(map(float, t)) for t in i_t4r]
    )

    res_image = cudaq.sample(
        dcqo_flexible_circuit_v2, num_qubits_circuit, steps,
        i_flat[0], i_flat[1], i_flat[2], i_flat[3],
        lambda_sched.tolist(), lambda_dot_sched.tolist(), dt,
        shots_count=shots
    )
    full_counts_image = prepend_fixed_prefix_to_counts(res_image)
    avg_energy_image, min_energy_image = calculate_energy(full_counts_image, i_t1, i_t2, i_t3, i_t4)
    print(f"   Result: <E> = {avg_energy_image:.4f}")
    print(f"   Top bitstring: {FIXED_FIRST_TWO_PREFIX}{res_image.most_probable()}\n")

    # ---------------- Comparison Summary ----------------
    print("--- Summary ---")
    print(f"{avg_energy_default} vs {avg_energy_image}")
    print(f"{min_energy_default} vs {min_energy_image}")

    top_default = res_default.most_probable()
    top_image = FIXED_FIRST_TWO_PREFIX + res_image.most_probable()
    if top_default == top_image:
        print("Convergence: Both Hamiltonians found the SAME most probable bitstring.")
    else:
        print("Convergence: The Hamiltonians favored DIFFERENT bitstrings.")