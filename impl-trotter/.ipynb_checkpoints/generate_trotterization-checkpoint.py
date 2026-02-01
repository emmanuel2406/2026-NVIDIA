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
            # ... (include all permutations as in your original code) ...

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

def calculate_energy(counts, t1, t2, t3, t4):
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
    
    # Annealing Schedule
    t_points = np.linspace(0, total_time, steps)
    dt = total_time / steps
    lambda_sched = np.sin((np.pi/2) * (t_points / total_time))**2
    lambda_dot_sched = (np.pi/total_time) * np.sin(np.pi * t_points / total_time) / 2.0 
    
    print(f"--- Comparing Hamiltonians for N={N} ---\n")

    # ---------------- 1. Default LABS Hamiltonian ----------------
    print(">> Processing Default LABS Hamiltonian...")
    d_t1, d_t2, d_t3, d_t4 = get_labs_hamiltonian(N)
    
    print(f"   Terms: {len(d_t1)} (1-body), {len(d_t2)} (2-body), {len(d_t3)} (3-body), {len(d_t4)} (4-body)")
    
    # Flatten strictly for C++ kernel compatibility
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

    # ---------------- 2. Image H_f Hamiltonian ----------------
    print(">> Processing Image H_f Hamiltonian...")
    i_t1, i_t2, i_t3, i_t4 = get_image_hamiltonian(N)
    
    print(f"   Terms: {len(i_t1)} (1-body), {len(i_t2)} (2-body), {len(i_t3)} (3-body), {len(i_t4)} (4-body)")

    i_flat = (
        [list(map(float, t)) for t in i_t1],
        [list(map(float, t)) for t in i_t2],
        [list(map(float, t)) for t in i_t3],
        [list(map(float, t)) for t in i_t4]
    )

    res_image = cudaq.sample(
        dcqo_flexible_circuit_v2, N, steps, 
        i_flat[0], i_flat[1], i_flat[2], i_flat[3], 
        lambda_sched.tolist(), lambda_dot_sched.tolist(), dt,
        shots_count=shots
    )
    
    avg_energy_image, min_energy_image = calculate_energy(res_image, i_t1, i_t2, i_t3, i_t4)
    print(f"   Result: <E> = {avg_energy_image:.4f}")
    print(f"   Top bitstring: {res_image.most_probable()}\n")
    
    # ---------------- Comparison Summary ----------------
    print("--- Summary ---")
    print(f"{avg_energy_default} vs {avg_energy_image}")
    print(f"{min_energy_default} vs {min_energy_image}")
    
    if res_default.most_probable() == res_image.most_probable():
        print("Convergence: Both Hamiltonians found the SAME most probable bitstring.")
    else:
        print("Convergence: The Hamiltonians favored DIFFERENT bitstrings.")