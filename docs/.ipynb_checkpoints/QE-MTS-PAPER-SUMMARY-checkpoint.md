# QE-MTS Paper Summary (arXiv:2511.04553v1)

Structured summary of **"Scaling advantage with quantum-enhanced memetic tabu search for LABS"** (Gomez Cadavid et al., Nov 2025). Use this alongside the PDF `2511.04553v1.pdf` and the tutorial notebook.

---

## 1. Main Result

- **QE-MTS** (quantum-enhanced memetic tabu search) seeds classical MTS with bitstrings from **digitized counterdiabatic quantum optimization (DCQO)**.
- For sequence length **N ∈ [27, 37]**, empirical time-to-solution (TTS) scaling is **O(1.24^N)** for QE-MTS vs **O(1.34^N)** for best classical MTS and **O(1.46^N)** for QAOA.
- DCQO uses **~6× fewer entangling gates** than 12-layer QAOA (e.g. N=67: 236k vs 1.4M gates).
- **Crossover:** QE-MTS is expected to give lower typical TTS than MTS for **N ≳ 47** (median N× ≈ 46.6, 95% CI [44.9, 48.9]).

---

## 2. LABS Problem

### Objective

- **Sequence:** \( s = (s_1, \ldots, s_N) \in \{\pm 1\}^N \).
- **Energy (minimize):**
  \[
  E(s) = \sum_{k=1}^{N-1} C_k^2, \qquad C_k = \sum_{i=1}^{N-k} s_i s_{i+k}.
  \]
- **Merit factor (maximize):** \( F(s) = N^2 / (2 E(s)) \).

### Hamiltonian (spin encoding)

Mapping \( s_i \mapsto \sigma_i^z \) gives long-range Ising \( H_f \) with 2- and 4-body terms:

\[
H_f = 2 \sum_{i=1}^{N-2} \sigma_i^z \sum_{k=1}^{\lfloor (N-i)/2 \rfloor} \sigma_{i+k}^z
+ 4 \sum_{i=1}^{N-3} \sigma_i^z \sum_{t=1}^{\lfloor (N-i-1)/2 \rfloor} \sum_{k=t+1}^{N-i-t} \sigma_{i+t}^z \sigma_{i+k}^z \sigma_{i+k+t}^z.
\]

- **Term counts:** \( n_{\text{two}}(N) \) grows quadratically, \( n_{\text{four}}(N) \) cubically (see paper Eqs. 3–4).
- **Spectrum:** Energies in steps of 4; upper bound \( E \leq N(N-1)(2N-1)/6 = O(N^3) \); at most \( O(N^3) \) distinct levels vs \( 2^{N-2} \) configurations.

### Symmetries

- **Bit flip:** \( (s_1,\ldots,s_N) \equiv (-s_1,\ldots,-s_N) \).
- **Reversal:** \( (s_1,\ldots,s_N) \equiv (s_N,\ldots,s_1) \) (and combined with flip).
- Effective search space: **2^{N-2}** distinct configurations. Flip can be enforced at the Hamiltonian by fixing one spin.

### Ruggedness

- **Single-flip local minima density** \( f_{\text{LO}}(N) \): fraction of configurations that are 1-flip stable. LABS has higher \( f_{\text{LO}} \) than random Sherrington–Kirkpatrick instances → many local traps for local search → motivation for quantum exploration + classical refinement.

---

## 3. Digitized Counterdiabatic Quantum Optimization (DCQO)

### Adiabatic setup

- **Path:** \( H_{\text{ad}}(\lambda) = (1-\lambda) H_i + \lambda H_f \), \( \lambda(t) \in [0,1] \), \( t \in [0,T] \).
- **Initial Hamiltonian:** \( H_i = \sum_i h_i^x \sigma_i^x \) with \( h_j^x = -1 \) → ground state \( |\psi(0)\rangle = |+\rangle^{\otimes N} \).

### Counterdiabatic (CD) term

- **Total:** \( H(\lambda) = H_{\text{ad}}(\lambda) + \dot\lambda A_\lambda^{(l)} \), with \( A_\lambda^{(l)} \) from nested-commutator expansion (Eq. 6). Paper uses **first-order** (\( l=1 \)), giving \(-i O_1\) (Eq. 7 in paper): 2-body \( \sigma^y \sigma^z \) and 4-body \( \sigma^y \sigma^z \sigma^z \sigma^z \) (and permutations) with coefficients \( h_p^x \).

### Impulse regime

- For **fast evolution**, \( H_{\text{ad}} \) is neglected vs \( H_{\text{CD}} \) → evolution under **\( H(\lambda) \approx \dot\lambda A_\lambda^{(1)} \)** only. This reduces resources while keeping performance.

### Digitization and Trotterization

- **Time evolution:** \( U(T,0) = \prod_{k=1}^{n_{\text{trot}}} \exp\bigl[\Delta t\, \alpha_1(k\Delta t)\, \dot\lambda(k\Delta t)\, O_1(k\Delta t)\bigr] \), \( \Delta t = T/n_{\text{trot}} \).
- **Angle:** \( \theta(t) = \Delta t\, \alpha(t)\, \dot\lambda(t) \); \( \alpha_1(\lambda) = -\Gamma_1(\lambda)/\Gamma_2(\lambda) \) (Appendix B for \( \Gamma_1, \Gamma_2 \)).

### Circuit structure (Appendix B, Eq. B3)

- **Two-qubit blocks:** \( R_{Y_i Z_{i+k}}(4\theta h_i^x) \), \( R_{Z_i Y_{i+k}}(4\theta h_{i+k}^x) \) — each block: **2 RZZ + 4 single-qubit** (Fig. 3).
- **Four-qubit blocks:** \( R_{YZZZ}, R_{ZYZZ}, R_{ZZYZ}, R_{ZZZY} \) (each with \( 8\theta h_p^x \)) — each block: **10 RZZ + 28 single-qubit** (Fig. 4).
- **Gate count:** One Trotter step of DCQO ≈ two QAOA layers in entangling cost. Example: N=67, QAOA(p=12) ~1.4M entangling gates, DCQO ~236k.

### Measurement

- After each run, measure all qubits \( n_{\text{shots}} \) times → bitstrings \( b_0\ldots b_{N-1} \in \{0,1\} \) → each sample gives an energy value of \( H_f \).

---

## 4. Quantum-Enhanced Memetic Tabu Search (QE-MTS)

### Algorithm 1 (high level)

1. **Initialize:** Use **DCQO** to obtain a set of bitstrings; take the **lowest-energy** bitstring and **replicate it K times** as the initial MTS population. (Baseline MTS: K random bitstrings.)
2. **Loop** until \( E(s^\star) \leq E_{\text{target}} \) or \( G \geq G_{\max} \):
   - With probability \( p_{\text{comb}} \): **Combine** two parents (tournament) → child **c**.
   - Else: pick random individual, **Mutate(c, p_mut)**.
   - **TabuSearch(c)** → locally improved child.
   - If \( E(c) < E(s^\star) \), set \( s^\star \leftarrow c \).
   - Replace a random population member by **c**; \( G \leftarrow G+1 \).
3. **Return** \( s^\star \).

### Parameters (paper)

- \( K = 100 \), \( p_{\text{comb}} = 0.9 \), \( p_{\text{mut}} = 1/N \), tournament size 2.
- Single-thread MTS; stop at first hit of certified optimum.
- **TTS** = number of objective evaluations until optimum first found (paper reports TTS without wall-clock; real time ≈ TTS × τ with τ = time per evaluation).

### Tabu search (Algorithm 2)

- One-flip neighborhood; short-term **tabu list** with randomized tenure \( \theta \in [M/10, M/50] \), \( M \in [N/2, 3N/2] \).
- **Aspiration:** tabu overridden if move yields new global best.
- Iterate for **M** steps; return best \( \tilde s \) found.

### Combine and Mutate (Algorithm 3)

- **Combine(p1, p2):** single-point crossover — choose \( k \in \{1,\ldots,N-1\} \), return \( p_1[1:k] \| p_2[k+1:N] \).
- **Mutate(s, p_mut):** for each index \( i \), flip \( s_i \) with probability \( p_{\text{mut}} \).

---

## 5. Scaling and Crossover (Paper Results)

### Setup

- **N ∈ [27, 37]**; methods: **MTS**, **QE-MTS**; 100 replicates × 100 seeds per (N, method).
- **Per-replicate summary:** \( \tilde Y_{N,m,r} = \text{median}_s Y_{N,m,r,s} \).
- **Quantiles:** \( Q_p(N,m) \) over replicates; fit \( \log Q_p(N,m) = \alpha_{m,p} + \beta_{m,p} N \) → scaling base \( \kappa_{m,p} = e^{\beta_{m,p}} \) (so \( Q_p \sim \kappa^N \)).

### Two-stage bootstrap

- Resample replicates and seeds; refit; report 95% CI for \( \kappa \) and \( R^2 \).

### Reported scaling (Table I)

| Method   | Summary | κ (95% CI)   | R² (95% CI) |
|----------|---------|--------------|-------------|
| QE-MTS   | Q0.50   | [1.23, 1.25] | [0.86, 0.89] |
| QE-MTS   | Q0.10   | [1.25, 1.27] | [0.83, 0.91] |
| QE-MTS   | Q0.90   | [1.24, 1.25] | [0.90, 0.92] |
| MTS      | Q0.50   | [1.36, 1.37] | [0.85, 0.87] |
| MTS      | Q0.10   | [1.34, 1.36] | [0.85, 0.88] |
| MTS      | Q0.90   | [1.37, 1.39] | [0.85, 0.87] |

- **Typical (Q0.50):** QE-MTS ~1.24^N, MTS ~1.37^N → **shallower scaling** for QE-MTS.
- At small N, MTS often has **lower** TTS (smaller intercept); as N grows, QE-MTS slope wins and crossover is projected at **N× ≈ 47**.

### Crossover definition (Eq. 9)

- \( N_\times^{(b)} = \frac{ \alpha_{\text{MTS},0.05}^{(b)} - \alpha_{\text{QE-MTS},0.95}^{(b)} }{ \beta_{\text{QE-MTS},0.95}^{(b)} - \beta_{\text{MTS},0.05}^{(b)} } \) (upper QE-MTS vs lower MTS).
- **Median N× = 46.6**, 95% CI [44.9, 48.9] → **QE-MTS safer choice for N ≳ 47** (excluding QPU sampling cost; including it shifts crossover but scaling gain remains).

---

## 6. Appendices (Quick Reference)

- **Appendix A:** Spectrum size \( O(N^3) \); minimum gap between energies \( \Delta E = 4 \).
- **Appendix B:** DCQO circuit construction; \( \Gamma_1, \Gamma_2 \) formulas (Eqs. B4–B5); 2- and 4-qubit gate decompositions (Figs. 3–4).
- **Appendix C:** Log-ratio \( \log_{10}(\text{TTS}_{\text{QE-MTS}}) - \log_{10}(\text{TTS}_{\text{MTS}}) \); distribution shifts negative as N increases.
- **Appendix D:** Alternative QE-MTS using **multiple DCQO runs** to build initial population (lowest-energy bitstrings across runs); can reduce TTS and improve single-run comparison vs MTS.
- **Appendix E:** GPU specs (A100, H200, B200); CUDA-Q simulation; e.g. P6-b200.48xlarge for N up to 37, \( n_{\text{shots}} = 10^5 \).
- **Appendix F:** Tabu search (Algorithm 2) and Combine/Mutate (Algorithm 3) as in main text.

---

## 7. Implementation Notes (This Repo)

- **Classical MTS:** `impl-mts/main.py` (CPU), `impl-mts/mts_h100_optimized.py` (GPU).
- **DCQO / image Hamiltonian:** `impl-trotter/generate_trotterization.py` (G2, G4, theta, 2/4-qubit blocks); `impl-trotter/qe_mts_image_hamiltonian.py` (full QE-MTS pipeline).
- **Tutorial (CUDA-Q):** `tutorial_notebook/01_quantum_enhanced_optimization_LABS.ipynb` — LABS, MTS, DCQO kernels, Trotter circuit, quantum vs random seeding, self-validation.
- **Evaluation:** `tutorial_notebook/evals/eval_util.py`, `answers.csv` (ground truth E, F_N, run-length); `evals/physics_tests.py` (symmetries).

---

## 8. References (from paper)

- LABS / optimal sequences: Packebusch & Mertens, J. Phys. A **49**, 165001 (2016); [arXiv:1512.02475](https://arxiv.org/pdf/1512.02475).
- MTS for LABS: Gallardo, Cotta, Fernández-Leiva, Appl. Soft Comput. **9**, 1252 (2009).
- QAOA + QMF for LABS: Shaydulin et al., Sci. Adv. **10**, eadm6761 (2024).
- DCQO / digitized CD: Hegade et al., Phys. Rev. Appl. **15**, 024038 (2021); Chandarana et al., Phys. Rev. Appl. **22**, 054037 (2024).
- BF-DCQO: Cadavid et al., Phys. Rev. Res. **7**, L022010 (2025).
- CUDA-Q: [NVIDIA CUDA-Q](https://nvidia.github.io/cuda-quantum/).
