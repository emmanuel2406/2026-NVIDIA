# Tutorial Notebook Guide: Quantum Enhanced Optimization for LABS

Guide to **`tutorial_notebook/01_quantum_enhanced_optimization_LABS.ipynb`**, which implements the QE-MTS workflow from the paper [Scaling advantage with quantum-enhanced memetic tabu search for LABS](https://arxiv.org/html/2511.04553v1) (arXiv:2511.04553v1).

**Kernel:** `Python 3 [cuda q-v0.13.0]`  
**Prerequisites:** Basic quantum computing (operators, gates); CUDA-Q installed.

---

## 1. Overview and Learning Goals

The notebook teaches:

1. **LABS problem** and its link to radar/communications (pulse compression, autocorrelation, sidelobes).
2. **Classical MTS** — limitations and scaling \( O(1.34^N) \).
3. **Counterdiabatic (DCQO) circuit** in CUDA-Q to get low-energy bitstrings.
4. **Quantum-enhanced workflow** — use DCQO samples to **seed** MTS and compare with random seeding.

**CUDA-Q used:** `cudaq.sample()`, `@cudaq.kernel`, `ry()`, `rx()`, `rz()`, `x()`, `h()`, `x.ctrl()`.

---

## 2. Section-by-Section Map

### 2.1 Setup and LABS Definition

- **Imports:** `cudaq`, `numpy`, `floor`, `auxiliary_files.labs_utils as utils`.
- **LABS objective:**
  - \( E(s) = \sum_{k=1}^{N-1} C_k^2 \), \( C_k = \sum_{i=1}^{N-k} s_i s_{i+k} \).
  - Sequences \( s \in \{\pm 1\}^N \); goal: minimize \( E(s) \).
- **Application:** Radar pulse compression; low autocorrelation → sharp peak, fewer sidelobes.
- **Interactive:** [LABS visualization widget](https://nvidia.github.io/cuda-q-academic/interactive_widgets/labs_visualization.html) (e.g. N=7).

### 2.2 Classical Solution: Memetic Tabu Search (MTS)

- **Why LABS is hard:** Exponential search space; symmetries and degeneracies; rugged landscape (many local minima).
- **Exercise 1:** Find symmetries (e.g. **complement** \( s \leftrightarrow -s \), **reversal** \( (s_1,\ldots,s_N) \leftrightarrow (s_N,\ldots,s_1) \) — same energy).
- **MTS sketch:** Population → tournament → Combine or Mutate → TabuSearch → update best and population. Figure: `images/quantum_enhanced_optimization_LABS/mts_algorithm.png`.
- **Exercise 2:** Implement MTS (combine/mutate as in paper); output best bitstring and energy; add a simple visualization of final population energies.
- **Combine/Mutate figure:** `images/quantum_enhanced_optimization_LABS/combine_mutate.png`.

The notebook includes a full MTS implementation (config, init, combine, mutate, tabu search, evolution loop) and runs it (e.g. N=20, 50 individuals, 100 generations) with detailed logging.

### 2.3 Building the Quantum-Enhanced Workflow

- **Idea:** Use a **quantum** routine to produce bitstrings, then **seed MTS** with them instead of random.
- **Paper:** QAOA can reach \( O(1.21^N) \) with QMF for 28≤N≤40, but hardware limits ~N≤20; DCQO gives **shallower circuits** (~6× fewer entangling gates than 12-layer QAOA) and fits hybrid QE-MTS.
- **Adiabatic setup:** \( H_{\text{ad}}(\lambda) = (1-\lambda) H_i + \lambda H_f \), \( H_i = \sum_i h_i^x \sigma_i^x \), ground state of \( H_i \) = \( |+\rangle^{\otimes N} \).
- **Problem Hamiltonian \( H_f \):** 2-body and 4-body terms (formula in notebook; matches paper Eq. 2).
- **Counterdiabatic (CD) term:** Suppresses diabatic transitions; in **impulse regime** \( H_{\text{ad}} \) is dropped → evolution under \( H_{\text{CD}}^{(1)} \) only.
- **Circuit (Eq. 15 / paper B3):** Trotter product over steps; **2-qubit blocks** \( R_{YZ}, R_{ZY} \) (angle \( 4\theta h^x \)); **4-qubit blocks** \( R_{YZZZ}, R_{ZYZZ}, R_{ZZYZ}, R_{ZZZY} \) (angle \( 8\theta h^x \)). Gate count: e.g. N=67, QAOA ~1.4M entangling gates, DCQO ~236k.
- **Exercise 3:** Implement CUDA-Q **kernels** for:
  - 2-qubit block: \( R_{YZ}(\theta) \cdot R_{ZY}(\theta) \) (Fig. 3: 2× RZZ + 4 single-qubit).
  - 4-qubit block: \( R_{YZZZ}, R_{ZYZZ}, R_{ZZYZ}, R_{ZZZY} \) (Fig. 4: 10 RZZ + 28 single-qubit).
  - Hint: RZZ via CNOT–RZ–CNOT; adjoint of rotation = opposite angle.
- **Figure:** `images/quantum_enhanced_optimization_LABS/kernels.png`.

### 2.4 Theta Schedule and Interaction Sets

- **Simplification:** \( h_i^x = 1 \), \( h_b^x = 0 \) (bias terms for generalization only).
- **Angles:** Come from \( \lambda(t) \) (schedule) and \( \alpha(t) = -\Gamma_1(t)/\Gamma_2(t) \) (paper Eqs. 16–17). `labs_utils.compute_theta(t, dt, total_time, N, G2, G4)` implements this.
- **Exercise 4:** Build **G2** and **G4** (lists of index lists) from the loop structure in Eq. 15:
  - **G2:** two-body pairs \( [i, i+k] \) for \( i = 0..N-3 \), \( k = 1..\lfloor (N-i-1)/2 \rfloor \).
  - **G4:** four-body \( [i, i+t, i+k, i+k+t] \) for the triple loop (i, t, k) as in the formula.
- **Exercise 5:** Build the **full Trotterized kernel**: init \( |+\rangle^{\otimes N} \), then for each Trotter step apply G2 blocks with \( 4\theta \), G4 blocks with \( 8\theta \); thetas precomputed with `compute_theta` (cannot compute inside kernel). Run with `cudaq.sample(...)`, compute energies for sampled bitstrings, report min/mean/std.

### 2.5 Generating Quantum-Enhanced Results

- **Exercise 6:**
  1. **Quantum population:** Run DCQO circuit many shots (e.g. 200); compute energy for each bitstring; take top-K lowest-energy as initial MTS population (or replicate single best K times, as in paper).
  2. **Random population:** Same size, random bitstrings.
  3. **Run MTS** for both (e.g. N=15, 50 individuals, 20 generations).
  4. **Compare:** Best energy, merit, and (optionally) distribution of final population energies; plot autocorrelation \( C_k \) for best QE-MTS vs best random-MTS.

Notebook code does: sample 200 from circuit → take 50 best (or replicate best) → MTS with quantum seed; same MTS with random seed → side-by-side stats and plots.

### 2.6 Interpretation and Paper Results

- **Caveats:** Single run; quantum vs **random** only (other heuristics could do better); benefit of DCQO is **efficiency** (fewer gates) for when classical heuristics struggle and QPU time matters.
- **Paper figure:** `images/quantum_enhanced_optimization_LABS/tabu_search_results.png` — median TTS for N=27..37.
- **Takeaways:** MTS often faster at small N; QE-MTS has **better scaling** (~1.24^N vs ~1.34^N); crossover around **N ≳ 47** where QE-MTS is expected to win; hybrid “quantum data + classical solver” is the main message.

### 2.7 Self-Validation (Phase 1 Deliverable)

- **Purpose:** Prove baseline and evaluation are correct (symmetries, energies, merit factors).
- **Ground truth:** `evals/answers.csv` (from [Packebusch & Mertens](https://arxiv.org/pdf/1512.02475), N ≤ 66).
- **Tools:** `evals/eval_util.py` — `runlength_to_sequence`, `compute_energy`, `compute_merit_factor`, `parse_answers_csv`, `run_full_validation`, `validate_solution`, `normalized_energy_distance`, `get_expected_optimal_energy`.
- **Notebook cell:** Runs `run_full_validation(csv_path)`, spot-checks N=3,6,20 (sequence → E, F_N vs expected), shows normalized energy distance, and validation counts per N.
- **Conclusion:** All 90 solutions in the CSV pass; spot-checks match expected E and F_N. Run `python evals/eval_util.py` or `pytest evals/` for full tests.

---

## 3. Key Files and Paths

| Item | Path |
|------|------|
| Notebook | `tutorial_notebook/01_quantum_enhanced_optimization_LABS.ipynb` |
| LABS utils (theta, topology) | `tutorial_notebook/auxiliary_files/labs_utils.py` |
| Evals (energy, merit, CSV) | `tutorial_notebook/evals/eval_util.py` |
| Ground truth | `tutorial_notebook/evals/answers.csv` |
| Physics/symmetry tests | `tutorial_notebook/evals/physics_tests.py` |
| Images | `tutorial_notebook/images/quantum_enhanced_optimization_LABS/` (radar, mts_algorithm, combine_mutate, counteradiabatic, kernels, tabu_search_results) |

---

## 4. Exercise Checklist (for Phase 1)

- **Exercise 1:** Identify symmetries (complement, reversal) — solution in markdown.
- **Exercise 2:** Implement MTS + visualization — notebook has full implementation.
- **Exercise 3:** CUDA-Q kernels for 2- and 4-qubit blocks — notebook implements and draws circuits.
- **Exercise 4:** `get_interactions(N)` → G2, G4 — implemented and tested for N=5,6,20.
- **Exercise 5:** Full Trotterized DCQO kernel + sampling + energy stats — implemented with configurable N, T, n_steps.
- **Exercise 6:** Quantum vs random initial population + MTS comparison — implemented with plots.
- **Self-Validation:** Report using `eval_util` and `answers.csv` — cell prints summary and spot-checks.

---

## 5. Relation to Paper and Repo Code

- **Paper:** `docs/QE-MTS-PAPER-SUMMARY.md` (and `2511.04553v1.pdf`) — same E(s), \( H_f \), DCQO structure, impulse regime, G2/G4 blocks, QE-MTS algorithm (Algorithm 1), scaling 1.24^N vs 1.34^N, crossover N≳47.
- **impl-trotter:** `generate_trotterization.py` has `get_image_hamiltonian`, `dcqo_flexible_circuit_v2`, `get_interactions`-style indices; `qe_mts_image_hamiltonian.py` runs full QE-MTS with GPU MTS and writes results/plots.
- **impl-mts:** `main.py` provides CPU MTS and energy/merit/bitstring helpers used by the notebook logic (and by impl-trotter when importing from impl-mts).

Use this guide to navigate the notebook, map exercises to deliverables, and cross-reference the paper and repo implementations.
