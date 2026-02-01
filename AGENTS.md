# AGENTS.md

Guidance for AI agents (Cursor, Claude Code, etc.) working in this repository. For human-oriented project context, see [CLAUDE.md](CLAUDE.md) and [README.md](README.md).

---

## 1. Project and Problem

- **Challenge:** NVIDIA iQuHACK 2026 — hybrid quantum-enhanced solver for **Low Autocorrelation Binary Sequences (LABS)**.
- **Goal:** Evolve classical Memetic Tabu Search (MTS) by integrating quantum algorithms (QAOA, Grover/QMF, Trotterization) and GPU acceleration.
- **LABS:**
  - Minimize sidelobe energy: `E(s) = Σ_{k=1}^{N-1} C_k²`, with `C_k = Σ_i s_i s_{i+k}`.
  - Merit factor (higher is better): `F(s) = N² / (2·E(s))`.
  - Sequences are **±1** (spin); conversion: binary `'0'`→+1, `'1'`→-1 via `bitstring_to_sequence` / `sequence_to_bitstring` in `impl-mts/main.py`.

---

## 2. Repository Layout (Where to Work)

| Path | Purpose |
|------|--------|
| **impl-mts/** | Classical MTS baseline: `main.py` (CPU), `mts_h100_optimized.py` (CuPy/GPU). Core energy/merit and `memetic_tabu_search()`. |
| **impl-qaoa/** | Digitized Counterdiabatic QAOA (DC-QAOA); CUDA-Q. |
| **impl-qmf/** | Quantum Minimum Finding (QAOA + Grover + MTS); hybrid pipeline. |
| **impl-trotter/** | Image Hamiltonian (1/2/3/4-body), Trotter circuits; `qe_mts_image_hamiltonian.py` = main QE-MTS entry; `generate_trotterization.py` = circuits. |
| **GPU_Optimised/** | GPU toolkit: CuPy batch energy, custom CUDA kernels, H100-focused scripts. |
| **tutorial_notebook/** | Phase 1 tutorial: `01_quantum_enhanced_optimization_LABS.ipynb`; kernel `Python 3 [cuda q-v0.13.0]`. |
| **tutorial_notebook/evals/** | `eval_util.py` (energy, merit, run-length, ground truth), `answers.csv`, `physics_tests.py`. |
| **impl-trotter/evals/** | Copy of evals for impl-trotter (eval_util, answers.csv, physics_tests). |
| **benchmarks/** | `run_benchmark.py` (N or N-range), `results.csv`. |
| **team-submissions/** | Deliverables: PRD, AI_REPORT, TEST_SUITE, presentation templates; see [team-submissions/README.md](team-submissions/README.md). |
| **research/** | Notes (e.g. approaches.md, qmf.md). |
| **skills.md** | CUDA-Q Python API quick reference (use to avoid hallucinated APIs). |

---

## 3. Commands to Run

**Tests**

```bash
# Eval utilities
python tutorial_notebook/evals/eval_util.py test
cd tutorial_notebook/evals && python -m pytest test_eval_util.py

# Trotter / impl-trotter (script in test.py runs QAOA/trotter demo)
python impl-trotter/test.py
```

**Benchmarks**

```bash
python benchmarks/run_benchmark.py 3 4 5 10 20
python benchmarks/run_benchmark.py 3-25
```

**Implementations**

```bash
# Tutorial (Phase 1 – qBraid)
jupyter notebook tutorial_notebook/01_quantum_enhanced_optimization_LABS.ipynb

# GPU MTS (Phase 2 – Brev)
python GPU_Optimised/run_mts_gpu.py --N 50 --population_size 100 --generations 1000

# QE-MTS (image Hamiltonian + GPU MTS)
python impl-trotter/qe_mts_image_hamiltonian.py 25 100 1000 500 10
# Args: N, population_size, max_generations, shots, trotter_steps
```

---

## 4. Conventions and APIs

- **Energy / merit:** Use `compute_energy(s)` and `compute_merit_factor(s)` (or `energy_and_merit(s)`). Sequences `s` are **NumPy ±1 arrays** in impl-mts/GPU_Optimised; eval_util also uses list-based `compute_energy`/`compute_merit_factor` with the same formulas.
- **Ground truth:** `tutorial_notebook/evals/answers.csv` and `impl-trotter/evals/answers.csv`; use `get_expected_optimal_energy(N)` and run-length helpers in `eval_util.py` for validation.
- **Run-length:** See `eval_util.runlength_to_sequence` / `sequence_to_runlength`; first run is +1, then alternating.
- **Physics checks:** Symmetry (e.g. complement, reversal) in `evals/physics_tests.py`; reuse or mirror these when adding features.
- **CUDA-Q:** Prefer APIs listed in `skills.md`; set backend with `cudaq.set_target(...)` (e.g. `"qpp"` for CPU, GPU targets on Brev).

---

## 5. Key Entry Points for Code Changes

- **Classical MTS and energy:** `impl-mts/main.py` — `memetic_tabu_search`, `compute_energy`, `compute_merit_factor`, `bitstring_to_sequence`, `sequence_to_bitstring`, `compute_delta_energy`.
- **GPU MTS:** `impl-mts/mts_h100_optimized.py`; `GPU_Optimised/run_mts_gpu.py`, `run_mts_h100.py`.
- **Quantum + MTS pipeline:** `impl-trotter/qe_mts_image_hamiltonian.py` (imports from `generate_trotterization.py` and impl-mts).
- **Circuit / Hamiltonian:** `impl-trotter/generate_trotterization.py` — image Hamiltonian, `dcqo_flexible_circuit_v2`, etc.
- **Validation and metrics:** `tutorial_notebook/evals/eval_util.py` — energy, merit, run-length, optimal lookup; `benchmarks/run_benchmark.py` uses this for timing and comparison.

---

## 6. Dependencies and Environment

- **CUDA-Q v0.13.0** — quantum simulation/compilation.
- **CuPy** (cupy-cuda12x) — GPU arrays in impl-mts GPU and GPU_Optimised.
- **NumPy** — CPU arrays.
- **Platforms:** Phase 1 = qBraid (CPU); Phase 2 = Brev (L4/A100/H100).

---

## 7. Agent-Oriented Guidelines

1. **Before changing energy/merit logic:** Ensure compatibility with `eval_util.py` and `answers.csv`; run `eval_util.py test` and physics_tests after edits.
2. **When adding or modifying quantum paths:** Keep `skills.md` in context to use correct CUDA-Q APIs; prefer existing patterns in `impl-trotter` and `impl-qaoa`.
3. **When touching MTS:** Preserve function signatures used by `qe_mts_image_hamiltonian.py` and `run_benchmark.py` (e.g. sequence as ±1, bitstring conversion).
4. **Deliverables:** New code/tests that belong in the submission should be documented or placed per [team-submissions/README.md](team-submissions/README.md) (e.g. TEST_SUITE, AI_REPORT).
5. **Paths:** Prefer adding `sys.path` or running from repo root so that `impl-mts`, `tutorial_notebook/evals`, and `impl-trotter/evals` resolve as in existing scripts.

---

## 8. References

- [CLAUDE.md](CLAUDE.md) — Commands, architecture, key functions.
- [README.md](README.md) — Challenge overview, phases, submission.
- [LABS-challenge-Phase1.md](LABS-challenge-Phase1.md), [LABS-challenge-Phase2.md](LABS-challenge-Phase2.md) — Milestones and requirements.
- [team-submissions/README.md](team-submissions/README.md) — Deliverables checklist and grading.
- [team-submissions/PRD.md](team-submissions/PRD.md) — Product requirements and verification plan.
- [skills.md](skills.md) — CUDA-Q API reference.
- **docs/** — Paper and notebook documentation:
  - [docs/QE-MTS-PAPER-SUMMARY.md](docs/QE-MTS-PAPER-SUMMARY.md) — Summary of 2511.04553v1 (LABS, DCQO, QE-MTS, scaling).
  - [docs/TUTORIAL-NOTEBOOK-GUIDE.md](docs/TUTORIAL-NOTEBOOK-GUIDE.md) — Guide to `tutorial_notebook/01_quantum_enhanced_optimization_LABS.ipynb`.
