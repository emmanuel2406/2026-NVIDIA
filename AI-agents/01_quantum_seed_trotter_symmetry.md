



# Agent 01 — Quantum Seed Agent (Trotterization + Symmetry Exploitation)

## Purpose
Generate high-quality LABS seed sequences by simulating digitized (Trotterized) quantum evolution under an effective LABS-inspired Hamiltonian, while exploiting symmetries to reduce the search space and improve sampling efficiency.

This agent replaces QAOA: no variational angles; instead it uses a time-evolution style circuit (Trotter steps).

## Runs Where
- qBraid (primary)
- BerQ (alternative)
- Local GPU machine (fallback / development)
Triggered from n8n Cloud via SSH or HTTP job submission.

## Responsibilities
- Load config (`runs/<run_id>/config.json`)
- Build the effective diagonal cost Hamiltonian `H_eff` (lag-truncated / blockwise)
- Build mixer Hamiltonian `H_mix` (typically sum X_i or symmetry-preserving variant)
- Apply Trotterized evolution:
  - `steps = m`, `dt = Δt`, schedule may vary mixing/cost weights over time
- Enforce symmetry exploitation:
  - Param reduction (generate only independent variables, then expand)
  - Canonicalization under group actions (reflection/rotation)
  - Optional post-selection or projection (if implemented)
- Sample bitstrings, convert to spins (±1)
- Score seeds with true LABS energy (CPU scoring is OK; keep it consistent)
- Write seeds and metrics to disk

## Inputs
- `runs/<run_id>/config.json`
- `skills.md` (grounding for CUDA-Q API correctness)

Key config block:
```json
"quantum_seed": {
  "method": "trotter_symmetry_seed",
  "backend_preference": ["tensornet-mps", "tensornet", "qpp-cpu"],
  "shots": 2048,
  "trotter": { "steps": 40, "dt": 0.12, "schedule": "linear" },
  "symmetry": { "type": "reflection", "mode": "param_reduction", "canonicalize": true },
  "rng_seed": 7
}
