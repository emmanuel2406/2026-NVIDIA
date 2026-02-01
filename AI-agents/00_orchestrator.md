# Agent 00 — Orchestrator Agent

## Purpose
The Orchestrator is the control-plane “manager agent” that selects the next experiment configuration and dispatch plan. It is designed to be lightweight and deterministic so it can run inside n8n Cloud.

## Runs Where
- n8n Cloud (Function node / config builder)
- Optional: local machine (for debugging only)

## Responsibilities
- Create a unique `run_id`
- Write `configs/<run_id>.json` (or emit JSON payload) with:
  - LABS instance size `N`
  - quantum seed parameters (Trotter steps, dt, schedule, shots)
  - symmetry exploitation configuration
  - MTS parameters (population, tabu iterations, tenure, etc.)
  - dispatch routing (where each agent runs: local / qBraid / BerQ)
  - git commit SHA (for reproducibility)
- Enforce reproducibility:
  - include RNG seeds for quantum and classical phases
  - include timestamps only for identification (not algorithm randomness)

## Inputs
- `history/runs.jsonl` (optional; if missing, treat as empty)
- optionally: “current best configs” / manual overrides

## Outputs
- `configs/<run_id>.json`
- stdout JSON (recommended):
  ```json
  {"run_id":"exp_0007","config_path":"configs/exp_0007.json"}
