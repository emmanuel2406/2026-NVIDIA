
# Agent 02 — Classical Optimization Agent (Memetic Tabu Search)

## Purpose
Refine quantum-generated seeds into high-quality LABS solutions using Memetic Tabu Search (MTS). This is the primary exploitation engine.

## Runs Where
- Local GPU machine (primary)
- Optional: BerQ / qBraid (if GPU or scheduler access exists)
Triggered from n8n Cloud via SSH/HTTP.

## Responsibilities
- Load `runs/<run_id>/seeds.npy` and `runs/<run_id>/config.json`
- Initialize MTS population (quantum seeds + optional random padding)
- Run tabu search using fast incremental ΔE updates
- Apply memetic operators (mutation/crossover/restarts) per configured policy
- Save best solution and metrics

## Inputs
- `runs/<run_id>/config.json`
- `runs/<run_id>/seeds.npy`

## Outputs
- `runs/<run_id>/mts_result.json` (required)
- optionally `runs/<run_id>/best_sequence.npy` or `.txt`
- append logs to `runs/<run_id>/logs.txt`

Example `mts_result.json`:
```json
{
  "best_E": 15320,
  "best_F": 0.006527,
  "iters": 30000,
  "time_mts_sec": 12.7,
  "source_seed_index": 3
}
