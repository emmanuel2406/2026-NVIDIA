
# Agent 03 — Verification Agent (QA / Correctness Gate)

## Purpose
Guarantee correctness and reproducibility. If this agent fails, the run is considered invalid and must not be logged as a success.

## Runs Where
- Same environment as MTS (preferred, because artifacts are local there)
Triggered from n8n Cloud via SSH/HTTP.

## Responsibilities
- Validate the artifact contract:
  - config exists
  - seeds.npy exists and contains only ±1
  - mts_result.json exists and contains best_E
- Recompute true LABS energy from the reported best sequence and confirm it matches best_E
- Run unit tests (pytest) if available
- Validate symmetry constraints if they are claimed in config
- Write `verify.json` pass/fail with reasons

## Inputs
- `runs/<run_id>/` directory

## Outputs
- `runs/<run_id>/verify.json` (required)
- append logs to `runs/<run_id>/logs.txt`

Required `verify.json` schema:
```json
{
  "pass": true,
  "reasons": [],
  "run_id": "exp_0007",
  "best_E": 15320,
  "backend_used": "tensornet-mps",
  "timings": {"seed_sec": 8.3, "mts_sec": 12.7},
  "tests": {"pytest_pass": true},
  "symmetry_checks": {"pass": true, "violations": 0}
}
