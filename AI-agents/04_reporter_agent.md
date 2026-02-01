
# Agent 04 — Reporter / Historian Agent

## Purpose
Persist experiment outcomes and provide human-friendly summaries. This agent is designed to run in n8n Cloud.

## Runs Where
- n8n Cloud

## Responsibilities
- Read run outputs (either directly via API/SSH cat, or after artifact sync)
- Build a one-line JSON summary record
- Append to `history/runs.jsonl`
- Optionally send notifications (Slack/Discord/Email)

## Inputs
- `runs/<run_id>/seed_metrics.json`
- `runs/<run_id>/mts_result.json`
- `runs/<run_id>/verify.json`

## Outputs
- `history/runs.jsonl` appended with one JSON line, e.g.:
```json
{
  "run_id": "exp_0007",
  "pass": true,
  "best_E": 15320,
  "backend_used": "tensornet-mps",
  "seed_energy_best": 17890,
  "seed_energy_median": 25610,
  "time_seed_sec": 8.3,
  "time_mts_sec": 12.7
}
