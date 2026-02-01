# Agent 06 — Artifact Sync Agent (Remote → Local)

## Purpose
When compute runs on qBraid/BerQ and downstream agents run elsewhere, we need deterministic artifact transfer. This agent ensures `runs/<run_id>/` artifacts are available where needed.

## Runs Where
- n8n Cloud (as coordinator), but actual copy happens via:
  - SSH/SCP
  - HTTP download
  - shared object storage (S3/GDrive)

## Responsibilities
- Fetch remote artifacts:
  - seeds.npy
  - seed_metrics.json
  - logs.txt
  - (later) mts_result.json and verify.json if those are remote too
- Place them in the expected `runs/<run_id>/` folder on the MTS/verification machine.

## Inputs
- run_id
- dispatch routing (`qbraid` / `berq` / `local`)
- remote job identifier (if applicable)

## Outputs
- local availability of:
  - `runs/<run_id>/seeds.npy`
  - `runs/<run_id>/seed_metrics.json`
  - `runs/<run_id>/logs.txt`

## Owner
- Workflow Lead / Infra Lead
