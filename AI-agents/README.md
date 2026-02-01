# Agentic Workflow Overview

This project uses an agentic workflow orchestrated by n8n Cloud (control plane) and executed on compute environments (execution plane). Each agent is responsible for a strict artifact contract under `runs/<run_id>/`.

## Agents
- Agent 00: Orchestrator (config + dispatch)
- Agent 01: Quantum Seed (Trotterization + symmetry exploitation)
- Agent 02: Classical MTS (GPU-accelerated refinement)
- Agent 03: Verification (QA gate)
- Agent 04: Reporter (history/logging)
- Agent 05: Failure/Repair (human-in-the-loop)
- Agent 06: Artifact Sync (remote results transport)

## Artifact Contract
All successful runs must contain:
- seeds.npy
- seed_metrics.json
- mts_result.json
- verify.json
- logs.txt

All runs must have:
- runs/<run_id>/config.json
