
# Agent 05 — Failure / Repair Agent (Human-in-the-loop)

## Purpose
Handle failures as first-class events. When Verification fails, this agent creates a loop that routes logs to the QA Lead and enables rapid patching via Claude Code / Cursor / VSCode.

## Runs Where
- n8n Cloud (notifications, ticket creation)
- Human dev environment (Claude Code sessions + git worktrees)

## Responsibilities
- On any failure:
  - collect `runs/<run_id>/logs.txt`
  - collect `verify.json` reasons
  - notify QA Lead with actionable context
  - optionally open a GitHub Issue with logs + repro steps
- Track which commit/branch/worktree the failure relates to
- Trigger re-run once patch lands

## Inputs
- `runs/<run_id>/verify.json`
- `runs/<run_id>/logs.txt`
- `git_commit` from config

## Outputs
- A visible failure report (notification / issue)
- Optional patch artifact (diff) if automated

## Owner
- QA Lead + Technical Lead
