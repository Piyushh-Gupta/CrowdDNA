# Implementation Kickoff Guide

## Project Overview
Transitioning from architectural scaffolding to runtime execution for CrowdDNA v1.0.

## Roles & Ownership
- **Piyush (Technical Lead)**: Backend, Data Engineering, ML Pipelines, API, Ops, Security.
- **Aayushi (Core Contributor)**: Frontend UI, Python SDK, Documentation, Developer Experience, E2E Testing.

## Communication & Daily Workflow
- **Async First**: Use Slack/Teams for non-blocking queries.
- **Daily Check-in**: 15-minute sync focusing purely on blockages and API contract negotiations.
- **GitHub Issues**: The sole source of truth for task state.

## Review Workflow
- **Review Strategy**: PRs must be reviewed by the assigned Reviewer.
- **Expected Review Times**: < 24 hours.

## Sprint Cadence
- 2-week iterations.
- Friday merge unblocks and sprint demos.

## Merge Strategy
- PRs merge into `develop`. 
- Squash and merge only.
- Strict adherence to Conventional Commits.

## Branch Naming
- Features: `feat/[epic-name]-[issue-desc]`
- Bugs: `fix/[epic-name]-[issue-desc]`

## Escalation Process
- If blocked for > 4 hours, escalate immediately in the async channel.
