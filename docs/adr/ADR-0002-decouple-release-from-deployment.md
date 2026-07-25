# ADR 0002: Decouple Release from Deployment

## Status
Accepted

## Context
Previously, artifact generation (building wheels, docker images) was coupled directly to deployment scripts, meaning artifacts were rebuilt per environment, risking inconsistency.

## Decision
Release Engineering (Phase 28) is now completely isolated from Deployment (Phase 24). Artifacts are built, signed, and packaged exactly once.

## Consequences
- Guarantees byte-for-byte identical artifacts in Staging and Production.
- Requires CI pipelines to explicitly separate "Build" steps from "Deploy" steps.
