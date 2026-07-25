# Project Governance

## Maintainers
CrowdDNA is governed by a core group of maintainers responsible for architecture direction, release management, and security patches.

## Ownership
Strict code ownership is enforced via [../.github/CODEOWNERS](../.github/CODEOWNERS).

## Decision Process
All major architectural changes must be proposed via an Architecture Decision Record (ADR) and approved by a consensus of maintainers.

## Review Policy
- Bugfixes/Chores: 1 Maintainer Approval.
- Architecture/Features: 2 Maintainer Approvals.

## Conflict Resolution
In the event of a deadlock, the Lead Maintainer possesses a tie-breaking vote.

## Deprecation Policy
Features slated for removal must be marked as `Deprecated` for exactly one major version cycle before removal.

## Release Authority
Releases are exclusively orchestrated by designated Release Managers using the automated pipeline defined in Phase 28.
