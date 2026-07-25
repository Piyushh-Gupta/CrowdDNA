# CrowdDNA Security Architecture (Phase 21)

## Overview
The Phase 21 Security & Policy Framework provides the canonical security layer for CrowdDNA. It ensures deterministic, auditable, and immutable security operations decoupled completely from AI model logic.

## Architecture Philosophy
- **Separation of Concerns**: Authentication is completely separated from Authorization.
- **Fail Fast**: Pre-execution checks abort runs before node scheduling.
- **Best-Effort Auditing**: Prevents network failures from killing ML workloads, while allowing strict mode overrides.
- **JIT Secrets**: Secrets are encapsulated in `SecretReference` and only unboxed via `SecretResolver` immediately prior to use.

## Core Abstractions
1. **Resource Abstraction**: Extends `Principal -> Action -> Resource` across the codebase.
2. **SecurityDecision**: The canonical evaluation trace returned by all authorization and policy engines.
3. **Session Caches**: `AuthenticationCache`, `AuthorizationCache`, `PolicyCache`.
4. **SecurityReport**: Reusable aggregate execution traces suitable for `WorkflowManifest` reproducibility.
5. **SecurityEvent**: Typed events integrated directly into Phase 18 Observability.

## Data Flow
1. **Initialize Engine** -> **Register Providers**
2. **Authenticate** -> Return `Identity` -> Create `SecuritySession`
3. **Authorize Request** -> Check `AuthorizationCache` -> Check RBAC -> Compile Policies -> Evaluate Policies -> Merge -> Output `SecurityDecision`
4. **Audit** -> Asynchronously append `AuditRecord`

## Dependencies
- Phase 17 Reproducibility (SecurityReport embedding)
- Phase 18 Observability (SecurityEvent dispatching)
- Phase 19 Plugins (Pre-load authorization)
- Phase 20 Orchestration (Node execution authorization)
