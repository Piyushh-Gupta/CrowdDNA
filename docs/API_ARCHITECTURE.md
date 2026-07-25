# CrowdDNA API Architecture (Phase 22)

## Overview
The API & Service Layer provides a robust, versioned, and secure entry point into the CrowdDNA ML platform. It completely shields internal models, workflows, and orchestrators from the web transport layer, employing strict encapsulation, registry-driven dependency resolution, and deterministic execution templates.

## Architecture Principles

### 1. Strict Layering
- **Transport Layer**: Handles HTTP requests/responses, routing, validation, and authentication extraction. Contains no business logic.
- **Service Layer**: Orchestrates cross-domain logic via a `BaseService` enforcing a strict pipeline: `authorize → validate → execute → observe → audit`.
- **Domain Layer**: The underlying CrowdDNA framework.

### 2. DTO Isolation & Universal Responses
- **No Domain Leakage**: Domain objects (e.g., `WorkflowManifest`) are never returned directly in HTTP responses.
- **Generic DTOs**: Standardized pagination and filtering capabilities via dedicated models.
- **Universal Envelope**: Every API response follows a strict schema:
  - `success`
  - `data`
  - `errors` (with typed API error codes)
  - `metadata`
  - `request_id`
  - `correlation_id`
  - `timestamp`

### 3. Execution Concurrency
Machine Learning operations are inherently slow.
- **Synchronous Execution**: Handled directly for read-only or fast operations.
- **Background Task Manager**: Sits between the Service Layer and Orchestration Layer to dispatch long-running jobs.
- **Job Management**: Dedicated subsystem for execution lookup, status checking, cancel, pause, resume, and historical tracking.
- **Streaming Package**: Dedicated SSE and WebSocket abstractions for real-time model telemetry or inference output.

### 4. Robust Metadata & Health
- **API Context**: Fully expanded to track authenticated identity, permissions, client IP, user agent, request start timestamp, and API version.
- **Metadata Versioning**: Detailed global metadata tracing `api_version`, `schema_version`, `build_id`, `git_commit`, `supported_versions`, and `crowddna_version`.
- **Tri-State Health Checks**: Explicit endpoints for `startup`, `liveness`, and `readiness` to support advanced Kubernetes deployments.

## Directory Structure
```
training/api/
├── application.py     # Framework initialization
├── routing.py         # Route registry
├── middleware.py      # Request ID, Logging, Timing
├── exceptions.py      # Global Error Mapper & Typed Codes
├── dto/               # Strict Request/Response Schemas & Envelopes
├── endpoints/         # HTTP Handlers (v1)
├── services/          # Business logic orchestrators & BaseService
├── streaming/         # SSE and WebSockets
├── tasks/             # BackgroundTaskManager
└── registry.py        # Endpoint, Service, and Middleware registries
```

## Integration Boundaries
- **Phase 20 Orchestration**: The Background Task Manager delegates long-running runs to the orchestrator.
- **Phase 21 Security**: Context population extracts identities. `BaseService` mandates authorization hooks prior to execution.
- **Phase 18 Observability**: Comprehensive audit trails, metrics, and error telemetry.
- **Phase 17 Reproducibility**: Execution and Job management tied directly to manifest serialization.
