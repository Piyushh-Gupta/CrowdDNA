# Architecture Overview

CrowdDNA follows a decoupled, feature-based microservices and modular monolith design.

## System Topology

```mermaid
graph TD
  UI[Frontend UI] -->|REST/SSE| API[Phase 22 API]
  SDK[SDK Clients] -->|REST/SSE| API
  API --> Security[Security Module]
  API --> Workflow[Workflow Engine]
  Workflow --> Inference[ML Inference]
  Workflow --> Observability[Observability & Exports]
  Operations[Operations Framework] --> API
```

## Frontend/Backend Interaction
The frontend is a fully isolated Vite/React application that communicates exclusively via the API layer. No shared memory or database connections exist between the UI and backend.

## Release Pipeline
```mermaid
graph LR
  Build[Artifact Builder] --> Sign[Signing Manager]
  Sign --> SBOM[SBOM Generator]
  SBOM --> Publish[Publish to Registry]
```

## Key Architectural Principles
- **Separation of Concerns**: Deployment, Release, Frontend, and ML Inference are strictly isolated.
- **Stateless API**: The API tier maintains no session state.
- **Event-Driven**: Internal subsystems communicate via domain events.
