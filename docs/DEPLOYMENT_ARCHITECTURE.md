# Deployment, Infrastructure & DevOps Architecture (Phase 24)

## Philosophy
The CrowdDNA Deployment Framework adheres strictly to immutable, containerized, and declarative infrastructure principles. By utilizing Docker as the canonical runtime and strictly separating continuous integration (CI) from continuous deployment (CD), we guarantee deterministic execution across all environments while maintaining an uncompromised security posture.

---

## Core Architectural Pillars

### 1. Immutable & Deterministic Releases
Every commit triggers the creation of an immutable Docker artifact.
- **Deployment Metadata**: Releases strictly embed a `ReleaseManifest` capturing the exact Git SHA, Container SHA256 digest, environment hashes, and timestamp.
- **Version Compatibility**: Manifests define explicit compatibility matrices ensuring the deployed Frontend version is validated against the Backend API version prior to boot.
- Artifacts are never modified post-build; all environmental drift is controlled exclusively via orchestrator-injected ConfigMaps/Secrets.

### 2. Container Architecture & Security Hardening
Docker acts as the universal runtime abstraction.
- **Multi-Stage Builds**: Explicit `builder` stages compile C-extensions and resolve dependencies, discarding build-chains (like GCC/Node) in the final `runtime` image.
- **Least Privilege**: Containers execute exclusively as a non-root user.
- **Immutable Filesystems**: Production containers enforce `read_only: true` root filesystems, utilizing explicit, constrained `tmpfs` mounts strictly for caching and temporary operational data.
- **Security Opts**: Containers enforce `no-new-privileges:true`.

### 3. Granular Deployment Responsibilities
The deployment intelligence, previously centralized, is strictly compartmentalized following the Single Responsibility Principle:
1. **DeploymentManager**: The high-level orchestrator coordinating the lifecycle.
2. **DeploymentManifest**: Defines schemas for `ReleaseMetadata`, `ArtifactManifest`, and `DeploymentHistory`.
3. **DeploymentValidator**: Performs pre-flight checks (container validation, compose validation, environment schema validation).
4. **DeploymentHealth**: Manages Startup, Readiness, and Liveness probes, augmented by **Synthetic Service Validation** (post-deployment mock transactions verifying the ML inference engine).
5. **DeploymentBackup**: Enforces `BackupManifest` creation and cryptographic validation *before* allowing data restoration.
6. **DeploymentRollback**: Deterministic state reconciliation triggered by health probe failures, network interruptions, or partial deployment states.

### 4. Advanced NGINX Reverse Proxy
NGINX acts as the robust ingress controller decoupling the Python backend and React SPA frontend from the internet.
- **Routing**: API requests (`/api/v1`) are reverse-proxied; SPA requests utilize `try_files $uri /index.html`.
- **Streaming**: Strict compatibility for Server-Sent Events (SSE) via `proxy_buffering off` and extended read timeouts. WebSockets are supported via `Upgrade` header propagation.
- **Security**: Injects strict HSTS, CSP, X-Frame-Options headers.
- **Resilience**: Basic `limit_req_zone` rate limiting mitigates volumetric abuse.
- **Performance**: Edge-level Gzip/Brotli compression applied to static assets and JSON payloads.

### 5. Infrastructure Layout
```text
deployment/
├── docker/
│   ├── Dockerfile
│   ├── Dockerfile.dev
│   ├── docker-compose.yml
│   ├── docker-compose.dev.yml
│   └── docker-compose.prod.yml
├── environments/
│   └── templates/
├── manifests/            # Tracks deployment state, history, and compatibility matrices
├── certificates/         # TLS lifecycle management
├── nginx/
│   └── nginx.conf        # Configured for SSE, WebSockets, SPA routing, and Rate Limiting
├── monitoring/
│   ├── prometheus.yml
│   └── alertmanager.yml  # Alerting abstraction based on SLA/SLO violations
├── release/              # Artifact packaging metadata
└── scripts/
    ├── build.sh
    ├── deploy.sh
    ├── rollback.sh
    └── backup.sh
```

---

## Deployment Lifecycle Flows

### 1. Verification & Pre-flight
- `DeploymentValidator` verifies environment variables against expected schemas.
- Compose syntax is validated natively.
- `VersionCompatibility` is verified against the target environment state.

### 2. Backup & Rollout
- `DeploymentBackup` captures database/state snapshots and generates a validated `BackupManifest`.
- Artifacts are deployed. Orchestration relies on native Docker healthchecks.

### 3. Health & Post-Deployment Validation
- **Startup Probes**: Validate ML weight loads.
- **Readiness Probes**: Validate database connectivity.
- **Liveness Probes**: Ensure event loop responsiveness.
- **Synthetic Validation**: `DeploymentHealth` executes a mock workflow ensuring end-to-end functionality before NGINX routes live traffic.

### 4. Failure Recovery & Rollback
- If Synthetic Validation or Readiness probes fail, `DeploymentRollback` initiates.
- Reverts container images to the previous `ReleaseManifest`.
- If state corruption is detected, restores data via the validated `BackupManifest`.

---

## CI/CD Integration
The framework exposes clean, decoupled shell interfaces for Phase 25 (CI/CD).
- Shell scripts strictly emit JSON-structured telemetry and standard POSIX exit codes (0 = success, >0 = specific failure state).
- The deployment framework is entirely agnostic of the CI/CD runner (e.g., GitHub Actions vs Jenkins), expecting only standard parameterized inputs.

---

## Architectural Deviations
None.

---

## Threat Model & Mitigations
- **Container Escape**: Mitigated by `read_only` filesystems, non-root execution, and `no-new-privileges`.
- **Secret Leakage**: Mitigated by strictly utilizing `SecretReference` paradigms; no secrets in `localStorage` or `.env` files in version control. 12-factor compliance enforced.
- **DDoS / Volumetric Attacks**: Mitigated by NGINX rate limiting and edge caching.
- **Data Corruption during Deploy**: Mitigated by strict `BackupManifest` cryptographic validation prior to any restore operation.
- **Incompatible Deployments**: Mitigated by explicit `VersionCompatibility` matrix checks preventing backend v2 from booting alongside frontend v1 if breaking changes exist.
