# Phase 20: Orchestration & Execution Framework Architecture

## Goal
The Orchestration Framework provides a declarative Directed Acyclic Graph (DAG) execution engine that acts as the single control plane for CrowdDNA, coordinating all downstream operations (training, evaluation, explainability, etc.) without hard-coding their dependencies.

## Architecture Splitting
The execution lifecycle is strictly isolated across five distinct actors:
1. **Engine (Coordinator)**: The public facade. Consumes the Manifest, routes state to the Validator, Planner, or RecoveryEngine, and hands the ExecutionPlan to the Scheduler.
2. **Validator (Pre-flight checks)**: Dedicated static validation engine that ensures the DAG is cycle-free, dependencies exist, and Node IDs are unique.
3. **Planner (Plan Builder)**: Translates a static `WorkflowDefinition` into an immutable `ExecutionPlan`, establishing topological execution tiers.
4. **Scheduler (Dispatcher)**: A pure queue manager. Pulls nodes using a mutable `ExecutionCursor`, evaluates the `RetryPolicy` and `FailurePolicy` on errors, handles `ResourceLock`s, and yields work to the Executor.
5. **Executor (Runner)**: Operates against the `BaseNode` contract (`prepare`, `execute`, `compensate`), trapping exceptions cleanly and returning explicit `NodeResult` payloads.

## State Management
- **WorkflowDefinition**: Pure, immutable graph topology. Includes `workflow_schema_version`.
- **WorkflowExecution**: The run-specific runtime state.
- **ExecutionPlan**: The strictly immutable generated execution blueprint.
- **ExecutionCursor**: The singular mutable object owning the runtime progression, retry queues, and completed nodes list.
- **RecoveryEngine**: Responsible exclusively for deserializing `checkpoint.json` dumps, verifying drift against the current `WorkflowDefinition`, and bootstrapping the `ExecutionCursor`.

## Registries & Extensibility
All abstractions are decoupled into registry lookups:
- **NodeRegistry**: Registers nodes across defined categories (Training, Evaluation, Validation, Reporting, Utility, Cleanup, Checkpoint).
- **PolicyRegistry**: Centralized access for `RetryPolicy`, `TimeoutPolicy`, `FailurePolicy`, and `ExecutionPolicy`.
- **ArtifactRegistry**: A canonical manifest of all produced artifacts containing ID, producer node, checksum, schema version, and storage location.

## Exceptions & Events
- Generic exceptions are banned in favor of strictly typed variants (e.g. `WorkflowCycleError`, `OrchestrationValidationError`).
- Emitted events (`WorkflowEvent`, `NodeEvent`, `SchedulerEvent`, `CheckpointEvent`) are strictly verified to contain `workflow_id`, `execution_id`, `node_id`, and `correlation_id` to integrate deeply with Phase 18 Observability without friction.
