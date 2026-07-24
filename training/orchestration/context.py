from dataclasses import dataclass
from training.orchestration.artifacts import ArtifactRegistry
import logging

@dataclass(frozen=True)
class WorkflowContext:
    workflow_id: str
    execution_id: str
    artifact_root: str
    artifact_registry: ArtifactRegistry

@dataclass(frozen=True)
class NodeContext:
    workflow_id: str
    execution_id: str
    node_id: str
    correlation_id: str
    artifact_root: str
    artifact_registry: ArtifactRegistry
    logger: logging.Logger
