from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from types import MappingProxyType
from training.framework.metadata import FrameworkMetadata
from training.orchestration.policies import RetryPolicy, TimeoutPolicy, FailurePolicy, ExecutionPolicy
from training.orchestration.locks import ResourceLock
from training.orchestration.state import NodeState

@dataclass(frozen=True)
class NodeMetadata:
    node_id: str
    node_type: str
    dependencies: List[str] = field(default_factory=list)
    configuration: Dict[str, Any] = field(default_factory=dict)
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    timeout_policy: TimeoutPolicy = field(default_factory=TimeoutPolicy)
    failure_policy: FailurePolicy = field(default_factory=FailurePolicy)
    resource_locks: List[ResourceLock] = field(default_factory=list)
    
    def __post_init__(self):
        object.__setattr__(self, 'dependencies', tuple(self.dependencies))
        object.__setattr__(self, 'configuration', MappingProxyType(dict(self.configuration)))
        object.__setattr__(self, 'resource_locks', tuple(self.resource_locks))

@dataclass(frozen=True)
class WorkflowDefinition:
    workflow_name: str
    workflow_schema_version: str
    nodes: Dict[str, NodeMetadata]
    global_execution_policy: ExecutionPolicy = field(default_factory=ExecutionPolicy)
    
    def __post_init__(self):
        object.__setattr__(self, 'nodes', MappingProxyType(dict(self.nodes)))

@dataclass(frozen=True)
class WorkflowExecution:
    execution_id: str
    session_id: str
    artifact_tree_root: str
    framework_metadata: FrameworkMetadata

@dataclass(frozen=True)
class WorkflowManifest:
    definition: WorkflowDefinition
    execution: WorkflowExecution

@dataclass(frozen=True)
class ExecutionPlan:
    """Immutable representation of execution tiers and dependencies."""
    tiers: List[List[str]]  # List of lists of node_ids
    node_dependencies: Dict[str, List[str]] # node_id -> parent_ids
    
    def __post_init__(self):
        frozen_tiers = tuple(tuple(tier) for tier in self.tiers)
        object.__setattr__(self, 'tiers', frozen_tiers)
        frozen_deps = {k: tuple(v) for k, v in self.node_dependencies.items()}
        object.__setattr__(self, 'node_dependencies', MappingProxyType(frozen_deps))

class ExecutionCursor:
    """Mutable cursor for runtime state tracking."""
    def __init__(self):
        self.node_states: Dict[str, NodeState] = {}
        self.retry_counts: Dict[str, int] = {}
        self.completed_nodes: set = set()
        self.failed_nodes: set = set()
        self.skipped_nodes: set = set()
        
    def get_state(self, node_id: str) -> NodeState:
        return self.node_states.get(node_id, NodeState.PENDING)
        
    def update_state(self, node_id: str, state: NodeState):
        self.node_states[node_id] = state
        if state == NodeState.SUCCESS:
            self.completed_nodes.add(node_id)
        elif state == NodeState.FAILED:
            self.failed_nodes.add(node_id)
        elif state == NodeState.SKIPPED:
            self.skipped_nodes.add(node_id)

@dataclass(frozen=True)
class NodeResult:
    node_id: str
    status: NodeState
    artifacts_produced: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    
    def __post_init__(self):
        object.__setattr__(self, 'artifacts_produced', tuple(self.artifacts_produced))
        object.__setattr__(self, 'metrics', MappingProxyType(dict(self.metrics)))
