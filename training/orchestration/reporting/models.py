from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from types import MappingProxyType

@dataclass(frozen=True)
class NodeReport:
    node_id: str
    status: str
    duration_seconds: float
    retries: int
    artifacts_produced: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        object.__setattr__(self, 'artifacts_produced', tuple(self.artifacts_produced))
        object.__setattr__(self, 'metrics', MappingProxyType(dict(self.metrics)))

@dataclass(frozen=True)
class ArtifactReport:
    artifact_id: str
    producer_node_id: str
    schema_version: str
    checksum: str
    location: str

@dataclass(frozen=True)
class FailureReport:
    node_id: str
    error_type: str
    error_message: str
    traceback: Optional[str] = None

@dataclass(frozen=True)
class PerformanceReport:
    total_duration_seconds: float
    lock_wait_times: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        object.__setattr__(self, 'lock_wait_times', MappingProxyType(dict(self.lock_wait_times)))

@dataclass(frozen=True)
class ExecutionReport:
    workflow_id: str
    execution_id: str
    status: str
    node_reports: List[NodeReport] = field(default_factory=list)
    artifact_reports: List[ArtifactReport] = field(default_factory=list)
    failure_reports: List[FailureReport] = field(default_factory=list)
    performance: PerformanceReport = field(default_factory=lambda: PerformanceReport(0.0))
    
    def __post_init__(self):
        object.__setattr__(self, 'node_reports', tuple(self.node_reports))
        object.__setattr__(self, 'artifact_reports', tuple(self.artifact_reports))
        object.__setattr__(self, 'failure_reports', tuple(self.failure_reports))
