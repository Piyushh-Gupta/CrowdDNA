from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Any, Optional

class EventPriority(Enum):
    CRITICAL = 4
    HIGH = 3
    NORMAL = 2
    LOW = 1

class HealthSeverity(Enum):
    CRITICAL = 4
    ERROR = 3
    WARNING = 2
    INFO = 1

@dataclass(frozen=True)
class Observation:
    """Canonical base class for all observability data points."""
    timestamp: float

@dataclass(frozen=True)
class MetricObservation(Observation):
    """Observation representing a numerical metric."""
    name: str
    value: float
    unit: str
    tags: Mapping[str, str] = field(default_factory=dict)

@dataclass(frozen=True)
class EventObservation(Observation):
    """Observation representing a discrete system event."""
    name: str
    priority: EventPriority
    message: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class TraceObservation(Observation):
    """Observation representing a distributed tracing span."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    name: str
    duration_sec: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class HealthObservation(Observation):
    """Observation representing a component's health status."""
    component: str
    severity: HealthSeverity
    message: str
    is_healthy: bool

@dataclass(frozen=True)
class AlertObservation(Observation):
    """Observation triggered by a proactive alerting rule."""
    alert_name: str
    rule_name: str
    severity: EventPriority
    message: str
    trigger_value: float

@dataclass(frozen=True)
class MonitoringContext:
    """Runtime configuration and environmental metadata."""
    session_id: str
    enabled_collectors: tuple[str, ...]
    enabled_exporters: tuple[str, ...]
    enabled_tracing: bool
    sampling_interval: float
    retention_policy: str
    output_directory: str
    metadata: Mapping[str, str] = field(default_factory=dict)

@dataclass(frozen=True)
class PipelineSnapshot:
    """Point-in-time capture of the active pipeline graph."""
    timestamp: float
    active_nodes: tuple[str, ...]
    queue_depths: Mapping[str, int]

@dataclass(frozen=True)
class MonitoringSession:
    """Container representing a complete execution run."""
    session_id: str
    start_time: float
    context: MonitoringContext
