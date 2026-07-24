from dataclasses import dataclass, field
import time
from typing import Optional, Dict, Any

@dataclass(frozen=True)
class OrchestrationEvent:
    workflow_id: str
    execution_id: str
    correlation_id: str
    timestamp: float = field(default_factory=time.time)
    node_id: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)

class WorkflowEvent(OrchestrationEvent):
    pass

class NodeEvent(OrchestrationEvent):
    pass

class SchedulerEvent(OrchestrationEvent):
    pass

class CheckpointEvent(OrchestrationEvent):
    pass
