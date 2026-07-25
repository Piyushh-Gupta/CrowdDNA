from enum import Enum
from dataclasses import dataclass
from typing import Mapping, Any

class OperationsState(Enum):
    NORMAL = "NORMAL"
    DEGRADED = "DEGRADED"
    MAINTENANCE_REQUESTED = "MAINTENANCE_REQUESTED"
    DRAINING = "DRAINING"
    MAINTENANCE = "MAINTENANCE"
    RECOVERING = "RECOVERING"

@dataclass(frozen=True)
class OperationsContext:
    operation_id: str
    correlation_id: str
    operator_identity: str
    timestamp: float
    maintenance_state: OperationsState
    slo_snapshot: Mapping[str, Any]
    execution_metadata: Mapping[str, Any]
