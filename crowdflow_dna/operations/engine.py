import time
import threading
from .context import OperationsState, OperationsContext
from types import MappingProxyType

class InvalidStateTransitionError(Exception):
    pass

class OperationsEngine:
    def __init__(self, registry):
        self.registry = registry
        self.state = OperationsState.NORMAL
        self.lock = threading.Lock()
        
    def _validate_transition(self, new_state: OperationsState):
        valid_transitions = {
            OperationsState.NORMAL: [OperationsState.DEGRADED, OperationsState.MAINTENANCE_REQUESTED],
            OperationsState.DEGRADED: [OperationsState.NORMAL, OperationsState.MAINTENANCE_REQUESTED],
            OperationsState.MAINTENANCE_REQUESTED: [OperationsState.DRAINING, OperationsState.NORMAL],
            OperationsState.DRAINING: [OperationsState.MAINTENANCE, OperationsState.NORMAL],
            OperationsState.MAINTENANCE: [OperationsState.RECOVERING],
            OperationsState.RECOVERING: [OperationsState.NORMAL, OperationsState.DEGRADED]
        }
        if new_state not in valid_transitions[self.state] and new_state != self.state:
            raise InvalidStateTransitionError(f"Cannot transition from {self.state} to {new_state}")

    def transition_state(self, new_state: OperationsState):
        with self.lock:
            if new_state == self.state:
                return self.state
            self._validate_transition(new_state)
            self.state = new_state
            return self.state

    def get_context(self, operator: str = "system") -> OperationsContext:
        with self.lock:
            current_state = self.state
        return OperationsContext(
            operation_id=f"op_{time.time()}",
            correlation_id="corr_1",
            operator_identity=operator,
            timestamp=time.time(),
            maintenance_state=current_state,
            slo_snapshot=MappingProxyType({}),
            execution_metadata=MappingProxyType({})
        )
