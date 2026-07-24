from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, Any, Optional

class LifecycleState(Enum):
    UNINITIALIZED = auto()
    INITIALIZED = auto()
    CONFIGURED = auto()
    RUNNING = auto()
    STOPPING = auto()
    STOPPED = auto()
    SHUTDOWN = auto()

class IllegalStateTransitionError(Exception):
    pass

@dataclass(frozen=True)
class LifecycleEvent:
    timestamp: float
    plugin_name: str
    state: LifecycleState
    duration: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

class StateMachine:
    _VALID_TRANSITIONS = {
        LifecycleState.UNINITIALIZED: {LifecycleState.INITIALIZED},
        LifecycleState.INITIALIZED: {LifecycleState.CONFIGURED, LifecycleState.SHUTDOWN},
        LifecycleState.CONFIGURED: {LifecycleState.RUNNING, LifecycleState.SHUTDOWN},
        LifecycleState.RUNNING: {LifecycleState.STOPPING, LifecycleState.STOPPED}, # sometimes crashes go to STOPPED
        LifecycleState.STOPPING: {LifecycleState.STOPPED},
        LifecycleState.STOPPED: {LifecycleState.SHUTDOWN},
        LifecycleState.SHUTDOWN: set()
    }
    
    @classmethod
    def validate_transition(cls, current: LifecycleState, target: LifecycleState) -> None:
        if target not in cls._VALID_TRANSITIONS[current]:
            raise IllegalStateTransitionError(f"Cannot transition from {current.name} to {target.name}")
