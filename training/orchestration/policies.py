from dataclasses import dataclass
from enum import Enum, auto

class FailureIsolationLevel(Enum):
    CRITICAL = auto()     # Aborts workflow immediately
    CASCADING = auto()    # Fails this node, skips downstream, continues independent branches
    ISOLATED = auto()     # Fails this node, workflow continues
    RECOVERABLE = auto()  # Employs retry policy

@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 0
    backoff_factor: float = 2.0
    initial_delay_seconds: float = 1.0

@dataclass(frozen=True)
class TimeoutPolicy:
    timeout_seconds: float = 3600.0

@dataclass(frozen=True)
class FailurePolicy:
    isolation_level: FailureIsolationLevel = FailureIsolationLevel.CRITICAL

@dataclass(frozen=True)
class ExecutionPolicy:
    max_parallel_nodes: int = 1
    abort_on_any_failure: bool = True
