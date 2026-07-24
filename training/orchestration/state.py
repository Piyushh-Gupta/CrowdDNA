from enum import Enum, auto

class NodeState(Enum):
    PENDING = auto()
    PREPARING = auto()
    RUNNING = auto()
    SUCCESS = auto()
    FAILED = auto()
    SKIPPED = auto()
    CANCELLED = auto()
    COMPENSATING = auto()
