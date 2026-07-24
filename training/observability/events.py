from training.observability.metadata import EventObservation, EventPriority
from training.observability.buffer import ObservationBuffer
from training.observability.clock import Clock

class EventBus:
    """Asynchronous Event Bus for emitting observations without blocking execution."""
    
    def __init__(self, buffer: ObservationBuffer):
        self._buffer = buffer
        
    def emit(self, name: str, priority: EventPriority, message: str, metadata: dict = None) -> None:
        """Emits an event into the buffer."""
        event = EventObservation(
            timestamp=Clock.now(),
            name=name,
            priority=priority,
            message=message,
            metadata=metadata or {}
        )
        self._buffer.add(event)
