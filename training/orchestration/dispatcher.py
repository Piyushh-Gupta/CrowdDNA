from typing import List, Callable
from training.orchestration.events import OrchestrationEvent

class EventDispatcher:
    _listeners: List[Callable[[OrchestrationEvent], None]] = []
    
    @classmethod
    def subscribe(cls, listener: Callable[[OrchestrationEvent], None]):
        cls._listeners.append(listener)
        
    @classmethod
    def clear(cls):
        cls._listeners.clear()
        
    @classmethod
    def dispatch(cls, event: OrchestrationEvent):
        for listener in cls._listeners:
            try:
                listener(event)
            except Exception:
                # Do not let a rogue listener crash orchestration
                pass
