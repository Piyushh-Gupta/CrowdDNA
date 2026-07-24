from collections import deque
from typing import List
from training.observability.metadata import Observation

class ObservationBuffer:
    """Bounded, thread-safe (mostly, due to GIL and deque atomicity) buffer for observations."""
    
    def __init__(self, maxlen: int = 10000):
        # deque is thread-safe for appends and pops in CPython
        self._buffer: deque = deque(maxlen=maxlen)
        
    def add(self, obs: Observation) -> None:
        """Adds a single observation to the buffer."""
        self._buffer.append(obs)
        
    def add_many(self, observations: List[Observation]) -> None:
        """Adds multiple observations to the buffer."""
        self._buffer.extend(observations)
        
    def flush(self) -> List[Observation]:
        """Flushes the buffer, returning all current observations and clearing it."""
        # Atomic swap for thread-safe flushing without dropped events
        old_buffer = self._buffer
        self._buffer = deque(maxlen=old_buffer.maxlen)
        return list(old_buffer)
        
    def __len__(self) -> int:
        return len(self._buffer)
