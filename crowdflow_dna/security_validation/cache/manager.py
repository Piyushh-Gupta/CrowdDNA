import threading
from typing import Any

class CacheManager:
    """Thread-safe deterministic cache manager."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._cache = {}

    def get(self, key: str) -> Any:
        with self._lock:
            return self._cache.get(key)

    def set(self, key: str, value: Any):
        with self._lock:
            self._cache[key] = value
