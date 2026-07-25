import time
from collections import OrderedDict
from typing import Any, Optional
import hashlib
import json
import threading

from crowdflow_dna.config import (
    PERFORMANCE_CACHE_MAX_SIZE,
    PERFORMANCE_CACHE_TTL_SECONDS
)

class CacheManager:
    def __init__(self, max_size: Optional[int] = None, ttl_seconds: Optional[int] = None):
        self.max_size = max_size if max_size is not None else PERFORMANCE_CACHE_MAX_SIZE
        self.ttl_seconds = ttl_seconds if ttl_seconds is not None else PERFORMANCE_CACHE_TTL_SECONDS
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.lock = threading.Lock()

    def _generate_key(self, raw_key: dict) -> str:
        # Deterministic cache key generation using reproducibility hash principles
        sorted_json = json.dumps(raw_key, sort_keys=True)
        return hashlib.sha256(sorted_json.encode('utf-8')).hexdigest()

    def set(self, raw_key: dict, value: Any):
        key = self._generate_key(raw_key)
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = (time.time(), value)
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)

    def get(self, raw_key: dict) -> Optional[Any]:
        key = self._generate_key(raw_key)
        with self.lock:
            if key in self.cache:
                timestamp, value = self.cache[key]
                if time.time() - timestamp > self.ttl_seconds:
                    del self.cache[key]
                    self.misses += 1
                    return None
                self.cache.move_to_end(key)
                self.hits += 1
                return value
            self.misses += 1
            return None

    def invalidate(self, raw_key: dict):
        key = self._generate_key(raw_key)
        with self.lock:
            if key in self.cache:
                del self.cache[key]

    def stats(self):
        with self.lock:
            return {"hits": self.hits, "misses": self.misses, "size": len(self.cache)}