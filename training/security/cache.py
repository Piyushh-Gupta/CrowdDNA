from typing import Dict, Optional
from training.security.metadata import Identity, SecurityDecision
import threading

class AuthenticationCache:
    def __init__(self):
        self._cache: Dict[str, Identity] = {}
        self._lock = threading.Lock()

    def get(self, token: str) -> Optional[Identity]:
        with self._lock:
            return self._cache.get(token)

    def set(self, token: str, identity: Identity):
        with self._lock:
            self._cache[token] = identity

    def clear(self):
        with self._lock:
            self._cache.clear()

class AuthorizationCache:
    def __init__(self):
        self._cache: Dict[str, SecurityDecision] = {}
        self._lock = threading.Lock()

    def _make_key(self, principal_id: str, action: str, resource_id: str) -> str:
        return f"{principal_id}::{action}::{resource_id}"

    def get(self, principal_id: str, action: str, resource_id: str) -> Optional[SecurityDecision]:
        with self._lock:
            return self._cache.get(self._make_key(principal_id, action, resource_id))

    def set(self, principal_id: str, action: str, resource_id: str, decision: SecurityDecision):
        with self._lock:
            self._cache[self._make_key(principal_id, action, resource_id)] = decision

    def clear(self):
        with self._lock:
            self._cache.clear()

class PolicyCache:
    def __init__(self):
        self._cache: Dict[str, SecurityDecision] = {}
        self._lock = threading.Lock()

    def _make_key(self, policy_name: str, context_hash: str) -> str:
        return f"{policy_name}::{context_hash}"

    def get(self, policy_name: str, context_hash: str) -> Optional[SecurityDecision]:
        with self._lock:
            return self._cache.get(self._make_key(policy_name, context_hash))

    def set(self, policy_name: str, context_hash: str, decision: SecurityDecision):
        with self._lock:
            self._cache[self._make_key(policy_name, context_hash)] = decision

    def clear(self):
        with self._lock:
            self._cache.clear()
