from typing import Dict, Any
import threading

class SecurityRegistry:
    _auth_providers: Dict[str, Any] = {}
    _secret_providers: Dict[str, Any] = {}
    _policy_compilers: Dict[str, Any] = {}
    _lock = threading.Lock()

    @classmethod
    def register_auth_provider(cls, name: str, provider: Any):
        with cls._lock:
            cls._auth_providers[name] = provider

    @classmethod
    def get_auth_provider(cls, name: str) -> Any:
        with cls._lock:
            if name not in cls._auth_providers:
                raise KeyError(f"Auth provider '{name}' not found.")
            return cls._auth_providers[name]

    @classmethod
    def register_secret_provider(cls, name: str, provider: Any):
        with cls._lock:
            cls._secret_providers[name] = provider

    @classmethod
    def get_secret_provider(cls, name: str) -> Any:
        with cls._lock:
            if name not in cls._secret_providers:
                raise KeyError(f"Secret provider '{name}' not found.")
            return cls._secret_providers[name]

    @classmethod
    def clear(cls):
        with cls._lock:
            cls._auth_providers.clear()
            cls._secret_providers.clear()
            cls._policy_compilers.clear()
