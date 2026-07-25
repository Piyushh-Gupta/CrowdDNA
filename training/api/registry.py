import threading
from typing import Dict, Any

class EndpointRegistry:
    _endpoints: Dict[str, Any] = {}
    _lock = threading.Lock()

    @classmethod
    def register(cls, name: str, handler: Any):
        with cls._lock:
            cls._endpoints[name] = handler

    @classmethod
    def get(cls, name: str) -> Any:
        with cls._lock:
            return cls._endpoints.get(name)

class ServiceRegistry:
    _services: Dict[str, Any] = {}
    _lock = threading.Lock()

    @classmethod
    def register(cls, name: str, service: Any):
        with cls._lock:
            cls._services[name] = service

    @classmethod
    def get(cls, name: str) -> Any:
        with cls._lock:
            return cls._services.get(name)

class MiddlewareRegistry:
    _middlewares: Dict[str, Any] = {}
    _lock = threading.Lock()

    @classmethod
    def register(cls, name: str, middleware: Any):
        with cls._lock:
            cls._middlewares[name] = middleware

    @classmethod
    def get(cls, name: str) -> Any:
        with cls._lock:
            return cls._middlewares.get(name)

class IdempotencyRegistry:
    _responses: Dict[str, Any] = {}
    _lock = threading.Lock()

    @classmethod
    def store(cls, key: str, fingerprint: str, response: Any):
        with cls._lock:
            cls._responses[key] = (fingerprint, response)

    @classmethod
    def get(cls, key: str) -> Any:
        with cls._lock:
            return cls._responses.get(key)
