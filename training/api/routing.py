from typing import Callable, Any
from training.api.registry import EndpointRegistry

class Router:
    @staticmethod
    def add_route(path: str, method: str, handler: Callable):
        EndpointRegistry.register(f"{method} {path}", handler)

    @staticmethod
    def route(request: Any) -> Any:
        # Mock routing
        pass
