from typing import Any
from training.api.registry import ServiceRegistry

class DependencyContainer:
    @staticmethod
    def get_service(name: str) -> Any:
        return ServiceRegistry.get(name)
