"""
Reporting Registry.

Provides a registry for report generators.
"""
from typing import Dict, Type, Any

class ReporterRegistry:
    """Registry for managing and instantiating reporters."""
    
    _registry: Dict[str, Type[Any]] = {}

    @classmethod
    def register(cls, name: str):
        def wrapper(reporter_cls: Type[Any]):
            if name in cls._registry:
                raise ValueError(f"Reporter '{name}' is already registered.")
            cls._registry[name] = reporter_cls
            return reporter_cls
        return wrapper

    @classmethod
    def get_reporter(cls, name: str) -> Type[Any]:
        if name not in cls._registry:
            raise ValueError(f"Reporter '{name}' not found.")
        return cls._registry[name]
