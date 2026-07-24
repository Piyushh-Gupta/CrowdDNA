from dataclasses import dataclass
from typing import Dict, Type
from abc import ABC, abstractmethod

@dataclass(frozen=True)
class ModuleMetadata:
    name: str
    version: str
    schema_version: str
    category: str

class ObservabilityModule(ABC):
    """Base class for all observability modules (collectors, monitors, exporters)."""
    
    @abstractmethod
    def initialize(self) -> None:
        pass
        
    @abstractmethod
    def start(self) -> None:
        pass
        
    @abstractmethod
    def stop(self) -> None:
        pass
        
    @abstractmethod
    def shutdown(self) -> None:
        pass

class ObservabilityRegistry:
    """Registry for managing observability plugins and their lifecycles."""
    
    _registry: Dict[str, Type[ObservabilityModule]] = {}
    _metadata: Dict[str, ModuleMetadata] = {}

    @classmethod
    def register(cls, metadata: ModuleMetadata):
        def wrapper(module_cls: Type[ObservabilityModule]) -> Type[ObservabilityModule]:
            if metadata.name in cls._registry:
                raise ValueError(f"Module '{metadata.name}' is already registered.")
            cls._registry[metadata.name] = module_cls
            cls._metadata[metadata.name] = metadata
            return module_cls
        return wrapper

    @classmethod
    def get_module(cls, name: str) -> Type[ObservabilityModule]:
        if name not in cls._registry:
            raise ValueError(f"Module '{name}' is not registered.")
        return cls._registry[name]

    @classmethod
    def get_metadata(cls, name: str) -> ModuleMetadata:
        if name not in cls._metadata:
            raise ValueError(f"Metadata for module '{name}' not found.")
        return cls._metadata[name]

    @classmethod
    def list_modules(cls) -> Dict[str, ModuleMetadata]:
        return dict(cls._metadata)
