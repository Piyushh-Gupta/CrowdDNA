from typing import Dict, Type
from training.framework.metadata import PluginMetadata
from training.framework.plugins.base import BasePlugin

class PluginManifestRegistry:
    """O(1) dictionary-backed registry for plugin manifests (metadata)."""
    
    _registry: Dict[str, PluginMetadata] = {}
    _classes: Dict[str, Type[BasePlugin]] = {}
    
    @classmethod
    def register(cls, metadata: PluginMetadata, plugin_class: Type[BasePlugin]) -> None:
        if metadata.name in cls._registry:
            raise ValueError(f"Plugin '{metadata.name}' is already registered.")
        cls._registry[metadata.name] = metadata
        cls._classes[metadata.name] = plugin_class
        
    @classmethod
    def get_metadata(cls, name: str) -> PluginMetadata:
        if name not in cls._registry:
            raise KeyError(f"Plugin '{name}' not found.")
        return cls._registry[name]
        
    @classmethod
    def get_class(cls, name: str) -> Type[BasePlugin]:
        if name not in cls._classes:
            raise KeyError(f"Plugin '{name}' not found.")
        return cls._classes[name]
        
    @classmethod
    def get_all_metadata(cls) -> Dict[str, PluginMetadata]:
        return cls._registry.copy()
        
    @classmethod
    def clear(cls) -> None:
        cls._registry.clear()
        cls._classes.clear()
