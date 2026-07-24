from typing import Dict, Type, Any

class BaseRegistry:
    """O(1) dictionary-backed registry for global plugin tracking."""
    
    _registry: Dict[str, Type[Any]] = {}
    
    @classmethod
    def register(cls, name: str, plugin_class: Type[Any]) -> None:
        if name in cls._registry:
            raise ValueError(f"Plugin '{name}' is already registered.")
        cls._registry[name] = plugin_class
        
    @classmethod
    def get(cls, name: str) -> Type[Any]:
        if name not in cls._registry:
            raise KeyError(f"Plugin '{name}' not found in registry.")
        return cls._registry[name]
        
    @classmethod
    def get_all(cls) -> Dict[str, Type[Any]]:
        return cls._registry.copy()
        
    @classmethod
    def clear(cls) -> None:
        cls._registry.clear()
