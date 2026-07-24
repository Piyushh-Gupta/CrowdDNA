from dataclasses import dataclass, field
from typing import Mapping, Any
from types import MappingProxyType

@dataclass(frozen=True)
class ConfigValue:
    value: Any
    provenance: str  # e.g., 'default', 'system', 'project', 'experiment', 'runtime', 'environment', 'cli'

@dataclass(frozen=True)
class ConfigurationBundle:
    configuration_version: str
    values: Mapping[str, ConfigValue] = field(default_factory=dict)
    
    def __post_init__(self):
        # Enforce true immutability at runtime
        object.__setattr__(self, 'values', MappingProxyType(dict(self.values)))
        

    def get(self, key: str, default: Any = None) -> Any:
        if key in self.values:
            return self.values[key].value
        return default

@dataclass(frozen=True)
class ResolvedConfiguration:
    bundle: ConfigurationBundle
