from dataclasses import dataclass, field
from typing import Mapping, Optional, FrozenSet
from types import MappingProxyType
from enum import Enum, auto

class HealthReport(Enum):
    HEALTHY = auto()
    DEGRADED = auto()
    UNAVAILABLE = auto()
    FAILED = auto()

@dataclass(frozen=True)
class VersionConstraint:
    min_version: str
    max_version: Optional[str] = None
    exact_version: Optional[str] = None

@dataclass(frozen=True)
class FrameworkMetadata:
    framework_version: str
    crowddna_version: str

@dataclass(frozen=True)
class PluginMetadata:
    name: str
    version: str
    schema_version: str
    api_version: str
    category: str
    capabilities: FrozenSet[str]
    entrypoint: str
    dependencies: Mapping[str, VersionConstraint] = field(default_factory=dict)
    optional_dependencies: Mapping[str, VersionConstraint] = field(default_factory=dict)
    author: Optional[str] = None
    description: Optional[str] = None

    def __post_init__(self):
        object.__setattr__(self, 'dependencies', MappingProxyType(dict(self.dependencies)))
        object.__setattr__(self, 'optional_dependencies', MappingProxyType(dict(self.optional_dependencies)))

@dataclass(frozen=True)
class FrameworkRuntime:
    session_id: str
    start_time: float
    metadata: FrameworkMetadata
