import enum
from dataclasses import dataclass
from typing import Mapping, Tuple
from types import MappingProxyType

class Severity(enum.Enum):
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

@dataclass(frozen=True)
class SecurityFinding:
    id: str
    severity: Severity
    category: str
    component: str
    description: str
    recommendation: str
    reference: str
    passed: bool
    metadata: Mapping[str, str]

    def __post_init__(self):
        if not isinstance(self.metadata, MappingProxyType):
            object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

@dataclass(frozen=True)
class ControlResult:
    control_id: str
    passed: bool
    findings: Tuple[SecurityFinding, ...]

@dataclass(frozen=True)
class PolicyRequirement:
    policy_id: str
    description: str
    required_controls: Tuple[str, ...]

@dataclass(frozen=True)
class Control:
    id: str
    description: str
