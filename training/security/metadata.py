from dataclasses import dataclass, field
from typing import Optional, Any, Tuple, Mapping
from types import MappingProxyType
from enum import Enum

class Severity(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

@dataclass(frozen=True)
class Identity:
    identity_id: str
    provider: str
    attributes: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

@dataclass(frozen=True)
class Principal:
    identity: Identity
    roles: Tuple[str, ...] = field(default_factory=tuple)

@dataclass(frozen=True)
class Role:
    name: str
    permissions: Tuple[str, ...] = field(default_factory=tuple)

@dataclass(frozen=True)
class Permission:
    name: str
    description: str

@dataclass(frozen=True)
class Capability:
    name: str
    required_permissions: Tuple[str, ...]

@dataclass(frozen=True)
class ResourceType:
    name: str

@dataclass(frozen=True)
class ResourceIdentifier:
    resource_type: ResourceType
    resource_id: str

@dataclass(frozen=True)
class ResourceAttributes:
    attributes: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

@dataclass(frozen=True)
class Resource:
    identifier: ResourceIdentifier
    attributes: ResourceAttributes

@dataclass(frozen=True)
class Policy:
    name: str
    effect: str # allow, deny
    conditions: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

@dataclass(frozen=True)
class SecurityDecision:
    allow: bool
    reason: str
    matched_role: Optional[str] = None
    matched_permission: Optional[str] = None
    matched_policy: Optional[str] = None
    evaluation_trace: Tuple[str, ...] = field(default_factory=tuple)
    severity: Severity = Severity.INFO

@dataclass(frozen=True)
class SecurityReport:
    workflow_id: str
    execution_id: str
    decisions: Tuple[SecurityDecision, ...] = field(default_factory=tuple)

@dataclass(frozen=True)
class AuditRecord:
    audit_id: str
    timestamp: float
    action: str
    principal_id: str
    resource_id: Optional[str]
    decision: SecurityDecision
    context_data: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
