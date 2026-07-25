from dataclasses import dataclass
from typing import Optional
from training.security.metadata import SecurityDecision

@dataclass(frozen=True)
class SecurityEvent:
    event_id: str
    timestamp: float
    workflow_id: Optional[str]
    execution_id: Optional[str]
    correlation_id: Optional[str]

@dataclass(frozen=True)
class AuthenticationEvent(SecurityEvent):
    identity_id: str
    success: bool
    provider: str

@dataclass(frozen=True)
class AuthorizationEvent(SecurityEvent):
    principal_id: str
    action: str
    resource_id: str
    decision: SecurityDecision

@dataclass(frozen=True)
class PolicyEvent(SecurityEvent):
    principal_id: str
    action: str
    resource_id: str
    decision: SecurityDecision

@dataclass(frozen=True)
class SecretEvent(SecurityEvent):
    secret_id: str
    principal_id: str
    success: bool

@dataclass(frozen=True)
class AuditEvent(SecurityEvent):
    audit_id: str
    action: str
    principal_id: str

@dataclass(frozen=True)
class PluginSecurityEvent(SecurityEvent):
    plugin_name: str
    decision: SecurityDecision

@dataclass(frozen=True)
class WorkflowSecurityEvent(SecurityEvent):
    node_id: str
    decision: SecurityDecision
