from dataclasses import dataclass, field
from typing import Any, Optional, Mapping
from types import MappingProxyType
from training.security.metadata import Identity

@dataclass(frozen=True)
class SecuritySession:
    session_id: str
    identity: Identity
    created_at: float
    expires_at: float
    session_data: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

@dataclass(frozen=True)
class SecurityContext:
    workflow_id: str
    execution_id: str
    correlation_id: str
    session: Optional[SecuritySession] = None
    environment_data: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
