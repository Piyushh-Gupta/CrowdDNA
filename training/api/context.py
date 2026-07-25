from dataclasses import dataclass, field
from typing import Tuple, Optional

@dataclass(frozen=True)
class APIContext:
    request_id: str
    correlation_id: str
    client_ip: str
    user_agent: str
    request_start_timestamp: float
    api_version: str
    identity: Optional[str] = None
    permissions: Tuple[str, ...] = field(default_factory=tuple)
    idempotency_key: Optional[str] = None
