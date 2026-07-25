from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, Mapping
from types import MappingProxyType

@dataclass(frozen=True)
class ApiError:
    code: str
    message: str
    details: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

@dataclass(frozen=True)
class ApiResponse:
    success: bool
    data: Optional[Any]
    errors: Tuple[ApiError, ...]
    metadata: Mapping[str, Any]
    request_id: str
    correlation_id: str
    timestamp: float
