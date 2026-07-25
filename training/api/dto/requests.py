from dataclasses import dataclass
from typing import Any, Mapping
from types import MappingProxyType

@dataclass(frozen=True)
class WorkflowExecutionRequest:
    workflow_name: str
    parameters: Mapping[str, Any] = MappingProxyType({})
