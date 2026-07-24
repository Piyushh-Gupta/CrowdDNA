"""
Data models for the experiment management subsystem.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass(frozen=True)
class Experiment:
    """Immutable representation of a completed experiment."""
    name: str
    path: str
    timestamp: str
    git_commit: str
    configuration: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    deployment: Dict[str, Any] = field(default_factory=dict)
    hardware: Dict[str, Any] = field(default_factory=dict)
    rank: Optional[int] = None

    def get_metric(self, key: str, default: Any = "N/A") -> Any:
        return self.metrics.get(key, default)
