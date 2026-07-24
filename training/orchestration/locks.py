from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class ResourceLock:
    resource_id: str
    resource_type: str  # e.g., 'GPU', 'Dataset', 'Artifact', 'Disk'
    exclusive: bool = True
    timeout_seconds: Optional[float] = None
