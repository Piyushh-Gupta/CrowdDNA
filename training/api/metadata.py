from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class ApiMetadata:
    api_version: str = "v1"
    schema_version: str = "1.0.0"
    build_id: str = "unknown"
    git_commit: str = "unknown"
    supported_versions: Tuple[str, ...] = ("v1",)
    crowddna_version: str = "0.22.0"
