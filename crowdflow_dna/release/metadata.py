from dataclasses import dataclass
from typing import Tuple, Mapping

@dataclass(frozen=True)
class BuildMetadata:
    builder_id: str
    timestamp: str

@dataclass(frozen=True)
class ArtifactMetadata:
    name: str
    type: str
    path: str
    sha256: str
    sha512: str

@dataclass(frozen=True)
class DistributionMetadata:
    targets: Tuple[str, ...]

@dataclass(frozen=True)
class SBOMMetadata:
    version: str
    dependencies: Tuple[Mapping[str, str], ...]

@dataclass(frozen=True)
class ReleaseMetadata:
    version: str
    commit: str
    build: BuildMetadata
    artifacts: Tuple[ArtifactMetadata, ...]
    distribution: DistributionMetadata
