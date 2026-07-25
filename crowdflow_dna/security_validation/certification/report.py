import json
from dataclasses import dataclass, asdict
from typing import Tuple
from .score import CertificationScore
from ..metadata import SecurityFinding, ControlResult

@dataclass(frozen=True)
class ProductionCertificationReport:
    score: CertificationScore
    critical_findings: Tuple[SecurityFinding, ...]
    warnings: Tuple[SecurityFinding, ...]
    passed_controls: Tuple[ControlResult, ...]
    failed_controls: Tuple[ControlResult, ...]
    skipped_controls: Tuple[ControlResult, ...]
    recommendations: Tuple[str, ...]
    generated_timestamp: str
    framework_version: str

    def to_json(self) -> str:
        # Deterministic JSON serialization
        return json.dumps(asdict(self), sort_keys=True)
