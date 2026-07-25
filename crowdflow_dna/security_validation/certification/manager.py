import threading
from datetime import datetime, timezone
from typing import Tuple
from .report import ProductionCertificationReport
from .score import calculate_score
from .gates import CertificationGate
from ..metadata import SecurityFinding

class CertificationManager:
    """Thread-safe certification manager."""
    
    def __init__(self, gate: CertificationGate):
        self._lock = threading.Lock()
        self._gate = gate
        self._reports = []

    def generate_report(self, findings: Tuple[SecurityFinding, ...]) -> ProductionCertificationReport:
        with self._lock:
            score = calculate_score(findings)
            _ = self._gate.evaluate(score)
            
            report = ProductionCertificationReport(
                score=score,
                critical_findings=tuple(f for f in findings if f.severity.name == "CRITICAL" and not f.passed),
                warnings=tuple(f for f in findings if f.severity.name in ["LOW", "MEDIUM"] and not f.passed),
                passed_controls=tuple(),
                failed_controls=tuple(),
                skipped_controls=tuple(),
                recommendations=tuple(["Review architecture."]),
                generated_timestamp=datetime.now(timezone.utc).isoformat(),
                framework_version="1.0.0"
            )
            self._reports.append(report)
            return report
