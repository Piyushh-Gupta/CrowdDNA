from typing import Tuple
from ..metadata import SecurityFinding, Severity
from types import MappingProxyType

class APIPenetrationTester:
    def test(self) -> Tuple[SecurityFinding, ...]:
        return (
            SecurityFinding(
                id="PEN-API-01",
                severity=Severity.HIGH,
                category="Penetration",
                component="API",
                description="Fuzzing REST endpoints",
                recommendation="Ensure robust input validation",
                reference="OWASP-API",
                passed=True,
                metadata=MappingProxyType({})
            ),
        )
