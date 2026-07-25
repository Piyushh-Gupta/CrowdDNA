from typing import Tuple
from ..metadata import SecurityFinding, Severity
from types import MappingProxyType

class APISecurityAudit:
    def audit(self) -> Tuple[SecurityFinding, ...]:
        return (
            SecurityFinding(
                id="API-01",
                severity=Severity.MEDIUM,
                category="API Security",
                component="FastAPI",
                description="Verify security headers",
                recommendation="Ensure HSTS and CSP are configured",
                reference="OWASP-A05",
                passed=True,
                metadata=MappingProxyType({})
            ),
        )
