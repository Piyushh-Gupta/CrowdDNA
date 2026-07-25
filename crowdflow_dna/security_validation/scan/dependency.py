from typing import Tuple
from ..metadata import SecurityFinding, Severity
from types import MappingProxyType

class DependencyProvider:
    def scan(self) -> Tuple[SecurityFinding, ...]:
        raise NotImplementedError

class TrivyProvider(DependencyProvider):
    def scan(self) -> Tuple[SecurityFinding, ...]:
        return (
            SecurityFinding(
                id="DEP-SCAN-01",
                severity=Severity.LOW,
                category="Dependency",
                component="Trivy",
                description="Dependency vulnerability scan",
                recommendation="Upgrade packages",
                reference="CVE-LIST",
                passed=True,
                metadata=MappingProxyType({})
            ),
        )

class DependencyScanner:
    def __init__(self, provider: DependencyProvider):
        self.provider = provider
    
    def scan(self) -> Tuple[SecurityFinding, ...]:
        return self.provider.scan()
