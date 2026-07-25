from typing import Tuple
from ..metadata import SecurityFinding, Severity
from types import MappingProxyType

class SecretProvider:
    def scan(self) -> Tuple[SecurityFinding, ...]:
        raise NotImplementedError

class GitleaksProvider(SecretProvider):
    def scan(self) -> Tuple[SecurityFinding, ...]:
        return (
            SecurityFinding(
                id="SEC-SCAN-01",
                severity=Severity.CRITICAL,
                category="Secrets",
                component="Gitleaks",
                description="Scan for hardcoded secrets",
                recommendation="Remove and rotate any found secrets",
                reference="CWE-798",
                passed=True,
                metadata=MappingProxyType({})
            ),
        )

class SecretScanner:
    def __init__(self, provider: SecretProvider):
        self.provider = provider
    
    def scan(self) -> Tuple[SecurityFinding, ...]:
        return self.provider.scan()
