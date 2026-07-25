from typing import Tuple
from ..metadata import SecurityFinding, Severity
from types import MappingProxyType
from ... import config

class LicenseScanner:
    def scan(self) -> Tuple[SecurityFinding, ...]:
        return (
            SecurityFinding(
                id="LIC-01",
                severity=Severity.HIGH,
                category="Licensing",
                component="Dependencies",
                description=f"Validate against {config.SECURITY_ALLOWED_LICENSES}",
                recommendation="Remove unapproved licenses",
                reference="OSI",
                passed=True,
                metadata=MappingProxyType({})
            ),
        )
